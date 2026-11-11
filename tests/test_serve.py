"""API tests of ornith-serve without the model weights: a fake engine with the same interface records every token it reads.
Its 'state' is a running hash of (position, token), its KV cache stores the token of each position, so after every request
we can check that the prefix cache restored exactly the state of the conversation (nothing lost, nothing read twice).
Run: python test_serve.py   (needs only the tokenizer files in ../ornith/model)"""
import json, os, sys, threading, time, urllib.request
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.isfile(os.path.join(HERE, 'ornith_serve.py')) else os.path.dirname(HERE)   # repository: tests/ below the server
sys.path.insert(0, ROOT)
import ornith_serve as osv

from fake_engine import FakeEngine as _FE, FakePrefill, FakeVision, state_hash, tokenizer

TOK = tokenizer()
END = TOK.convert_tokens_to_ids('<|im_end|>')
PAD = TOK.convert_tokens_to_ids('<|image_pad|>')


def FakeEngine(): return _FE(TOK)


E = FakeEngine()
a = osv.parse_args(['--config', 'no-such-config.json', '--port', '18731', '--snapshots', '', '--temperature', '0', '--park-gb', '1', '--ctx', '65536'])
SRV, HTTPD = osv.serve(a, E, FakePrefill(E), TOK)
threading.Thread(target=HTTPD.serve_forever, daemon=True).start()
SRV.ready.wait(10)
URL = f'http://127.0.0.1:{a.port}'
FAILS = []


def check(name, cond, info=''):
    print(('ok   ' if cond else 'FAIL ') + name + (f'  [{info}]' if info and not cond else ''))
    if not cond: FAILS.append(name)


def consistent(name):
    """the engine holds exactly the server's context: KV positions, recurrent state and token count"""
    n = len(SRV.ctx)
    real = [PAD if t < 0 else t for t in SRV.ctx]                  # ctx holds cache keys: image pads are negative
    good = (int(E.conv_state) == n and int(E.S) == state_hash(real) and E.kv()[:n].tolist() == real)
    check(name + ': engine state == context', good, f'n={n} conv={int(E.conv_state)}')


def post(path, body, headers=None):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json', **(headers or {})})
    with urllib.request.urlopen(req) as r: return json.loads(r.read())


def stream(path, body):
    req = urllib.request.Request(URL + path, json.dumps(dict(body, stream=True)).encode(), {'Content-Type': 'application/json'})
    events = []
    with urllib.request.urlopen(req) as r:
        for line in r.read().decode().splitlines():
            if line.startswith('data: '):
                d = line[6:]
                events.append(d if d == '[DONE]' else json.loads(d))
    return events


def last_log():
    return dict(cached=SRV.stats['cached_tokens'], prompt=SRV.stats['prompt_tokens'])


SYS = {'role': 'system', 'content': 'You are a helpful assistant.'}
TOOLS = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'read a file',
                                           'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'limit': {'type': 'integer'}}}}}]

# 1 plain answer with reasoning
E.say('Let me think.\n</think>\n\nHallo!')
r = post('/v1/chat/completions', {'model': 'x', 'messages': [SYS, {'role': 'user', 'content': 'Hi'}]})
m = r['choices'][0]['message']
check('openai: content', m['content'] == 'Hallo!', m)
check('openai: reasoning_content', m.get('reasoning_content') == 'Let me think.', m)
check('openai: finish stop', r['choices'][0]['finish_reason'] == 'stop')
consistent('turn 1')

# 2 next turn: the client sends the answer back without reasoning -> must reuse the prompt-end checkpoint
p_before = SRV.stats['prompt_tokens']; c_before = SRV.stats['cached_tokens']
E.say('ok\n</think>\n\nZweite Antwort.')
msgs = [SYS, {'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hallo!'}, {'role': 'user', 'content': 'Und?'}]
r = post('/v1/chat/completions', {'messages': msgs})
check('turn 2: content', r['choices'][0]['message']['content'] == 'Zweite Antwort.', r)
check('turn 2: prompt cache used', r['usage']['prompt_tokens_details']['cached_tokens'] > 10, r['usage'])
consistent('turn 2')

# 3 tool call
E.say('need file\n</think>\n\nIch lese die Datei.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>')
msgs3 = [SYS, {'role': 'user', 'content': 'Lies a.txt'}]
r = post('/v1/chat/completions', {'messages': msgs3, 'tools': TOOLS})
m = r['choices'][0]['message']
tc = (m.get('tool_calls') or [{}])[0].get('function', {})
check('tools: finish tool_calls', r['choices'][0]['finish_reason'] == 'tool_calls', r)
check('tools: name + typed arguments', tc.get('name') == 'read_file' and json.loads(tc.get('arguments', '{}')) == {'path': 'a.txt', 'limit': 5}, tc)
check('tools: text before call', m['content'] == 'Ich lese die Datei.', m)
consistent('tool call')

# 4 tool result goes back
E.say('done\n</think>\n\nDie Datei enthält 42.')
msgs4 = msgs3 + [{'role': 'assistant', 'content': m['content'], 'tool_calls': m['tool_calls']},
                 {'role': 'tool', 'tool_call_id': m['tool_calls'][0]['id'], 'content': '42'}]
r = post('/v1/chat/completions', {'messages': msgs4, 'tools': TOOLS})
check('tool result: answer', r['choices'][0]['message']['content'] == 'Die Datei enthält 42.', r)
check('tool result: cache used', r['usage']['prompt_tokens_details']['cached_tokens'] > 50, r['usage'])
consistent('tool result')

# 5 streaming (OpenAI)
E.say('hm\n</think>\n\nStream <b>ok</b>.')
ev = stream('/v1/chat/completions', {'messages': [SYS, {'role': 'user', 'content': 'stream'}], 'stream_options': {'include_usage': True}})
txt = ''.join(e['choices'][0]['delta'].get('content') or '' for e in ev if e != '[DONE]' and e['choices'])
rs = ''.join(e['choices'][0]['delta'].get('reasoning_content') or '' for e in ev if e != '[DONE]' and e['choices'])
check('stream: content', txt == 'Stream <b>ok</b>.', repr(txt))
check('stream: reasoning', rs.strip() == 'hm', repr(rs))
check('stream: usage chunk + DONE', ev[-1] == '[DONE]' and 'usage' in ev[-2], ev[-2:])
consistent('stream')

# 6 streaming tool call (OpenAI)
E.say('x\n</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nb.txt\n</parameter>\n</function>\n</tool_call>')
ev = stream('/v1/chat/completions', {'messages': [SYS, {'role': 'user', 'content': 'b'}], 'tools': TOOLS})
calls = [c for e in ev if e != '[DONE]' and e['choices'] for c in e['choices'][0]['delta'].get('tool_calls') or []]
fins = [e['choices'][0]['finish_reason'] for e in ev if e != '[DONE]' and e['choices'] and e['choices'][0]['finish_reason']]
txt = ''.join(e['choices'][0]['delta'].get('content') or '' for e in ev if e != '[DONE]' and e['choices'])
check('stream tools: call', len(calls) == 1 and json.loads(calls[0]['function']['arguments']) == {'path': 'b.txt'}, calls)
check('stream tools: no XML leaks into content', '<tool' not in txt and fins == ['tool_calls'], (txt, fins))

# 7 Anthropic messages + tool_use + tool_result
E.say('t\n</think>\n\nSchaue nach.\n<tool_call>\n<function=read_file>\n<parameter=path>\nc.txt\n</parameter>\n</function>\n</tool_call>')
atools = [{'name': 'read_file', 'description': 'read', 'input_schema': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}]
ab = {'model': 'claude-x', 'max_tokens': 1000, 'system': 'You are helpful.', 'tools': atools, 'messages': [{'role': 'user', 'content': 'c lesen'}]}
r = post('/v1/messages', ab)
kinds = [b['type'] for b in r['content']]
check('anthropic: blocks thinking/text/tool_use', kinds == ['thinking', 'text', 'tool_use'], kinds)
check('anthropic: stop_reason tool_use', r['stop_reason'] == 'tool_use', r['stop_reason'])
tu = r['content'][2]
E.say('u\n</think>\n\nInhalt: 7.')
ab2 = dict(ab, messages=ab['messages'] + [{'role': 'assistant', 'content': r['content']},
                                          {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': tu['id'], 'content': '7'}]}])
r2 = post('/v1/messages', ab2)
check('anthropic: answer after tool_result', r2['content'][-1] == {'type': 'text', 'text': 'Inhalt: 7.'}, r2['content'])
check('anthropic: cache_read_input_tokens', r2['usage']['cache_read_input_tokens'] > 20, r2['usage'])
consistent('anthropic')

# 8 Anthropic streaming
E.say('v\n</think>\n\nHallo Stream.')
ev = stream('/v1/messages', dict(ab, tools=[], messages=[{'role': 'user', 'content': 'hi'}]))
types = [e['type'] for e in ev if e != '[DONE]']
text = ''.join(e['delta'].get('text', '') for e in ev if e != '[DONE]' and e['type'] == 'content_block_delta')
check('anthropic stream: event order', types[0] == 'message_start' and types[-1] == 'message_stop' and 'message_delta' in types, types)
check('anthropic stream: text', text == 'Hallo Stream.', repr(text))

# 9 switch between two conversations: A is parked exactly in RAM and comes back without a new read-in
conv_a = [SYS, {'role': 'user', 'content': 'Gespräch A ' + 'lang ' * 300}]
E.say('a\n</think>\n\nA1')
post('/v1/chat/completions', {'messages': conv_a})
E.say('b\n</think>\n\nB1')
post('/v1/chat/completions', {'messages': [SYS, {'role': 'user', 'content': 'Gespräch B, etwas ganz anderes'}]})
consistent('conversation B')
steps0 = E.steps
E.say('a2\n</think>\n\nA2')
r = post('/v1/chat/completions', {'messages': conv_a + [{'role': 'assistant', 'content': 'A1'}, {'role': 'user', 'content': 'weiter'}]})
check('switch back: answer', r['choices'][0]['message']['content'] == 'A2', r)
check('switch back: A restored from RAM park', r['usage']['prompt_tokens_details']['cached_tokens'] >= 300, r['usage'])
check('switch back: only the new tokens read', E.steps - steps0 < 40, E.steps - steps0)
consistent('switch back')

# 10 long shared system prompt: a new conversation of the same harness starts at its checkpoint
big = {'role': 'system', 'content': 'Regel: ' + ' '.join(f'Punkt {i} ist wichtig.' for i in range(400))}
E.say('s\n</think>\n\nS1')
post('/v1/chat/completions', {'messages': [big, {'role': 'user', 'content': 'erste Sitzung'}]})
E.say('s\n</think>\n\nS2')
post('/v1/chat/completions', {'messages': [SYS, {'role': 'user', 'content': 'dazwischen etwas anderes'}]})
steps0 = E.steps
E.say('t\n</think>\n\nS3')
r = post('/v1/chat/completions', {'messages': [big, {'role': 'user', 'content': 'zweite Sitzung, neues Thema'}]})
check('shared system prompt reused', r['usage']['prompt_tokens_details']['cached_tokens'] >= 1024 and E.steps - steps0 < 60,
      (r['usage'], E.steps - steps0))
consistent('shared system prompt')

# 11 limits, stop sequences, settings, misc endpoints
E.say('w\n</think>\n\nEins zwei drei vier fünf sechs sieben acht neun zehn.')
r = post('/v1/chat/completions', {'messages': [SYS, {'role': 'user', 'content': 'zähl'}], 'max_tokens': 6})
check('max_tokens -> finish length', r['choices'][0]['finish_reason'] == 'length' and r['usage']['completion_tokens'] == 6, r)
consistent('after cut answer')
E.say('w\n</think>\n\nEins zwei STOP drei.')
r = post('/v1/chat/completions', {'messages': [SYS, {'role': 'user', 'content': 'zähl2'}], 'stop': ['STOP']})
check('stop sequence', r['choices'][0]['message']['content'] == 'Eins zwei', r['choices'][0]['message'])
E.say('Ohne Denken.')
r = post('/v1/chat/completions', {'messages': [SYS, {'role': 'user', 'content': 'schnell'}], 'reasoning_effort': 'none'})
check('reasoning off', r['choices'][0]['message']['content'] == 'Ohne Denken.' and 'reasoning_content' not in r['choices'][0]['message'], r)
r = post('/v1/messages/count_tokens', {'messages': [{'role': 'user', 'content': 'hallo'}]})
check('count_tokens', r['input_tokens'] > 3, r)
s = post('/ornith/config', {'quality': 'exact'})
check('config: quality exact', s['quality'] == 'exact' and E.quality == 0.0, (s['quality'], E.quality))
E.say('q\n</think>\n\nQ')
post('/v1/chat/completions', {'messages': [SYS, {'role': 'user', 'content': 'q'}], 'ornith': {'quality': 'turbo'}})
check('per-request quality override', E.quality == -0.15)
with urllib.request.urlopen(URL + '/v1/models') as rr: mods = json.loads(rr.read())
check('models list', mods['data'][0]['id'] == osv.MODEL_ID)
with urllib.request.urlopen(URL + '/ornith/status') as rr: st = json.loads(rr.read())
check('status', st['state'] == 'ready' and st['requests'] > 10, st)
try:
    post('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'x ' * 70000}]}); code = 200
except urllib.error.HTTPError as ex:
    code = ex.code
check('too long prompt -> 400', code == 400, code)
SRV.a.api_key = 'k1'
try:
    post('/v1/chat/completions', {'messages': [SYS]}); code = 200
except urllib.error.HTTPError as ex:
    code = ex.code
check('api key required', code == 401, code)
SRV.a.api_key = ''


def status_code(path, body, raw=None, headers=None):
    try:
        req = urllib.request.Request(URL + path, raw if raw is not None else json.dumps(body).encode(), {'Content-Type': 'application/json', **(headers or {})})
        with urllib.request.urlopen(req) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as ex:
        return ex.code, json.loads(ex.read() or b'{}')


# 12 robustness: tool call cut by max_tokens is dropped, tool XML without tools stays text, odd sampling values
E.say('x\n</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nvery/long/path/name.txt\n</parameter>\n</function>\n</tool_call>')
r = post('/v1/chat/completions', {'messages': [SYS, {'role': 'user', 'content': 'cut'}], 'tools': TOOLS, 'max_tokens': 14})
check('cut tool call is not returned', not r['choices'][0]['message'].get('tool_calls') and r['choices'][0]['finish_reason'] == 'length', r['choices'][0])
consistent('cut tool call')
E.say('x\n</think>\n\nSo sieht ein Aufruf aus: <tool_call> ... </tool_call>')
r = post('/v1/chat/completions', {'messages': [SYS, {'role': 'user', 'content': 'xml'}]})
check('no tools: tool XML stays in the text', '<tool_call>' in r['choices'][0]['message']['content'], r['choices'][0]['message'])
E.say('x\n</think>\n\nGierig.')
r = post('/v1/chat/completions', {'messages': [SYS, {'role': 'user', 'content': 'g'}], 'temperature': 1.0, 'top_p': 0, 'top_k': 0})
check('top_p 0 -> greedy, no crash', r['choices'][0]['message']['content'] == 'Gierig.', r)
E.say('x\n</think>\n\n{"a": 1}')
r = post('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'json'}], 'response_format': {'type': 'json_object'}})
check('response_format adds a JSON instruction', 'valid JSON object' in TOK.decode(SRV.ctx[:60]), TOK.decode(SRV.ctx[:60]))
E.say('x\n</think>\n\nEins ENDE zwei')
r = post('/v1/messages', {'max_tokens': 100, 'messages': [{'role': 'user', 'content': 'stop'}], 'stop_sequences': ['ENDE']})
check('anthropic stop_sequence', r['stop_reason'] == 'stop_sequence' and r['stop_sequence'] == 'ENDE' and r['content'][-1]['text'] == 'Eins', r)

# 13 bad requests get a clear 400 (and nothing changes in the engine)
n0 = len(SRV.ctx)
for name, path, body in [('no messages', '/v1/chat/completions', {}), ('only system', '/v1/chat/completions', {'messages': [SYS]}),
                         ('bad role', '/v1/chat/completions', {'messages': [{'role': 'robot', 'content': 'x'}]}),
                         ('bad quality', '/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'x'}], 'ornith': {'quality': 'ultra'}}),
                         ('n=2', '/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'x'}], 'n': 2}),
                         ('bad temperature', '/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'x'}], 'temperature': 'hot'}),
                         ('anthropic bad role', '/v1/messages', {'messages': [{'role': 'system', 'content': 'x'}]}),
                         ('unknown setting', '/ornith/config', {'turbo_boost': True}), ('completion without prompt', '/v1/completions', {})]:
    code, b = status_code(path, body)
    check(f'400 for {name}', code == 400 and (b.get('error') or {}).get('message'), (code, b))
code, b = status_code('/v1/chat/completions', None, raw=b'{not json')
check('400 for broken JSON', code == 400, code)
code, b = status_code('/v1/nothing', {})
check('404 for unknown endpoint', code == 404, code)
check('bad requests leave the engine alone', len(SRV.ctx) == n0)

# 14 chunked request body (some clients send no Content-Length)
import http.client
conn = http.client.HTTPConnection('127.0.0.1', a.port)
E.say('x\n</think>\n\nChunked ok.')
conn.putrequest('POST', '/v1/chat/completions'); conn.putheader('Content-Type', 'application/json'); conn.putheader('Transfer-Encoding', 'chunked')
conn.endheaders()
data = json.dumps({'messages': [{'role': 'user', 'content': 'chunked'}]}).encode()
for i in range(0, len(data), 20): conn.send(b'%x\r\n' % len(data[i:i + 20]) + data[i:i + 20] + b'\r\n')
conn.send(b'0\r\n\r\n')
resp = conn.getresponse(); rb = json.loads(resp.read()); conn.close()
check('chunked body', rb['choices'][0]['message']['content'] == 'Chunked ok.', rb)

# 15 keep-alive pings while a long prompt is read in, then status works from another thread during generation
osv.KEEPALIVE_S = 0.3
E.delay = 0.0015
E.say('x\n</think>\n\nNach langem Einlesen.')
req = urllib.request.Request(URL + '/v1/chat/completions', json.dumps({'stream': True, 'messages': [{'role': 'user', 'content': 'lang ' * 1200}]}).encode(),
                             {'Content-Type': 'application/json'})
stat_err = []


def poll_status():
    for _ in range(40):
        try:
            with urllib.request.urlopen(URL + '/ornith/status') as rr: json.loads(rr.read())
        except Exception as ex:
            stat_err.append(repr(ex))
        time.sleep(0.05)
th = threading.Thread(target=poll_status); th.start()
with urllib.request.urlopen(req) as rr: raw_stream = rr.read().decode()
th.join()
check('keep-alive pings during prefill', ': ping' in raw_stream, raw_stream[:200])
check('stream still complete', 'Nach langem Einlesen.' in ''.join(json.loads(l[6:])['choices'][0]['delta'].get('content') or ''
                                                                  for l in raw_stream.splitlines() if l.startswith('data: {') and '"choices": [{' in l))
check('status during generation never fails', not stat_err, stat_err[:2])
E.delay = 0.0

# 16 client disconnects mid-answer -> generation stops, next request is fine
E.delay = 0.01
E.say('x\n</think>\n\n' + 'wort ' * 400)
conn = http.client.HTTPConnection('127.0.0.1', a.port)
conn.request('POST', '/v1/chat/completions', json.dumps({'stream': True, 'messages': [{'role': 'user', 'content': 'abbrechen'}]}), {'Content-Type': 'application/json'})
sock = conn.sock
resp = conn.getresponse(); resp.read1(200)
import socket
sock.shutdown(socket.SHUT_RDWR); sock.close(); resp.close(); conn.close()
t0 = time.time()
while SRV.queue_len and time.time() - t0 < 10: time.sleep(0.01)    # the cancelled answer ends (400 words would take > 4 s)
E.delay = 0.0
E.say('x\n</think>\n\nWieder da.')
r = post('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'danach'}]})
check('disconnect cancels the answer', time.time() - t0 < 3.5 and r['choices'][0]['message']['content'] == 'Wieder da.', time.time() - t0)
consistent('after disconnect')

# 17 config file + port already in use
import tempfile
cf = os.path.join(tempfile.mkdtemp(), 'c.json')
json.dump({'quality': 'fast', 'ram_gb': 6, 'think': False, 'port': 18799}, open(cf, 'w'))
a2 = osv.parse_args(['--config', cf, '--ram-gb', '8'])
check('config file + command line wins', a2.quality == 'fast' and a2.ram_gb == 8 and a2.think is False and a2.port == 18799, vars(a2))
json.dump({'qualty': 'fast'}, open(cf, 'w'))
try:
    osv.parse_args(['--config', cf]); ok = False
except SystemExit as ex:
    ok = 'unknown option' in str(ex)
check('config file typo is reported', ok)
try:
    osv.serve(osv.parse_args(['--config', 'none.json', '--port', str(a.port)]), E, FakePrefill(E), TOK); ok = False
except SystemExit as ex:
    ok = 'cannot listen' in str(ex)
check('busy port fails at once', ok)
code, b = status_code('/ornith/config', {'temperature': 0.3, 'max_tokens': 1000})
check('config: temperature + max_tokens', code == 200 and b['temperature'] == 0.3 and b['max_tokens'] == 1000, b)
post('/ornith/config', {'temperature': 0.0, 'max_tokens': 32768})

# 18b images: encoded, expanded, cached by content (a different image never reuses the old one)
import base64, io
from PIL import Image
SRV.vision = FakeVision()


def img_url(color, w=300, h=200):
    b = io.BytesIO(); Image.new('RGB', (w, h), color).save(b, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def img_msg(color, text='Was ist auf dem Bild?'):
    return {'role': 'user', 'content': [{'type': 'text', 'text': text}, {'type': 'image_url', 'image_url': {'url': img_url(color)}}]}
E.say('x\n</think>\n\nEin rotes Bild.')
r = post('/v1/chat/completions', {'messages': [img_msg('red')]})
check('image: answer', r['choices'][0]['message']['content'] == 'Ein rotes Bild.', r)
check('image: 6 pads + embeddings + 3D RoPE', SRV.ctx.count(-1) == 0 and sum(1 for t in SRV.ctx if t < 0) == 6 and E.emb_rows == 6 and E.image_ropes[-1][1:] == (2, 3),
      (sum(1 for t in SRV.ctx if t < 0), E.emb_rows, E.image_ropes[-1:]))
consistent('image')
E.say('x\n</think>\n\nImmer noch rot.')
r = post('/v1/chat/completions', {'messages': [img_msg('red'), {'role': 'assistant', 'content': 'Ein rotes Bild.'}, {'role': 'user', 'content': 'Sicher?'}]})
check('image: follow-up reuses the cache, encoder not called again', r['usage']['prompt_tokens_details']['cached_tokens'] > 10 and FakeVision.calls == 1,
      (r['usage'], FakeVision.calls))
start = next(i for i, t in enumerate(SRV.ctx) if t < 0)
E.say('x\n</think>\n\nEin blaues Bild.')
r = post('/v1/chat/completions', {'messages': [img_msg('blue')]})
check('image: another image with the same text is not taken from the cache', r['usage']['prompt_tokens_details']['cached_tokens'] <= start
      and r['choices'][0]['message']['content'] == 'Ein blaues Bild.', (r['usage'], start))
consistent('second image')
b64 = img_url('green', 500, 400).split(',', 1)[1]
E.say('x\n</think>\n\nGrün.')
r = post('/v1/messages', {'max_tokens': 50, 'messages': [{'role': 'user', 'content': [
    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64}}, {'type': 'text', 'text': 'Farbe?'}]}]})
check('anthropic image', r['content'][-1]['text'] == 'Grün.' and E.image_ropes[-1][1:] == (4, 5), (r['content'], E.image_ropes[-1:]))
code, b = status_code('/v1/chat/completions', {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]}]})
check('broken image -> 400', code == 400, (code, b))
r = post('/v1/messages/count_tokens', {'messages': [{'role': 'user', 'content': [{'type': 'image', 'source': {'type': 'base64', 'data': b64}}]}]})
check('count_tokens counts image tokens', r['input_tokens'] > 20, r)

# 18c Ollama API
with urllib.request.urlopen(URL + '/api/tags') as rr: tags = json.loads(rr.read())
check('ollama tags', tags['models'][0]['name'].startswith(osv.MODEL_ID), tags)
E.say('x\n</think>\n\nHallo Ollama.')
r = post('/api/chat', {'model': 'ornith', 'stream': False, 'messages': [{'role': 'user', 'content': 'hi'}]})
check('ollama chat (no stream)', r['message']['content'] == 'Hallo Ollama.' and r['message'].get('thinking') == 'x' and r['done'] and r['done_reason'] == 'stop', r)
E.say('x\n</think>\n\nGestreamt.')
rq = urllib.request.Request(URL + '/api/chat', json.dumps({'messages': [{'role': 'user', 'content': 'stream bitte'}]}).encode(), {'Content-Type': 'application/json'})
with urllib.request.urlopen(rq) as rr: lines = [json.loads(l) for l in rr.read().decode().splitlines() if l.strip()]
check('ollama chat streams NDJSON by default', ''.join(l['message']['content'] for l in lines) == 'Gestreamt.' and lines[-1]['done'] and not lines[0]['done'], lines[-1])
E.say('x\n</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nz.txt\n</parameter>\n</function>\n</tool_call>')
r = post('/api/chat', {'stream': False, 'tools': TOOLS, 'messages': [{'role': 'user', 'content': 'lies z'}]})
tcs = r['message'].get('tool_calls') or []
check('ollama tool call (arguments as object)', tcs and tcs[0]['function'] == {'name': 'read_file', 'arguments': {'path': 'z.txt'}}, r['message'])
E.say('x\n</think>\n\nInhalt: 9.')
r = post('/api/chat', {'stream': False, 'tools': TOOLS, 'messages': [{'role': 'user', 'content': 'lies z'},
                                                                      {'role': 'assistant', 'content': '', 'tool_calls': tcs},
                                                                      {'role': 'tool', 'content': '9', 'tool_name': 'read_file'}]})
check('ollama tool result', r['message']['content'] == 'Inhalt: 9.' and r['prompt_eval_count'] < 60, r)
E.say('Ohne Denken.')
r = post('/api/generate', {'stream': False, 'prompt': 'sag was', 'think': False, 'images': [img_url('red').split(',', 1)[1]]})
check('ollama generate with image, think off', r['response'] == 'Ohne Denken.' and E.image_ropes[-1][1:] == (2, 3), r)
code, b = status_code('/api/chat', {'messages': []})
check('ollama error shape', code == 400 and isinstance(b.get('error'), str), b)
code, b = status_code('/api/embed', {'input': 'x'})
check('ollama embed -> 501', code == 501, code)

# 18 disk snapshots (fake kvsnap with the same API, exact): a conversation survives RAM eviction and a server restart
import types, shutil
fk = types.ModuleType('kvsnap')


def _fk_save(e, folder, ids):
    os.makedirs(folder, exist_ok=True)
    torch.save({'S': e.S.clone(), 'conv': e.conv_state.clone(), 'Kr': e.Kr.clone(), 'kv': e.kv()[:len(ids)].clone(), 'ids': list(ids)},
               os.path.join(folder, 'state.pt'))


def _fk_load(e, folder, ids):
    d = torch.load(os.path.join(folder, 'state.pt'))
    if d['ids'] != list(ids): return False
    e.reset(); e.ensure_kv(len(ids))
    e.S.copy_(d['S']); e.conv_state.copy_(d['conv']); e.Kr.copy_(d['Kr']); e.kv()[:len(ids)] = d['kv']
    return True
fk.save, fk.load, fk.wait, fk.enforce_cap = _fk_save, _fk_load, lambda: None, lambda root, cap, keep=None: None
sys.modules['kvsnap'] = fk
snapdir = tempfile.mkdtemp()


def new_server(port):
    e2 = FakeEngine()
    a3 = osv.parse_args(['--config', 'none.json', '--port', str(port), '--snapshots', snapdir, '--snap-min', '100',
                         '--park-gb', '0', '--temperature', '0', '--ctx', '65536'])
    s3, h3 = osv.serve(a3, e2, FakePrefill(e2), TOK)
    threading.Thread(target=h3.serve_forever, daemon=True).start()
    s3.ready.wait(10)
    return e2, s3, h3


def post_to(port, body):
    rq = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(rq) as rr: return json.loads(rr.read())


E2, S2, H2 = new_server(18741)
conv = [{'role': 'user', 'content': 'Snapshot-Gespräch ' + 'inhalt ' * 200}]
E2.say('x\n</think>\n\nC1'); post_to(18741, {'messages': conv})
E2.say('x\n</think>\n\nD1'); post_to(18741, {'messages': [{'role': 'user', 'content': 'anderes Thema'}]})
check('snapshot written when switching away', len(S2.snaps) == 1 and not S2.parked, (len(S2.snaps), len(S2.parked)))
H2.shutdown(); H2.server_close()
E3, S3, H3 = new_server(18742)                                      # restart: the snapshot is found on disk
check('snapshot found after restart', len(S3.snaps) == 1, len(S3.snaps))
steps0 = E3.steps
E3.say('x\n</think>\n\nC2')
r = post_to(18742, {'messages': conv + [{'role': 'assistant', 'content': 'C1'}, {'role': 'user', 'content': 'weiter nach Neustart'}]})
check('conversation continues from the snapshot', r['choices'][0]['message']['content'] == 'C2' and r['usage']['prompt_tokens_details']['cached_tokens'] >= 200
      and E3.steps - steps0 < 40, (r['usage'], E3.steps - steps0))
n = len(S3.ctx)
check('snapshot restore: engine state == context', int(E3.conv_state) == n and int(E3.S) == state_hash(S3.ctx) and E3.kv()[:n].tolist() == S3.ctx)
E3.say('x\n</think>\n\nE1'); post_to(18742, {'messages': [{'role': 'user', 'content': 'zwischendurch'}]})
shutil.rmtree(os.path.join(snapdir, next(iter(S3.snaps)) + '.kv'))  # a snapshot that vanished from disk -> plain read-in
E3.say('x\n</think>\n\nC3')
r = post_to(18742, {'messages': conv + [{'role': 'assistant', 'content': 'C1'}, {'role': 'user', 'content': 'weiter nach Neustart'},
                                        {'role': 'assistant', 'content': 'C2'}, {'role': 'user', 'content': 'nochmal'}]})
check('missing snapshot falls back to reading the prompt', r['choices'][0]['message']['content'] == 'C3', r)
H3.shutdown(); H3.server_close()
print(f'\n{len(FAILS)} failed' if FAILS else '\nall passed')
sys.exit(1 if FAILS else 0)
