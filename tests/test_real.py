"""Tests with the real model (needs the install): every cache path must give the same next-token distribution as reading the
whole prompt fresh. Prints a summary and writes test_real.json.
  rewind (ckpt), RAM park, disk snapshot (lossy by design), tool calls, thinking, Anthropic stream, checkpoint cost."""
import json, os, sys, tempfile, threading, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.isfile(os.path.join(HERE, 'ornith_serve.py')) else os.path.dirname(HERE)   # repository: tests/ below the server
sys.path.insert(0, ROOT)
import torch
import ornith_serve as osv

PORT = 18990
SNAP = tempfile.mkdtemp(prefix='ornith-snap-')
a = osv.parse_args(['--config', 'none.json', '--port', str(PORT), '--snapshots', SNAP, '--snap-min', '1000', '--quality', 'exact',
                    '--langs', 'all', '--temperature', '0'])
S, httpd = osv.serve(a)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
t0 = time.time()
while not S.ready.is_set():
    if osv.STATUS['state'] == 'error': sys.exit('start failed: ' + osv.STATUS['what'])
    time.sleep(2)
RES = {'startup_s': round(time.time() - t0, 1)}
print(f'ready after {RES["startup_s"]}s', flush=True)

FIRST = []
_orig = osv.Server._sample


def _rec(self, *a, **k):
    if getattr(self, '_want', False):
        FIRST.append(self.e.logits.float().cpu().clone()); self._want = False
    return _orig(self, *a, **k)
osv.Server._sample = _rec
FAILS = []


def check(name, ok, info=''):
    print(('ok   ' if ok else 'FAIL ') + name + (f'  [{info}]' if info else ''), flush=True)
    if not ok: FAILS.append(name)
    RES[name] = {'ok': bool(ok), 'info': str(info)[:300]}


def post(path, body):
    S._want = True
    rq = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t = time.time()
    with urllib.request.urlopen(rq, timeout=3600) as r: out = json.loads(r.read())
    out['_s'] = round(time.time() - t, 1)
    return out


def chat(msgs, **kw):
    r = post('/v1/chat/completions', dict({'messages': msgs, 'max_tokens': 60, 'reasoning_effort': 'none'}, **kw))
    return r, FIRST[-1]


def compare(name, lc, lf, kl_max):
    p, q = torch.log_softmax(lf, -1), torch.log_softmax(lc, -1)
    ok_ = torch.isfinite(p) & torch.isfinite(q)
    kl = float((p.exp() * (p - q))[ok_].sum())
    same = int(lc.argmax()) == int(lf.argmax())
    check(f'{name}: same next token as a fresh read, KL {kl:.2e}', same and kl < kl_max, f'argmax {int(lc.argmax())} vs {int(lf.argmax())}')


def fresh_again(msgs, **kw):
    """the same request read in full (cache off), as the reference"""
    S.call(S._prepare_fresh)
    S.a.cache = False
    try:
        r, lf = chat(msgs, **kw)
    finally:
        S.a.cache = True
    assert r['usage']['prompt_tokens_details']['cached_tokens'] == 0
    return r, lf


ONLY = set(os.environ.get('ONLY', '').split(',')) - {''}     # e.g. ONLY=park,snap: run just these parts


def part(name): return not ONLY or name in ONLY


SYS = {'role': 'system', 'content': 'You are a concise assistant.'}
# 1 plain answer
if part('basic'):
    m1 = [SYS, {'role': 'user', 'content': 'What is the capital of France? Answer with one word.'}]
    r, _ = chat(m1)
    txt = r['choices'][0]['message']['content']
    check('plain answer', 'paris' in txt.lower(), txt)
    RES['plain_s'] = r['_s']

    # 2 next turn uses the checkpoint before the assistant header (rewind without reset)
    m2 = m1 + [{'role': 'assistant', 'content': txt}, {'role': 'user', 'content': 'And of Germany? One word.'}]
    r, lc = chat(m2)
    c2 = r['usage']['prompt_tokens_details']['cached_tokens']
    check('turn 2 reuses the cache', c2 > 20, r['usage'])
    check('turn 2 answer', 'berlin' in r['choices'][0]['message']['content'].lower(), r['choices'][0]['message']['content'])
    _, lf = fresh_again(m2)
    compare('rewind', lc, lf, 1e-2)

# 3 thinking
if part('think'):
    r, _ = chat([{'role': 'user', 'content': 'Is 91 a prime number? Think briefly, then answer yes or no.'}], reasoning_effort='low', max_tokens=1500)
    msg = r['choices'][0]['message']
    check('thinking returned separately', bool(msg.get('reasoning_content')) and 'no' in (msg['content'] or '').lower(), (msg.get('reasoning_content') or '')[:80] + ' | ' + str(msg['content'])[:80])
    RES['tok_s_thinking'] = round(r['usage']['completion_tokens'] / max(1e-6, r['_s']), 1)

# 4 tool call + tool result
if part('tools'):
    tools = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a text file from disk',
                                               'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}]
    m4 = [{'role': 'user', 'content': 'Read the file notes.txt and tell me what it says.'}]
    r, _ = chat(m4, tools=tools, max_tokens=300)
    msg = r['choices'][0]['message']
    calls = msg.get('tool_calls') or []
    check('tool call', r['choices'][0]['finish_reason'] == 'tool_calls' and calls and json.loads(calls[0]['function']['arguments']).get('path', '').endswith('notes.txt'), msg)
    if calls:
        m4b = m4 + [{'role': 'assistant', 'content': msg['content'] or '', 'tool_calls': calls},
                    {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': 'Buy milk and call Sam at 5 pm.'}]
        r, lc = chat(m4b, tools=tools, max_tokens=80)
        check('answer after tool result', 'milk' in (r['choices'][0]['message']['content'] or '').lower() and r['usage']['prompt_tokens_details']['cached_tokens'] > 50,
              (r['choices'][0]['message']['content'], r['usage']))
        _, lf = fresh_again(m4b, tools=tools, max_tokens=80)
        compare('rewind after tool call', lc, lf, 1e-2)

# 5 switching conversations: RAM park
if part('park'):
    long_text = open(os.path.join(ROOT, 'README.md'), encoding='utf-8').read()[:6000]
    mA = [SYS, {'role': 'user', 'content': 'Here is a README:\n\n' + long_text + '\n\nWhat is it about? One sentence.'}]
    r, _ = chat(mA, max_tokens=80)
    ansA = r['choices'][0]['message']['content']
    RES['prefill_tok_s'] = round(r['usage']['prompt_tokens'] / max(1e-6, r['_s']), 1)
    chat([{'role': 'user', 'content': 'Say hello.'}])
    mA2 = mA + [{'role': 'assistant', 'content': ansA}, {'role': 'user', 'content': 'Which GPU does it need?'}]
    r, lc = chat(mA2)
    check('back to A from the RAM park', r['usage']['prompt_tokens_details']['cached_tokens'] > 1000, r['usage'])
    _, lf = fresh_again(mA2)
    compare('RAM park', lc, lf, 1e-2)

# 6 disk snapshot (no RAM park): lossy codec, so a looser bound
if part('snap'):
    S.a.park_gb = 0.0
    S.parked.clear()
    r, _ = chat(mA, max_tokens=80)
    ansA = r['choices'][0]['message']['content']
    chat([{'role': 'user', 'content': 'Say hello again.'}])
    mA2 = mA + [{'role': 'assistant', 'content': ansA}, {'role': 'user', 'content': 'Which GPU does it need?'}]
    t = time.time()
    r, lc = chat(mA2)
    check('back to A from the disk snapshot', r['usage']['prompt_tokens_details']['cached_tokens'] > 1000 and len(S.snaps) >= 1, (r['usage'], len(S.snaps)))
    RES['snapshot_restore_s'] = round(time.time() - t, 1)
    _, lf = fresh_again(mA2)
    compare('disk snapshot', lc, lf, 0.1)
    S.a.park_gb = 3.0

# 6b image input
if part('image'):
    import base64, io
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (800, 600), 'white'); d = ImageDraw.Draw(img)
    d.rectangle([80, 120, 380, 420], fill='red'); d.ellipse([450, 150, 720, 420], fill='blue')
    buf = io.BytesIO(); img.save(buf, 'PNG')
    url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    mi = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}},
                                       {'type': 'text', 'text': 'Which shapes and colors do you see? Answer in one short sentence.'}]}]
    t = time.time()
    r, _ = chat(mi, max_tokens=60)
    txt = (r['choices'][0]['message']['content'] or '').lower()
    RES['image_first_s'] = round(time.time() - t, 1)
    check('image: sees red and blue', 'red' in txt and 'blue' in txt, txt)
    mi2 = mi + [{'role': 'assistant', 'content': r['choices'][0]['message']['content']}, {'role': 'user', 'content': 'Which shape is on the left?'}]
    t = time.time()
    r, lc = chat(mi2, max_tokens=40)
    RES['image_followup_s'] = round(time.time() - t, 1)
    check('image: follow-up from the cache', r['usage']['prompt_tokens_details']['cached_tokens'] > 400, (r['usage'], r['choices'][0]['message']['content']))
    _, lf = fresh_again(mi2, max_tokens=40)
    compare('image rewind', lc, lf, 1e-2)

# 7 Anthropic stream
if part('anthropic'):
    rq = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/messages', json.dumps({'max_tokens': 50, 'stream': True, 'thinking': {'type': 'disabled'},
                                                                                  'messages': [{'role': 'user', 'content': 'Count from 1 to 5.'}]}).encode(),
                                {'Content-Type': 'application/json'})
    with urllib.request.urlopen(rq, timeout=600) as rr: raw = rr.read().decode()
    text = ''.join(json.loads(l[6:])['delta'].get('text', '') for l in raw.splitlines() if l.startswith('data: ') and '"text_delta"' in l)
    check('anthropic stream', '5' in text and 'message_stop' in raw, text)

# 8 decode speed + checkpoint cost
if part('speed'):
    r, _ = chat([{'role': 'user', 'content': 'Write a short paragraph about trains.'}], max_tokens=200)
    RES['tok_s_decode'] = round(r['usage']['completion_tokens'] / max(1e-6, r['_s']), 1)


def ck_cost():
    t = time.time()
    for _ in range(3): osv.Ckpt(S.e, S.ctx)
    return (time.time() - t) / 3
RES['checkpoint_ms'] = round(S.call(ck_cost) * 1000)
print(json.dumps({k: v for k, v in RES.items() if not isinstance(v, dict)}), flush=True)
json.dump(RES, open(os.path.join(HERE, 'test_real.json'), 'w'), indent=1)
print(f'\n{len(FAILS)} failed: {FAILS}' if FAILS else '\nall passed', flush=True)
os._exit(1 if FAILS else 0)
