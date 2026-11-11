"""ornith-serve: the Ornith-1.5-35B-A3B engine for the GTX 1070 (8 GB, Pascal sm_61) as a local model server.

Any harness or client points at http://127.0.0.1:8000 and talks one of the two common wire formats:
  OpenAI     GET /v1/models, POST /v1/chat/completions, POST /v1/completions   (stream or not, tools, reasoning)
  Anthropic  POST /v1/messages, POST /v1/messages/count_tokens                (stream or not, tools, thinking)
  Own        GET /, GET /health, GET /ornith/status, POST /ornith/config      (live settings: quality, RAM, languages)

Engine settings (expert RAM, quality mode, context, languages, snapshots) come from the command line or a JSON config file
and can be overridden per request with an extra field "ornith": {"quality": "exact"|"fast"|"turbo", "langs": ...}.

Harnesses resend the whole conversation on every call. The server keeps exact checkpoints of the engine state (DeltaNet
state, conv state, exact attention ring, logits) before the assistant header of every prompt and at the end of a long system
prompt, and only reads in what is new. Conversations it switches away from are parked exactly in RAM (LRU, --park-gb) and,
if enabled, as compressed snapshots on disk (kvsnap, --snapshots), so going back takes seconds instead of a long prefill.
One GPU = one request at a time; the others wait in a queue.

Threads: HTTP handler threads parse, validate and tokenize requests (own tokenizer copy); one worker thread owns the GPU.
"""
import argparse, collections, copy, hashlib, json, os, queue, re, sys, threading, time, traceback, uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

__version__ = '1.0.0'
HERE = os.path.dirname(os.path.abspath(__file__))
INSTALL = os.path.join(HERE, 'install.json')               # written by install.py: where the weights live
if os.path.isfile(INSTALL):
    try:
        _inst = json.load(open(INSTALL, encoding='utf-8'))
        for _k, _v in (('ORNITH_MODEL', _inst.get('model_dir')), ('ORNITH_EXPERTS', _inst.get('experts_bin')), ('ORNITH_ENGINE', _inst.get('engine_dir'))):
            if _v and not os.environ.get(_k): os.environ[_k] = _v
    except (OSError, ValueError) as _ex:
        print(f'warning: cannot read {INSTALL}: {_ex}', file=sys.stderr)
ENGINE_DIR = os.environ.get('ORNITH_ENGINE') or next(
    (d for d in (os.path.join(HERE, 'engine'), os.path.abspath(os.path.join(HERE, '..', 'ornith'))) if os.path.isdir(d)), os.path.join(HERE, 'engine'))
sys.path.insert(0, ENGINE_DIR)

MODEL_ID = 'ornith-1.5-35b-a3b'
QUALITY = {'exact': 0.0, 'fast': -0.10, 'turbo': -0.15}   # router mass below which non-VRAM experts are skipped (see README)
QUALITY_ALIAS = {'exakt': 'exact', 'schnell': 'fast'}
LANGS = ('auto', 'all', 'latin')
EFFORT = {'none': 0, 'minimal': 0, 'low': 1024, 'medium': 4096, 'high': -1, 'xhigh': -1, 'max': -1}   # thinking budget, -1 = no limit
STATUS = {'state': 'starting', 'what': '', 'done': 0, 'total': 0, 'since': time.time()}
MAX_BODY = 256 * 2 ** 20                 # bytes per request body
KEEPALIVE_S = 10                         # streaming: ping while the prompt is read in, so clients do not time out
CONFIG_KEYS = ('quality', 'langs', 'ram_gb', 'think', 'temperature', 'max_tokens')


class BadRequest(ValueError):
    """the client's fault: answered with 400 and the message"""


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, file=sys.stderr, flush=True)


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]: i += 1
    return i


def ids_hash(ids):
    return hashlib.sha1(json.dumps(list(ids)).encode()).hexdigest()


def _held(s, tag):
    """length of s without a trailing piece that could be the start of tag (held back while streaming)"""
    for j in range(min(len(s), len(tag) - 1), 0, -1):
        if tag.startswith(s[-j:]): return len(s) - j
    return len(s)


# ====================================================================================================== request -> chat
def _text_of(content):
    """OpenAI / Anthropic content (string or list of parts) -> plain text; images are not supported yet"""
    if content is None: return ''
    if isinstance(content, str): return content
    if isinstance(content, dict): content = [content]
    if not isinstance(content, list): raise BadRequest(f'content must be a string or a list of parts, not {type(content).__name__}')
    out = []
    for p in content:
        if isinstance(p, str): out.append(p); continue
        if not isinstance(p, dict): continue
        t = p.get('type')
        if t in ('text', 'input_text', 'output_text'): out.append(p.get('text') or '')
        elif t in ('image', 'image_url', 'input_image'): out.append('[image omitted: this server does not read images yet]')
        elif t == 'tool_result': out.append(_text_of(p.get('content')))
    return '\n'.join(out)


MAX_IMAGE_BYTES = 20 * 2 ** 20


def _image_bytes(ref):
    """OpenAI image_url / Anthropic image source -> raw image bytes (data URL, base64 or http(s) URL)"""
    import base64, urllib.request
    try:
        if isinstance(ref, dict) and ref.get('type') == 'base64': return base64.b64decode(ref.get('data') or '')
        url = ref.get('url') if isinstance(ref, dict) else ref
        if not isinstance(url, str) or not url: raise BadRequest('image without data or url')
        if url.startswith('data:'):
            return base64.b64decode(url.split(',', 1)[1])
        if url.startswith(('http://', 'https://')):
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'ornith-serve'}), timeout=20) as r:
                data = r.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES: raise BadRequest(f'image larger than {MAX_IMAGE_BYTES // 2 ** 20} MB: {url[:80]}')
            return data
        raise BadRequest('image url must be a data: or http(s): URL')
    except BadRequest:
        raise
    except Exception as ex:
        raise BadRequest(f'cannot read image: {type(ex).__name__}: {ex}')


def _parts(content):
    """message content -> plain string, or a list of text / image parts for the chat template (images as raw bytes)"""
    if content is None or isinstance(content, str): return content or ''
    if isinstance(content, dict): content = [content]
    if not isinstance(content, list): raise BadRequest(f'content must be a string or a list of parts, not {type(content).__name__}')
    out = []
    for p in content:
        if isinstance(p, str): out.append({'type': 'text', 'text': p}); continue
        if not isinstance(p, dict): continue
        t = p.get('type')
        if t in ('text', 'input_text', 'output_text'): out.append({'type': 'text', 'text': p.get('text') or ''})
        elif t in ('image_url', 'input_image'): out.append({'type': 'image', 'image': _image_bytes(p.get('image_url'))})
        elif t == 'image': out.append({'type': 'image', 'image': _image_bytes(p.get('source') or {})})
        elif t == 'tool_result':
            sub = _parts(p.get('content'))
            out += sub if isinstance(sub, list) else [{'type': 'text', 'text': sub}]
    if not any(q['type'] == 'image' for q in out): return '\n'.join(q['text'] for q in out)
    return out


def _images_of(msgs):
    """raw bytes of every image in the order the chat template renders them"""
    return [q['image'] for m in msgs if isinstance(m.get('content'), list) for q in m['content'] if q.get('type') == 'image']


def _args_dict(a):
    if isinstance(a, dict): return a
    try:
        v = json.loads(a or '{}')
        return v if isinstance(v, dict) else {'arguments': v}
    except (ValueError, TypeError):
        return {'arguments': a}


def _check_msgs(msgs):
    if not any(m['role'] == 'user' for m in msgs): raise BadRequest('messages must contain at least one user message')
    for m in msgs:
        if m['role'] not in ('system', 'user', 'assistant', 'tool'): raise BadRequest(f'unknown message role {m["role"]!r}')


def _json_note(rf):
    """OpenAI response_format -> instruction for the system prompt (the engine has no constrained decoding)"""
    if not isinstance(rf, dict): return ''
    if rf.get('type') == 'json_object': return 'Answer with one valid JSON object only, no other text.'
    if rf.get('type') == 'json_schema':
        sch = (rf.get('json_schema') or {}).get('schema') or {}
        return 'Answer with one valid JSON object only, no other text. It must follow this JSON schema:\n' + json.dumps(sch, ensure_ascii=False)
    return ''


def openai_to_chat(body):
    """OpenAI chat request -> (messages in the template's format, tools)"""
    raw = body.get('messages')
    if not isinstance(raw, list) or not raw: raise BadRequest('messages must be a non-empty list')
    system, msgs = [], []
    for m in raw:
        if not isinstance(m, dict) or 'role' not in m: raise BadRequest('every message needs a role')
        role = 'system' if m['role'] == 'developer' else m['role']
        if role == 'system': system.append(_text_of(m.get('content'))); continue   # the template wants one, at the start
        d = {'role': role, 'content': _parts(m.get('content')) if role in ('user', 'tool') else _text_of(m.get('content'))}
        if role == 'assistant':
            rc = m.get('reasoning_content') or m.get('reasoning')
            if isinstance(rc, str): d['reasoning_content'] = rc
            calls = [c for c in (m.get('tool_calls') or []) if isinstance(c, dict) and c.get('function')]
            if calls: d['tool_calls'] = [{'name': c['function'].get('name', ''), 'arguments': _args_dict(c['function'].get('arguments'))} for c in calls]
        msgs.append(d)
    note = _json_note(body.get('response_format'))
    if note: system.append(note)
    if system: msgs.insert(0, {'role': 'system', 'content': '\n\n'.join(s for s in system if s)})
    _check_msgs(msgs)
    tools = [t['function'] if 'function' in t else t for t in (body.get('tools') or []) if isinstance(t, dict) and t.get('type', 'function') == 'function']
    tools = [t for t in tools if t.get('name')]
    if body.get('tool_choice') == 'none': tools = []
    return msgs, tools


def anthropic_to_chat(body):
    """Anthropic messages request -> (messages, tools)"""
    raw = body.get('messages')
    if not isinstance(raw, list) or not raw: raise BadRequest('messages must be a non-empty list')
    msgs = []
    sysm = body.get('system')
    if sysm: msgs.append({'role': 'system', 'content': _text_of(sysm)})
    for m in raw:
        if not isinstance(m, dict) or m.get('role') not in ('user', 'assistant'): raise BadRequest('message roles must be user or assistant')
        c = m.get('content')
        if isinstance(c, str): msgs.append({'role': m['role'], 'content': c}); continue
        if not isinstance(c, list): raise BadRequest('content must be a string or a list of blocks')
        if m['role'] == 'assistant':
            d = {'role': 'assistant', 'content': '', 'tool_calls': []}
            for p in c:
                if not isinstance(p, dict): continue
                if p.get('type') == 'text': d['content'] += p.get('text') or ''
                elif p.get('type') == 'thinking': d['reasoning_content'] = p.get('thinking') or ''
                elif p.get('type') == 'tool_use': d['tool_calls'].append({'name': p.get('name', ''), 'arguments': p.get('input') or {}})
            if not d['tool_calls']: del d['tool_calls']
            msgs.append(d)
        else:                                                     # user: tool results become tool messages, the rest stays user
            rest = []
            for p in c:
                if isinstance(p, dict) and p.get('type') == 'tool_result':
                    res = _parts(p.get('content'))
                    if p.get('is_error'):
                        res = 'Error: ' + res if isinstance(res, str) else [{'type': 'text', 'text': 'Error: '}] + res
                    msgs.append({'role': 'tool', 'content': res})
                else:
                    rest.append(p)
            res = _parts(rest)
            if (isinstance(res, list) and res) or (isinstance(res, str) and res.strip()): msgs.append({'role': 'user', 'content': res})
    _check_msgs(msgs)
    tools = [{'name': t['name'], 'description': t.get('description', ''), 'parameters': t.get('input_schema') or {'type': 'object', 'properties': {}}}
             for t in (body.get('tools') or []) if isinstance(t, dict) and t.get('name')]
    if (body.get('tool_choice') or {}).get('type') == 'none': tools = []
    return msgs, tools


OLLAMA_EFFORT = {True: None, False: 'none', 'low': 'low', 'medium': 'medium', 'high': 'high'}


def ollama_to_openai(body, generate=False):
    """Ollama /api/chat or /api/generate request -> an OpenAI chat body (then the normal path)"""
    o = body.get('options') if isinstance(body.get('options'), dict) else {}
    if generate:
        msgs = []
        if body.get('system'): msgs.append({'role': 'system', 'content': body['system']})
        msgs.append({'role': 'user', 'content': body.get('prompt') or '', 'images': body.get('images') or []})
    else:
        msgs = body.get('messages')
        if not isinstance(msgs, list) or not msgs: raise BadRequest('messages must be a non-empty list')
    out = []
    for m in msgs:
        if not isinstance(m, dict): raise BadRequest('every message must be an object')
        d = {'role': m.get('role'), 'content': m.get('content') or ''}
        if m.get('images'):
            d['content'] = [{'type': 'text', 'text': d['content']}] + [
                {'type': 'image_url', 'image_url': {'url': im if str(im).startswith('data:') else 'data:image/png;base64,' + im}} for im in m['images']]
        if m.get('thinking'): d['reasoning_content'] = m['thinking']
        if m.get('tool_calls'):
            d['tool_calls'] = [{'type': 'function', 'function': {'name': (c.get('function') or {}).get('name', ''),
                                                                 'arguments': (c.get('function') or {}).get('arguments') or {}}} for c in m['tool_calls']]
        out.append(d)
    ob = {'messages': out, 'tools': body.get('tools'), 'temperature': o.get('temperature'), 'top_p': o.get('top_p'), 'top_k': o.get('top_k'),
          'min_p': o.get('min_p'), 'seed': o.get('seed'), 'stop': o.get('stop'), 'max_tokens': o.get('num_predict') if (o.get('num_predict') or 0) > 0 else None,
          'ornith': body.get('ornith')}
    th = body.get('think')
    if th in OLLAMA_EFFORT and OLLAMA_EFFORT[th]: ob['reasoning_effort'] = OLLAMA_EFFORT[th]
    elif th is True: ob['chat_template_kwargs'] = {'enable_thinking': True}
    fmt = body.get('format')
    if fmt == 'json': ob['response_format'] = {'type': 'json_object'}
    elif isinstance(fmt, dict): ob['response_format'] = {'type': 'json_schema', 'json_schema': {'schema': fmt}}
    return {k: v for k, v in ob.items() if v is not None}


def parse_calls(text, tools):
    """complete <tool_call><function=NAME><parameter=P>\\nVALUE\\n</parameter>...</function></tool_call> blocks -> [(name, args)].
    Values are JSON-decoded unless the tool schema says the parameter is a string. A call cut off by max_tokens is dropped
    (a harness would run it with missing arguments)."""
    schema = {t['name']: (t.get('parameters') or {}).get('properties') or {} for t in tools}
    calls = []
    for blk in re.findall(r'(?s)<tool_call>(.*?)</tool_call>', text):
        m = re.search(r'<function=([^>\s]+)>', blk)
        if not m: continue
        args = {}
        for pm in re.finditer(r'(?s)<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>', blk):
            k, v = pm.group(1), pm.group(2)
            if (schema.get(m.group(1), {}).get(k) or {}).get('type') != 'string':
                try: v = json.loads(v)
                except ValueError: pass
            args[k] = v
        calls.append((m.group(1), args))
    return calls


def _num(body, key, default, lo, hi, cast=float):
    v = body.get(key)
    if v is None: return default
    try: v = cast(v)
    except (TypeError, ValueError): raise BadRequest(f'{key} must be a number')
    return min(hi, max(lo, v))


def _think_settings(a, body, anthropic=False):
    """(think on/off, thinking token budget -1 = unlimited, hide reasoning) from the request; default = server setting"""
    think, budget, hide = a.think, -1, False
    if anthropic:
        t = body.get('thinking') or {}
        if t.get('type') == 'enabled': think, budget = True, int(t.get('budget_tokens') or -1)
        elif t.get('type') == 'disabled': think = False
    else:
        r = body.get('reasoning') if isinstance(body.get('reasoning'), dict) else {}   # OpenRouter: {effort, max_tokens, enabled, exclude}
        eff = body.get('reasoning_effort') or r.get('effort')
        if eff in EFFORT: think, budget = EFFORT[eff] != 0, EFFORT[eff]
        if r.get('enabled') is True: think = True
        if r.get('enabled') is False: think = False
        if isinstance(r.get('max_tokens'), int): think, budget = True, r['max_tokens']
        hide = r.get('exclude') is True or body.get('include_reasoning') is False
        ctk = body.get('chat_template_kwargs') or {}
        if 'enable_thinking' in ctk: think = bool(ctk['enable_thinking'])
    return think, budget, hide


def build_request(a, body, msgs, tools, anthropic=False):
    think, budget, hide = _think_settings(a, body, anthropic)
    stop = body.get('stop_sequences') if anthropic else body.get('stop')
    if isinstance(stop, str): stop = [stop]
    if stop is not None and not (isinstance(stop, list) and all(isinstance(s, str) for s in stop)): raise BadRequest('stop must be a string or a list of strings')
    mt = body.get('max_tokens') or body.get('max_completion_tokens') or a.max_tokens
    ov = body.get('ornith') if isinstance(body.get('ornith'), dict) else {}
    if ov.get('quality') is not None and QUALITY_ALIAS.get(ov['quality'], ov['quality']) not in QUALITY: raise BadRequest(f'ornith.quality must be one of {list(QUALITY)}')
    if ov.get('langs') is not None and ov['langs'] not in LANGS: raise BadRequest(f'ornith.langs must be one of {list(LANGS)}')
    if body.get('n') not in (None, 1): raise BadRequest('n > 1 is not supported (one GPU, one answer per request)')
    return {'messages': msgs, 'tools': tools, 'max_tokens': int(_num({'m': mt}, 'm', a.max_tokens, 1, 10 ** 7, int)),
            'temperature': _num(body, 'temperature', a.temperature, 0.0, 2.0), 'top_p': _num(body, 'top_p', 0.95, 0.0, 1.0),
            'top_k': _num(body, 'top_k', 20, 0, 1000, int), 'min_p': _num(body, 'min_p', 0.0, 0.0, 1.0),
            'stop': [s for s in (stop or []) if s][:16], 'seed': body.get('seed'), 'think': think, 'think_budget': budget,
            'hide_reasoning': hide, 'ornith': ov}


# ====================================================================================================== engine side
class Ckpt:
    """exact engine state after the tokens ids (a prefix of the live context, or parked with its own copy of the KV cache)"""

    def __init__(self, e, ids, msg_end=False):
        self.ids, self.n, self.msg_end = list(ids), len(ids), msg_end
        self.S, self.conv = e.S.to('cpu', copy=True), e.conv_state.to('cpu', copy=True)
        self.Kr, self.Vr = e.Kr.to('cpu', copy=True), e.Vr.to('cpu', copy=True)
        self.logits = e.logits.to('cpu', copy=True)
        self.rope = e.rope_state(self.n) if hasattr(e, 'rope_state') else None
        self.kv = None
        self.disk = False                                         # already written as a kvsnap snapshot

    def add_kv(self, e):
        if self.kv is None:
            chunks = (self.n + e.CK - 1) // e.CK
            self.kv = e.arena[:chunks * e.CSB].to('cpu', copy=True)

    def nbytes(self):
        t = [self.S, self.conv, self.Kr, self.Vr, self.logits] + ([self.kv] if self.kv is not None else [])
        return sum(x.numel() * x.element_size() for x in t)

    def restore(self, e, full):
        """full: also the KV cache (parked state, new conversation in the engine); else only the recurrent part (rewind)"""
        if full:
            e.reset(); e.ensure_kv(max(1, self.n))
            e.arena[:self.kv.numel()].copy_(self.kv)
        if self.rope is not None and hasattr(e, 'set_rope_state'): e.set_rope_state(self.rope)
        e.S.copy_(self.S); e.conv_state.copy_(self.conv); e.Kr.copy_(self.Kr); e.Vr.copy_(self.Vr)
        e.logits.copy_(self.logits)
        if hasattr(e, 'next_tok'): e.next_tok.copy_(e.logits.argmax().view(1))


class Server:
    """owns the engine; everything that touches the GPU runs on one worker thread (self.jobs)"""

    def __init__(self, a, engine=None, prefill=None, tok=None):
        self.a = a
        self.jobs = queue.Queue()
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.tok_lock = threading.Lock()                          # the HTTP threads share one tokenizer copy
        self.queue_len = 0
        self.stats = {'requests': 0, 'prompt_tokens': 0, 'cached_tokens': 0, 'gen_tokens': 0, 'last_tok_s': 0.0, 'errors': 0}
        self.ctx, self.live, self.parked, self.snaps = [], [], collections.OrderedDict(), {}
        self._inject = (engine, prefill, tok)
        threading.Thread(target=self._worker, daemon=True, name='gpu-worker').start()

    # ---------------------------------------------------------------- start-up (worker thread)
    def _boot(self):
        a = self.a
        engine, prefill, tok = self._inject
        if engine is None:
            import torch
            from transformers import AutoTokenizer
            STATUS.update(state='loading', what='checking GPU')
            if not torch.cuda.is_available(): raise SystemExit('no CUDA GPU found (driver installed? torch with CUDA?)')
            cc, name = torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            if (cc != (6, 1) or vram < 7.5) and not a.force:
                raise SystemExit(f'{name} (sm_{cc[0]}{cc[1]}, {vram:.1f} GB) found. ornith-serve is built and tuned for the '
                                 f'GTX 1070 (Pascal sm_61, 8 GB) only; start with --force to try anyway.')
            from engine3 import Ornith3
            from prefill import Prefill
            from weights import PATH
            STATUS.update(what='loading tokenizer')
            tok = AutoTokenizer.from_pretrained(PATH)
            STATUS.update(what='loading model')
            engine = Ornith3(ram_gb=a.ram_gb, prefetch=8, max_ctx=a.ctx, kv8=True, free_gb=a.free_gb, warm_cache=True, log=lambda s: None)
            prefill = Prefill(engine)
        self.e, self.pf, self.tok = engine, prefill, tok
        self.tok_http = copy.deepcopy(tok)
        self.image_pad = tok.convert_tokens_to_ids('<|image_pad|>')
        self.vision, self.vision_lock, self.img_cache = getattr(self, 'vision', None), threading.Lock(), collections.OrderedDict()
        self.end_id = tok.convert_tokens_to_ids('<|im_end|>')
        self.end_ids = {self.end_id, tok.eos_token_id} - {None}
        self.im_start = tok.convert_tokens_to_ids('<|im_start|>')
        self.close_think = tok('\n</think>\n\n', add_special_tokens=False).input_ids
        self.quality = None
        self.set_quality(a.quality)
        try:
            import langfilter
            self.langfilter = langfilter if hasattr(self.e, 'set_langs') else None
        except ImportError:
            self.langfilter = None
        self.kvsnap = None
        if a.snapshots:
            try:
                import kvsnap
                self.kvsnap = kvsnap
                os.makedirs(a.snapshots, exist_ok=True)
                self.snaps = self._scan_snaps()
            except ImportError:
                log('kvsnap not found: disk snapshots off')
        hot = os.path.join(a.home, 'hot.json')
        if hasattr(self.e, 'load_hot') and os.path.exists(hot):
            STATUS.update(what='preloading frequent experts')
            try: self.e.load_hot(hot)
            except Exception: traceback.print_exc()
        STATUS.update(state='ready', what='', done=0, total=0)
        self.ready.set()
        log(f'ready: OpenAI http://{a.host}:{a.port}/v1, Anthropic http://{a.host}:{a.port} (model {MODEL_ID}, quality {self.quality}, '
            f'RAM {a.ram_gb} GB, context {a.ctx}, snapshots {"on, " + str(len(self.snaps)) + " stored" if self.kvsnap else "off"})')

    def _worker(self):
        try:
            self._boot()
        except BaseException as ex:
            STATUS.update(state='error', what=str(ex)); log('start failed:', ex)
            if not isinstance(ex, SystemExit): traceback.print_exc()
            if getattr(self.a, 'exit_on_error', False): os._exit(1)
            return
        n = 0
        while True:
            fn, args, out = self.jobs.get()
            try:
                out.put(('ok', fn(*args)))
            except BaseException as ex:
                if not isinstance(ex, BadRequest): traceback.print_exc()
                out.put(('err', ex))
            n += 1
            if n % 20 == 0: self._save_hot()

    def submit(self, fn, *args):
        """queue fn on the GPU thread; returns a queue that receives ('ok', value) or ('err', exception)"""
        out = queue.Queue()
        self.jobs.put((fn, args, out))
        return out

    def call(self, fn, *args):
        with self.lock: self.queue_len += 1
        try:
            kind, val = self.submit(fn, *args).get()
        finally:
            with self.lock: self.queue_len -= 1
        if kind == 'err': raise val
        return val

    def _save_hot(self):
        if hasattr(self.e, 'save_hot'):
            p = os.path.join(self.a.home, 'hot.json')
            try: self.e.save_hot(p + '.tmp'); os.replace(p + '.tmp', p)
            except Exception: traceback.print_exc()

    def shutdown(self):
        """worker thread: keep the hot experts for the next start, finish snapshot writes"""
        self._save_hot()
        if self.kvsnap: self.kvsnap.wait()

    # ---------------------------------------------------------------- tokenizing (HTTP threads)
    def render(self, req, raw=None):
        """prompt text + ids of a request, and where its system prompt ends; BadRequest for template errors / too long"""
        tools = req['tools']
        kw = {'tools': tools} if tools else {}
        with self.tok_lock:
            try:
                prompt = raw if raw is not None else self.tok_http.apply_chat_template(
                    req['messages'], add_generation_prompt=True, tokenize=False, enable_thinking=req['think'], **kw)
            except Exception as ex:                           # jinja errors: the request does not fit the chat template
                raise BadRequest(f'chat template: {ex}')
            ids = self.tok_http(prompt, add_special_tokens=False).input_ids
            sys_len = 0
            if raw is None and req['messages'][0]['role'] == 'system':
                head = self.tok_http.apply_chat_template(req['messages'][:1] + [{'role': 'user', 'content': 'x'}], tokenize=False, **kw)
                sys_ids = self.tok_http(head[:head.index('<|im_start|>user')], add_special_tokens=False).input_ids
                if ids[:len(sys_ids)] == sys_ids: sys_len = len(sys_ids)
        ids, keys, images = self._expand_images(ids, _images_of(req['messages']) if raw is None else [])
        if len(ids) + 1 > self.a.ctx - 16:
            raise BadRequest(f'prompt too long: {len(ids)} tokens, the server context is {self.a.ctx} (start with a larger --ctx)')
        req.update(prompt=prompt, ids=ids, keys=keys, images=images, sys_len=sys_len, raw=raw)
        return req

    # ---------------------------------------------------------------- images (HTTP threads, CPU)
    def _encode_image(self, data):
        """raw image bytes -> (embeddings [n, 2048], grid h, grid w, content hash); LRU cache, the encoder loads on first use"""
        h = hashlib.sha1(data).hexdigest()
        with self.vision_lock:
            if h in self.img_cache:
                self.img_cache.move_to_end(h); return self.img_cache[h] + (h,)
            if self.vision is None:
                STATUS.update(what='loading the image encoder (first image)')
                from vision import Vision
                t = time.time()
                self.vision = Vision(max_pixels=self.a.max_image_pixels)
                log(f'image encoder loaded in {time.time() - t:.1f}s')
            import io
            from PIL import Image
            try:
                img = Image.open(io.BytesIO(data)); img.load()
            except Exception as ex:
                raise BadRequest(f'not a readable image: {ex}')
            t = time.time()
            emb, (gh, gw) = self.vision.encode(img)
            log(f'image {img.size[0]}x{img.size[1]} -> {gh * gw} tokens in {time.time() - t:.1f}s')
            self.img_cache[h] = (emb, gh, gw)
            while len(self.img_cache) > 32: self.img_cache.popitem(last=False)
            return emb, gh, gw, h

    def _expand_images(self, ids, datas):
        """each single <|image_pad|> of the template becomes gh*gw pads. keys = ids for the prefix cache, with the pads of
        an image replaced by negative numbers from its content hash (different images must never look equal)"""
        pads = [i for i, t in enumerate(ids) if t == self.image_pad]
        if len(pads) != len(datas): raise BadRequest(f'{len(datas)} images but {len(pads)} image slots in the prompt')
        if not datas: return ids, list(ids), []
        out, keys, images, last = [], [], [], 0
        for p, data in zip(pads, datas):
            emb, gh, gw, h = self._encode_image(data)
            out += ids[last:p]; keys += ids[last:p]
            start, n = len(out), gh * gw
            base = int(h[:12], 16)
            out += [self.image_pad] * n
            keys += [-1 - (base + j) % (2 ** 31 - 2) for j in range(n)]
            images.append((start, gh, gw, emb))
            last = p + 1
        out += ids[last:]; keys += ids[last:]
        return out, keys, images

    def count_tokens(self, msgs, tools):
        """prompt tokens of these messages, images included (they are encoded to know their size)"""
        return len(self.render({'messages': msgs, 'tools': tools, 'think': self.a.think})['ids'])

    # ---------------------------------------------------------------- settings (worker thread)
    def set_quality(self, q):
        q = QUALITY_ALIAS.get(q, q)
        if q not in QUALITY: raise BadRequest(f'quality must be one of {list(QUALITY)}')
        if q != self.quality and hasattr(self.e, 'set_quality'): self.e.set_quality(0.0, QUALITY[q])
        self.quality = q

    def configure(self, b):
        bad = [k for k in b if k not in CONFIG_KEYS]
        if bad: raise BadRequest(f'unknown setting(s) {bad}; known: {list(CONFIG_KEYS)}')
        if 'langs' in b and b['langs'] not in LANGS: raise BadRequest(f'langs must be one of {list(LANGS)}')
        if 'quality' in b: self.set_quality(b['quality']); self.a.quality = self.quality
        if 'langs' in b: self.a.langs = b['langs']
        if 'ram_gb' in b:
            if not hasattr(self.e, 'set_ram_gb'): raise BadRequest('this engine cannot change its RAM while running')
            self.a.ram_gb = min(13.5, max(0.5, float(b['ram_gb']))); self.e.set_ram_gb(self.a.ram_gb)
        if 'think' in b: self.a.think = bool(b['think'])
        if 'temperature' in b: self.a.temperature = min(2.0, max(0.0, float(b['temperature'])))
        if 'max_tokens' in b: self.a.max_tokens = max(1, int(b['max_tokens']))
        log('settings:', {k: getattr(self.a, k) for k in CONFIG_KEYS})
        return self.status()

    def status(self):
        """safe from any thread (reads copies of what the worker may change at the same time)"""
        a = self.a
        s = dict(STATUS, version=__version__, model=MODEL_ID, quality=a.quality, langs=a.langs, ram_gb=a.ram_gb, ctx_max=a.ctx,
                 think=a.think, temperature=a.temperature, max_tokens=a.max_tokens, queue=self.queue_len, cache=a.cache, park_gb=a.park_gb,
                 snapshots_on=bool(a.snapshots), snap_cap_gb=a.snap_cap_gb, max_image_pixels=a.max_image_pixels, host=a.host, port=a.port,
                 api_key=bool(a.api_key), **self.stats)
        if self.ready.is_set():
            live, parked = list(self.live), list(self.parked.values())
            s.update(live_tokens=len(self.ctx), checkpoints=[c.n for c in live], parked=[c.n for c in parked],
                     parked_gb=round(sum(c.nbytes() for c in parked) / 2 ** 30, 2), snapshots=len(self.snaps))
            if hasattr(self.e, 'vram_split'):
                try:
                    kv, ex = self.e.vram_split(); s.update(vram_kv_gb=round(kv, 2), vram_experts_gb=round(ex, 2))
                except Exception:
                    pass
        return s

    def _apply_langs(self, text, langs):
        if not self.langfilter: return
        if langs == 'all': want = {'cjk': True, 'other': True}
        elif langs == 'latin': want = {'cjk': False, 'other': False}
        else: want = self.langfilter.present(text)              # auto: a script group is on once it appears in the request
        self.e.set_langs(**want)

    # ---------------------------------------------------------------- disk snapshots (worker thread)
    def _snap_dir(self, key): return os.path.join(self.a.snapshots, key + '.kv')

    def _scan_snaps(self):
        """snapshots on disk: <key>.kv folders with ids.npy + logits.pt next to kvsnap's files"""
        import numpy as np
        out = {}
        for f in os.listdir(self.a.snapshots):
            d = os.path.join(self.a.snapshots, f)
            if f.endswith('.kv') and os.path.isfile(os.path.join(d, 'ids.npy')) and os.path.isfile(os.path.join(d, 'logits.pt')):
                try:
                    ids = np.load(os.path.join(d, 'ids.npy')).tolist()
                    out[f[:-3]] = {'n': len(ids), 'ids': ids}
                except Exception:
                    log(f'skipping broken snapshot {f}')
        return out

    def _save_snap(self, ck):
        import numpy as np, torch
        t0 = time.time()
        key = next((k for k, v in self.snaps.items() if v['n'] <= ck.n and ck.ids[:v['n']] == v['ids']), None)
        key = key or uuid.uuid4().hex[:12]                        # an older snapshot of the same conversation grows in place
        d = self._snap_dir(key)
        ck.restore(self.e, full=False)                            # kvsnap reads the engine state: put it back to ck
        self.kvsnap.save(self.e, d, ck.ids)                       # GPU part now, compression + writing in the background
        np.save(os.path.join(d, 'ids.npy'), np.asarray(ck.ids, dtype=np.int32))
        torch.save(ck.logits, os.path.join(d, 'logits.pt'))
        self.snaps[key] = {'n': ck.n, 'ids': ck.ids}
        self.kvsnap.enforce_cap(self.a.snapshots, int(self.a.snap_cap_gb * 2 ** 30), keep=d)
        ck.disk = True
        log(f'snapshot {key}: {ck.n} tokens ({time.time() - t0:.1f}s on the GPU, writing in the background)')

    # ---------------------------------------------------------------- prefix cache (worker thread)
    def _park(self, ck, disk=True):
        """keep checkpoint ck while the engine works on something else: exact copy in RAM (LRU, --park-gb) and, once,
        a compressed snapshot on disk"""
        e = self.e
        if disk and self.kvsnap and not ck.disk and ck.n >= self.a.snap_min:
            try:
                self._save_snap(ck)
            except Exception:
                log('snapshot failed (the conversation stays in RAM only):'); traceback.print_exc()
        h = ids_hash(ck.ids)
        if h in self.parked: self.parked.move_to_end(h); return
        budget = self.a.park_gb * 2 ** 30
        ck.add_kv(e)
        if ck.nbytes() > budget:
            ck.kv = None
            log(f'conversation of {ck.n} tokens is larger than --park-gb {self.a.park_gb}: not parked in RAM')
            return
        self.parked[h] = ck
        while sum(c.nbytes() for c in self.parked.values()) > budget: self.parked.popitem(last=False)

    def _park_target(self):
        """the live checkpoint worth keeping when the engine leaves this conversation: the latest one before an assistant
        header (the next request of this conversation matches it), else the latest"""
        return next((c for c in reversed(self.live) if c.msg_end), self.live[-1])

    def _prepare(self, ids):
        """bring the engine to the longest known prefix of ids -> (tokens of ids already in the engine, source)"""
        e = self.e
        if not self.a.cache: return self._prepare_fresh()       # --cache off: every request reads its whole prompt
        n_ctx = lcp(self.ctx, ids)
        best, how = 0, 'none'
        if n_ctx == len(self.ctx) and self.ctx: best, how = n_ctx, 'live'
        live = max((c for c in self.live if c.n <= n_ctx), key=lambda c: c.n, default=None)
        if live and live.n > best: best, how = live.n, 'ckpt'
        park = max((c for c in self.parked.values() if best < c.n <= len(ids) and ids[:c.n] == c.ids), key=lambda c: c.n, default=None)
        if park: best, how = park.n, 'park'
        snap = None
        for k, v in self.snaps.items():
            if best < v['n'] <= len(ids) and ids[:v['n']] == v['ids']: snap, best, how = k, v['n'], 'snap'
        if best == len(ids) and how == 'live':                   # nothing new to read but no logits for it: use a checkpoint
            how, best = ('ckpt', live.n) if live else ('none', 0)
        if self.live and (how in ('park', 'snap', 'none') or (how == 'ckpt' and live.n < self.live[-1].n)):
            self._park(self._park_target())                       # leaving / branching off the live conversation
        if how == 'live':
            return best, how
        if how == 'ckpt':
            live.restore(e, full=False)
            self.ctx = self.ctx[:live.n]
            self.live = [c for c in self.live if c.n <= live.n]
            return best, how
        if how == 'park':
            STATUS.update(what='restoring parked conversation')
            park.restore(e, full=True)
            self.ctx, self.live = list(park.ids), [park]
            self.parked.move_to_end(ids_hash(park.ids))
            return best, how
        if how == 'snap':
            STATUS.update(what='loading snapshot')
            import torch
            d = self._snap_dir(snap)
            try:
                ok = os.path.isdir(d) and self.kvsnap.load(e, d, self.snaps[snap]['ids'])
                if ok: e.logits.copy_(torch.load(os.path.join(d, 'logits.pt')).to(e.logits.device))
            except Exception:
                traceback.print_exc(); ok = False
            if ok:
                self.ctx = list(self.snaps[snap]['ids'])
                ck = Ckpt(e, self.ctx, msg_end=True); ck.disk = True
                self.live = [ck]
                return best, how
            log(f'snapshot {snap} could not be loaded: reading the prompt instead')
            self.snaps.pop(snap, None)
        return self._prepare_fresh()

    def _prepare_fresh(self):
        self.e.reset(); self.ctx, self.live = [], []
        return 0, 'none'

    def _feed(self, req, end):
        """read prompt positions len(ctx) .. end-1 of req into the engine (with the image embeddings and 3D RoPE of the
        images in that range); ctx gets the cache keys"""
        import torch
        p0 = len(self.ctx)
        if end <= p0: return
        ids = req['ids'][p0:end]
        imgs = [im for im in req['images'] if p0 <= im[0] < end]
        if any(st + gh * gw > end for st, gh, gw, _ in imgs): raise RuntimeError('internal: a feed boundary splits an image')
        for st, gh, gw, _ in imgs: self.e.add_image_rope(st, gh, gw)
        emb = (torch.cat([torch.arange(st - p0, st - p0 + gh * gw) for st, gh, gw, _ in imgs]),
               torch.cat([em for *_, em in imgs])) if imgs else None
        if len(ids) > 48 or emb is not None:                      # images need the prefill (it can replace input embeddings)
            def cb(msg):
                m = re.match(r'prefill (\d+)/(\d+)', msg)
                if m: STATUS.update(what='reading prompt', done=p0 + int(m.group(1)), total=p0 + int(m.group(2)))
            self.pf.run(ids, p0=p0, log=cb, **({'emb': emb} if emb is not None else {}))
        else:
            for i, t in enumerate(ids): self.e.step(t, p0 + i)
        self.ctx += req['keys'][p0:end]

    def _checkpoint(self, msg_end=False):
        if not self.a.cache: return None
        if self.live and self.live[-1].n == len(self.ctx): return self.live[-1]
        ck = Ckpt(self.e, self.ctx, msg_end)
        self.live.append(ck)
        self.live = self.live[-self.a.live_ckpts:]
        return ck

    # ---------------------------------------------------------------- generation (worker thread)
    def _sample(self, temp, top_p, top_k, min_p, gen):
        import torch
        lg = self.e.logits
        if temp <= 0 or top_p <= 0 or top_k == 1: return int(lg.argmax())
        k = min(top_k or 40, lg.numel())
        top = torch.topk(lg.float(), k)
        p = torch.softmax(top.values / temp, -1)
        if min_p: p = p * (p >= min_p * p[0])
        if top_p < 1: p = p * ((p.cumsum(-1) - p) < top_p)
        s = p.sum()
        if not torch.isfinite(s) or s <= 0: return int(lg.argmax())
        return int(top.indices[torch.multinomial((p / s).cpu(), 1, generator=gen)])

    def generate(self, req, emit, cancel):
        """req (see build_request + render). emit(kind, text) with kind 'reasoning' | 'content'. Returns a result dict."""
        try:
            return self._generate(req, emit, cancel)
        except BaseException:
            self._prepare_fresh()                                 # engine and context may disagree now: start clean
            raise
        finally:
            STATUS.update(state='ready', what='', done=0, total=0)

    def _generate(self, req, emit, cancel):
        import torch
        t_start = time.time()
        ov, ids = req['ornith'], req['keys']                      # the prefix cache compares keys (images by content)
        self.set_quality(ov.get('quality') or self.a.quality)
        self._apply_langs(req['prompt'], ov.get('langs') or self.a.langs)
        STATUS.update(state='busy', what='preparing', done=0, total=len(ids))
        cached, how = self._prepare(ids)
        # long shared system prompt: checkpoint where it ends, so new conversations of the same harness start there
        if self.a.cache and req['sys_len'] >= 1024 and len(self.ctx) < req['sys_len']:
            self._feed(req, req['sys_len'])
            self._park(self._checkpoint(), disk=False)
        # checkpoint before the assistant header: the client sends the answer back as '<think>\n\n</think>...', which
        # tokenizes differently from the header '<think>\n' that ends this prompt
        hdr = next((i for i in range(len(ids) - 1, len(self.ctx) - 1, -1) if ids[i] == self.im_start), -1)
        if hdr > len(self.ctx):
            self._feed(req, hdr); self._checkpoint(msg_end=True)
        self._feed(req, len(ids))
        if hdr < 0: self._checkpoint()                            # raw completion: no header, keep the prompt end
        t_gen = time.time()
        STATUS.update(what='generating', done=0, total=0)
        gen = torch.Generator().manual_seed(int(req['seed'])) if isinstance(req.get('seed'), int) else None
        max_new = max(1, min(req['max_tokens'], self.a.ctx - 16 - len(self.ctx)))
        budget = req['think_budget']
        out, text, dec_from = [], '', 0
        in_think = req['think'] and req['raw'] is None
        think_open = in_think                                     # still inside <think> (the prompt opened it)
        think_end = -1                                            # index of '</think>' in text
        reasoning, content, sent_r, sent_c = '', '', 0, 0
        finish, stop_hit = 'length', None
        stops, tools = req['stop'], req['tools']
        while len(out) < max_new:
            if cancel.is_set(): finish = 'cancelled'; break
            cur = self._sample(req['temperature'], req['top_p'], req['top_k'], req['min_p'], gen)
            self.e.step(cur, len(self.ctx)); self.ctx.append(cur)
            if cur in self.end_ids: finish = 'stop'; break
            out.append(cur)
            if think_open and budget >= 0 and len(out) >= budget:   # thinking budget used up: close the think block
                for t in self.close_think:
                    self.e.step(t, len(self.ctx)); self.ctx.append(t); out.append(t)
                think_open = False
            chunk = self.tok.decode(out[dec_from:])
            if chunk.endswith('�'): continue              # an incomplete UTF-8 character: wait for the next token
            text += chunk; dec_from = len(out)
            if in_think and think_end < 0:
                think_end = text.find('</think>', max(0, len(text) - len(chunk) - 8))
                if think_end >= 0: think_open = False
            if in_think:
                reasoning = text if think_end < 0 else text[:think_end]
                content = '' if think_end < 0 else text[think_end + 8:].lstrip()
            else:
                content = text.lstrip()
            if stops and content:
                hits = [(content.find(s), s) for s in stops if s in content[-(len(chunk) + max(map(len, stops)) + 1):] or s in content]
                hits = [h for h in hits if h[0] >= 0]
                if hits:
                    pos, stop_hit = min(hits); content = content[:pos]; finish = 'stop'; break
            r = reasoning.lstrip('\n')
            if think_open: r = r[:_held(r, '</think>')]
            if len(r) > sent_r: emit('reasoning', r[sent_r:]); sent_r = len(r)
            k = content.find('<tool_call>') if tools else -1
            vis = content[:k].rstrip() if k >= 0 else (content[:_held(content, '<tool_call>')] if tools else content)
            if stops: vis = vis[:min(_held(vis, s) for s in stops)]
            if len(vis) > sent_c: emit('content', vis[sent_c:]); sent_c = len(vis)
        if self.ctx[-1] not in self.end_ids:                     # cut, stopped or cancelled: close the turn in the cache
            self.e.step(self.end_id, len(self.ctx)); self.ctx.append(self.end_id)
        calls = parse_calls(content, tools) if tools and '<tool_call>' in content else []
        k = content.find('<tool_call>') if tools else -1
        vis = (content if k < 0 else content[:k]).rstrip()
        if len(vis) > sent_c: emit('content', vis[sent_c:])
        if calls and finish == 'stop': finish = 'tool_calls'
        n_gen = len(out)
        dt = max(1e-6, time.time() - t_gen)
        with self.lock:
            self.stats['requests'] += 1; self.stats['prompt_tokens'] += len(ids); self.stats['cached_tokens'] += cached
            self.stats['gen_tokens'] += n_gen; self.stats['last_tok_s'] = round(n_gen / dt, 1)
        log(f'[{req.get("rid", "-")}] {len(ids)} prompt tokens ({cached} cached via {how}), read {t_gen - t_start:.1f}s; '
            f'{n_gen} generated, {n_gen / dt:.1f} tok/s; finish {finish}' + (f', {len(calls)} tool call(s)' if calls else ''))
        return {'reasoning': reasoning.strip() if in_think else '', 'content': vis, 'calls': calls, 'finish': finish,
                'stop_hit': stop_hit, 'prompt_tokens': len(ids), 'cached_tokens': cached, 'completion_tokens': n_gen}


# ====================================================================================================== HTTP
DISCONNECT = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = f'ornith-serve/{__version__}'
    S = None   # Server

    def log_message(self, *a): pass

    # ---------------------------------------------------------------- basics
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b)))
        self.send_header('Access-Control-Allow-Origin', '*')
        if code == 503: self.send_header('Retry-After', '5')
        self.end_headers(); self.wfile.write(b)

    def _err(self, code, msg, anthropic=False):
        kind = {400: 'invalid_request_error', 401: 'authentication_error', 404: 'not_found_error', 413: 'request_too_large',
                429: 'rate_limit_error', 503: 'overloaded_error'}.get(code, 'api_error')
        if code >= 500 and code != 503:
            with self.S.lock: self.S.stats['errors'] += 1
        if self.path.startswith('/api/'): self._send(code, {'error': msg})          # Ollama's error shape
        elif anthropic: self._send(code, {'type': 'error', 'error': {'type': kind, 'message': msg}})
        else: self._send(code, {'error': {'message': msg, 'type': kind, 'code': code}})

    def _auth(self):
        key = self.S.a.api_key
        if not key: return True
        got = self.headers.get('x-api-key') or (self.headers.get('Authorization') or '').removeprefix('Bearer ').strip()
        return got == key

    def _body(self):
        if 'chunked' in (self.headers.get('Transfer-Encoding') or '').lower():
            data = b''
            while True:
                n = int(self.rfile.readline().split(b';')[0].strip() or b'0', 16)
                if n == 0: self.rfile.readline(); break
                data += self.rfile.read(n); self.rfile.readline()
                if len(data) > MAX_BODY: raise OverflowError
            return data
        n = int(self.headers.get('Content-Length') or 0)
        if n > MAX_BODY: raise OverflowError
        return self.rfile.read(n)

    def do_OPTIONS(self):
        self.send_response(204)
        for h, v in (('Access-Control-Allow-Origin', '*'), ('Access-Control-Allow-Methods', 'GET, POST, OPTIONS'),
                     ('Access-Control-Allow-Headers', '*'), ('Content-Length', '0')):
            self.send_header(h, v)
        self.end_headers()

    def do_GET(self):
        p = self.path.split('?')[0].rstrip('/')
        if p == '/panel' or (p == '' and 'text/html' in (self.headers.get('Accept') or '')):   # settings panel for browsers
            try:
                page = open(os.path.join(HERE, 'panel.html'), 'rb').read()
            except OSError:
                return self._err(404, 'panel.html is missing')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8'); self.send_header('Content-Length', str(len(page)))
            self.end_headers(); self.wfile.write(page); return
        if p == '':
            return self._send(200, {'name': 'ornith-serve', 'version': __version__, 'model': MODEL_ID, 'status': STATUS['state'],
                                    'openai_base_url': f'http://{self.headers.get("Host")}/v1', 'anthropic_base_url': f'http://{self.headers.get("Host")}',
                                    'endpoints': ['GET /v1/models', 'POST /v1/chat/completions', 'POST /v1/completions', 'POST /v1/messages',
                                                  'POST /v1/messages/count_tokens', 'GET /health', 'GET /ornith/status', 'POST /ornith/config']})
        if p == '/health':
            return self._send(200 if self.S.ready.is_set() else 503, {'status': STATUS['state'], 'what': STATUS['what'],
                                                                      'done': STATUS['done'], 'total': STATUS['total']})
        if not self._auth(): return self._err(401, 'invalid API key')
        info = {'id': MODEL_ID, 'object': 'model', 'created': 0, 'owned_by': 'local', 'type': 'model',
                'display_name': 'Ornith 1.5 35B-A3B (GTX 1070)', 'context_length': self.S.a.ctx, 'max_output_tokens': self.S.a.max_tokens}
        if p in ('/v1/models', '/models'): return self._send(200, {'object': 'list', 'data': [info], 'has_more': False})
        if p.startswith('/v1/models/'): return self._send(200, info)
        if p == '/ornith/status': return self._send(200, self.S.status())
        if p == '/api/version': return self._send(200, {'version': '0.12.0'})
        if p == '/api/tags': return self._send(200, {'models': [self._ollama_model()]})
        if p == '/api/ps': return self._send(200, {'models': [dict(self._ollama_model(), size_vram=7 * 2 ** 30, expires_at='2099-01-01T00:00:00Z')]
                                                    if self.S.ready.is_set() else []})
        self._err(404, f'unknown endpoint GET {p}')

    # ---------------------------------------------------------------- Ollama
    def _ollama_model(self):
        return {'name': MODEL_ID + ':latest', 'model': MODEL_ID + ':latest', 'modified_at': '2026-11-01T00:00:00Z', 'size': 23 * 2 ** 30,
                'digest': hashlib.sha256(MODEL_ID.encode()).hexdigest(),
                'details': {'parent_model': '', 'format': 'safetensors', 'family': 'qwen35moe', 'families': ['qwen35moe'],
                            'parameter_size': '35B', 'quantization_level': 'NVFP4'}}

    def _ollama_show(self):
        return self._send(200, {'modelfile': f'FROM {MODEL_ID}', 'parameters': f'temperature {self.S.a.temperature}', 'template': '{{ .Prompt }}',
                                'details': self._ollama_model()['details'],
                                'model_info': {'general.architecture': 'qwen35moe', 'general.parameter_count': 35_000_000_000,
                                               'qwen35moe.context_length': self.S.a.ctx},
                                'capabilities': ['completion', 'tools', 'thinking', 'vision']})

    def _ollama(self, body, generate=False):
        ob = ollama_to_openai(body, generate)
        if generate and body.get('raw'):
            raw = body.get('prompt') or ''
            if not raw: raise BadRequest('prompt must not be empty')
            req = build_request(self.S.a, dict(ob, max_tokens=ob.get('max_tokens') or 256), [], [])
            req['think'] = False
        else:
            msgs, tools = openai_to_chat(ob)
            req, raw = build_request(self.S.a, ob, msgs, tools), None
        try:
            req = self._prepare_req(req, raw)
        except QueueFull:
            return self._send(429, {'error': 'too many requests are waiting'})
        model, t0 = body.get('model') or MODEL_ID + ':latest', time.time()
        now = lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        key = 'response' if generate else 'message'

        def piece(content='', thinking=''):
            if generate: return {'model': model, 'created_at': now(), 'response': content, **({'thinking': thinking} if thinking else {}), 'done': False}
            m = {'role': 'assistant', 'content': content}
            if thinking: m['thinking'] = thinking
            return {'model': model, 'created_at': now(), 'message': m, 'done': False}

        def final(r, streamed):
            out = piece('' if streamed else r['content'], '' if streamed or req['hide_reasoning'] else r['reasoning'])
            if r['calls'] and not generate:
                out['message']['tool_calls'] = [{'function': {'name': n, 'arguments': a}} for n, a in r['calls']]
            ns = lambda s: int(s * 1e9)
            out.update(done=True, done_reason={'length': 'length'}.get(r['finish'], 'stop'), total_duration=ns(time.time() - t0),
                       load_duration=0, prompt_eval_count=r['prompt_tokens'] - r['cached_tokens'], eval_count=r['completion_tokens'])
            if generate: out['context'] = []
            return out
        if body.get('stream') is False:                           # Ollama streams unless told otherwise
            r = self._run(req, lambda k, t: None)
            return self._send(200, final(r, False))
        self.send_response(200)
        self.send_header('Content-Type', 'application/x-ndjson'); self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Connection', 'close'); self.end_headers()
        self.close_connection = True

        def line(obj):
            self.wfile.write((json.dumps(obj) + '\n').encode()); self.wfile.flush()

        def ev(kind, text):
            if kind == 'content': line(piece(text))
            elif not req['hide_reasoning']: line(piece('', text))
        try:
            r = self._run(req, ev, lambda: line(piece()))
        except DISCONNECT:
            return
        except Exception as ex:
            line({'error': f'{type(ex).__name__}: {ex}'}); return
        line(final(r, True))

    def do_POST(self):
        p = self.path.split('?')[0].rstrip('/')
        anthropic = p.startswith('/v1/messages')
        if not self._auth(): return self._err(401, 'invalid API key', anthropic)
        try:
            raw = self._body()
            body = json.loads(raw or b'{}')
            if not isinstance(body, dict): raise ValueError
        except OverflowError:
            return self._err(413, f'request body larger than {MAX_BODY // 2 ** 20} MB', anthropic)
        except ValueError:
            return self._err(400, 'body is not a valid JSON object', anthropic)
        if not self.S.ready.is_set():
            return self._err(503, f'model is {STATUS["state"]}: {STATUS["what"]}', anthropic)
        try:
            if p in ('/v1/chat/completions', '/chat/completions'): return self._openai_chat(body)
            if p in ('/v1/completions', '/completions'): return self._openai_completion(body)
            if p == '/v1/messages': return self._anthropic(body)
            if p == '/v1/messages/count_tokens':
                msgs, tools = anthropic_to_chat(body)
                return self._send(200, {'input_tokens': self.S.count_tokens(msgs, tools)})
            if p == '/ornith/config': return self._send(200, self.S.call(self.S.configure, body))
            if p == '/api/chat': return self._ollama(body)
            if p == '/api/generate': return self._ollama(body, generate=True)
            if p == '/api/show': return self._ollama_show()
            if p in ('/api/embed', '/api/embeddings', '/api/pull', '/api/push', '/api/create', '/api/copy', '/api/delete'):
                return self._send(501, {'error': f'{p} is not supported by ornith-serve (one built-in model, no embeddings)'})
        except BadRequest as ex:
            return self._err(400, str(ex), anthropic)
        except DISCONNECT:
            return
        except Exception as ex:
            traceback.print_exc()
            return self._err(500, f'{type(ex).__name__}: {ex}', anthropic)
        self._err(404, f'unknown endpoint POST {p}', anthropic)

    # ---------------------------------------------------------------- running a request
    def _prepare_req(self, req, raw=None):
        """validate + tokenize on this thread, so errors come back as a clean 400 before any streaming starts"""
        if self.S.queue_len >= self.S.a.max_queue: raise QueueFull
        req['rid'] = uuid.uuid4().hex[:8]
        return self.S.render(req, raw)

    def _run(self, req, on_event, on_idle=None):
        """run the request on the GPU thread; on_event(kind, text) is called on this thread, on_idle() every KEEPALIVE_S
        seconds without events (streaming pings). A client that disconnects cancels the request."""
        evq, cancel = queue.Queue(), threading.Event()
        with self.S.lock: self.S.queue_len += 1
        try:
            out = self.S.submit(self.S.generate, req, lambda k, t: evq.put((k, t)), cancel)
            last = time.time()
            while True:
                try:
                    kind, val = out.get(timeout=0.05)
                except queue.Empty:
                    kind = None
                try:
                    while not evq.empty():
                        on_event(*evq.get()); last = time.time()
                    if kind is None and on_idle and time.time() - last > KEEPALIVE_S:
                        on_idle(); last = time.time()
                except DISCONNECT:
                    cancel.set(); out.get(); raise
                if kind is not None: break
        finally:
            with self.S.lock: self.S.queue_len -= 1
        if kind == 'err': raise val
        return val

    def _sse_start(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream'); self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*'); self.send_header('Connection', 'close')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()
        self.close_connection = True

    def _sse(self, data, event=None):
        msg = (f'event: {event}\n' if event else '') + f'data: {data if isinstance(data, str) else json.dumps(data)}\n\n'
        self.wfile.write(msg.encode()); self.wfile.flush()

    def _ping(self):
        self.wfile.write(b': ping\n\n'); self.wfile.flush()

    # ---------------------------------------------------------------- OpenAI
    def _openai_chat(self, body):
        msgs, tools = openai_to_chat(body)
        try:
            req = self._prepare_req(build_request(self.S.a, body, msgs, tools))
        except QueueFull:
            return self._err(429, f'{self.S.queue_len} requests are already waiting (--max-queue {self.S.a.max_queue})')
        cid, created = 'chatcmpl-' + req['rid'] + uuid.uuid4().hex[:16], int(time.time())
        model = body.get('model') or MODEL_ID
        hide = req['hide_reasoning']
        fin = {'cancelled': 'stop'}

        def usage(r):
            return {'prompt_tokens': r['prompt_tokens'], 'completion_tokens': r['completion_tokens'],
                    'total_tokens': r['prompt_tokens'] + r['completion_tokens'],
                    'prompt_tokens_details': {'cached_tokens': r['cached_tokens']}}

        def tool_calls(r):
            return [{'id': 'call_' + uuid.uuid4().hex[:24], 'type': 'function', 'function': {'name': n, 'arguments': json.dumps(a, ensure_ascii=False)}}
                    for n, a in r['calls']]
        if not body.get('stream'):
            r = self._run(req, lambda k, t: None)
            msg = {'role': 'assistant', 'content': r['content'] or (None if r['calls'] else '')}
            if r['reasoning'] and not hide: msg['reasoning_content'] = msg['reasoning'] = r['reasoning']
            if r['calls']: msg['tool_calls'] = tool_calls(r)
            return self._send(200, {'id': cid, 'object': 'chat.completion', 'created': created, 'model': model,
                                    'choices': [{'index': 0, 'message': msg, 'logprobs': None, 'finish_reason': fin.get(r['finish'], r['finish'])}],
                                    'usage': usage(r)})
        self._sse_start()

        def chunk(delta, finish=None, **extra):
            return {'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model,
                    'choices': [{'index': 0, 'delta': delta, 'logprobs': None, 'finish_reason': finish}], **extra}
        self._sse(chunk({'role': 'assistant', 'content': ''}))

        def ev(kind, text):
            if kind == 'reasoning' and not hide: self._sse(chunk({'reasoning_content': text, 'reasoning': text}))
            elif kind == 'content': self._sse(chunk({'content': text}))
        try:
            r = self._run(req, ev, self._ping)
        except DISCONNECT:
            return
        except Exception as ex:
            if not isinstance(ex, BadRequest): traceback.print_exc()
            self._sse({'error': {'message': f'{type(ex).__name__}: {ex}', 'type': 'server_error'}}); self._sse('[DONE]'); return
        if r['calls']:
            self._sse(chunk({'tool_calls': [dict(c, index=i) for i, c in enumerate(tool_calls(r))]}))
        self._sse(chunk({}, fin.get(r['finish'], r['finish'])))
        if (body.get('stream_options') or {}).get('include_usage'):
            self._sse({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [], 'usage': usage(r)})
        self._sse('[DONE]')

    def _openai_completion(self, body):
        prompt = body.get('prompt', '')
        if isinstance(prompt, list) and all(isinstance(x, str) for x in prompt): prompt = ''.join(prompt)
        if not isinstance(prompt, str) or not prompt: raise BadRequest('prompt must be a non-empty string')
        req = build_request(self.S.a, dict(body, max_tokens=body.get('max_tokens') or 256), [], [])
        req['think'] = False
        try:
            req = self._prepare_req(req, raw=prompt)
        except QueueFull:
            return self._err(429, 'too many requests are waiting')
        cid, created, model = 'cmpl-' + req['rid'], int(time.time()), body.get('model') or MODEL_ID
        fin = {'cancelled': 'stop'}

        def piece(text, finish=None):
            return {'id': cid, 'object': 'text_completion', 'created': created, 'model': model,
                    'choices': [{'index': 0, 'text': text, 'logprobs': None, 'finish_reason': finish}]}
        if not body.get('stream'):
            r = self._run(req, lambda k, t: None)
            out = piece(r['content'], fin.get(r['finish'], r['finish']))
            out['usage'] = {'prompt_tokens': r['prompt_tokens'], 'completion_tokens': r['completion_tokens'],
                            'total_tokens': r['prompt_tokens'] + r['completion_tokens']}
            return self._send(200, out)
        self._sse_start()
        try:
            r = self._run(req, lambda k, t: k == 'content' and self._sse(piece(t)), self._ping)
        except DISCONNECT:
            return
        except Exception as ex:
            self._sse({'error': {'message': f'{type(ex).__name__}: {ex}', 'type': 'server_error'}}); self._sse('[DONE]'); return
        self._sse(piece('', fin.get(r['finish'], r['finish'])))
        self._sse('[DONE]')

    # ---------------------------------------------------------------- Anthropic
    def _anthropic(self, body):
        msgs, tools = anthropic_to_chat(body)
        try:
            req = self._prepare_req(build_request(self.S.a, body, msgs, tools, anthropic=True))
        except QueueFull:
            return self._err(429, 'too many requests are waiting', True)
        mid, model = 'msg_' + req['rid'] + uuid.uuid4().hex[:16], body.get('model') or MODEL_ID

        def stop_of(r):
            if r['stop_hit']: return 'stop_sequence', r['stop_hit']
            return {'stop': 'end_turn', 'tool_calls': 'tool_use', 'length': 'max_tokens', 'cancelled': 'end_turn'}[r['finish']], None

        def usage(r):
            return {'input_tokens': r['prompt_tokens'] - r['cached_tokens'], 'output_tokens': r['completion_tokens'],
                    'cache_read_input_tokens': r['cached_tokens'], 'cache_creation_input_tokens': 0}
        if not body.get('stream'):
            r = self._run(req, lambda k, t: None)
            content = []
            if r['reasoning']: content.append({'type': 'thinking', 'thinking': r['reasoning'], 'signature': ''})
            if r['content']: content.append({'type': 'text', 'text': r['content']})
            for n, a in r['calls']: content.append({'type': 'tool_use', 'id': 'toolu_' + uuid.uuid4().hex[:24], 'name': n, 'input': a})
            stop, seq = stop_of(r)
            return self._send(200, {'id': mid, 'type': 'message', 'role': 'assistant', 'model': model, 'content': content,
                                    'stop_reason': stop, 'stop_sequence': seq, 'usage': usage(r)})
        self._sse_start()
        self._sse({'type': 'message_start', 'message': {'id': mid, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [],
                                                        'stop_reason': None, 'stop_sequence': None,
                                                        'usage': {'input_tokens': len(req['ids']), 'output_tokens': 0}}}, 'message_start')
        st = {'idx': -1, 'kind': None}

        def close_block():
            if st['kind'] is None: return
            if st['kind'] == 'reasoning':
                self._sse({'type': 'content_block_delta', 'index': st['idx'], 'delta': {'type': 'signature_delta', 'signature': ''}}, 'content_block_delta')
            self._sse({'type': 'content_block_stop', 'index': st['idx']}, 'content_block_stop')
            st['kind'] = None

        def ev(kind, text):
            if st['kind'] != kind:
                close_block()
                st['idx'] += 1; st['kind'] = kind
                blk = {'type': 'thinking', 'thinking': ''} if kind == 'reasoning' else {'type': 'text', 'text': ''}
                self._sse({'type': 'content_block_start', 'index': st['idx'], 'content_block': blk}, 'content_block_start')
            d = {'type': 'thinking_delta', 'thinking': text} if kind == 'reasoning' else {'type': 'text_delta', 'text': text}
            self._sse({'type': 'content_block_delta', 'index': st['idx'], 'delta': d}, 'content_block_delta')
        try:
            r = self._run(req, ev, lambda: self._sse({'type': 'ping'}, 'ping'))
        except DISCONNECT:
            return
        except Exception as ex:
            if not isinstance(ex, BadRequest): traceback.print_exc()
            self._sse({'type': 'error', 'error': {'type': 'api_error', 'message': f'{type(ex).__name__}: {ex}'}}, 'error'); return
        close_block()
        for n, a in r['calls']:
            st['idx'] += 1
            self._sse({'type': 'content_block_start', 'index': st['idx'],
                       'content_block': {'type': 'tool_use', 'id': 'toolu_' + uuid.uuid4().hex[:24], 'name': n, 'input': {}}}, 'content_block_start')
            self._sse({'type': 'content_block_delta', 'index': st['idx'],
                       'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(a, ensure_ascii=False)}}, 'content_block_delta')
            self._sse({'type': 'content_block_stop', 'index': st['idx']}, 'content_block_stop')
        stop, seq = stop_of(r)
        self._sse({'type': 'message_delta', 'delta': {'stop_reason': stop, 'stop_sequence': seq}, 'usage': usage(r)}, 'message_delta')
        self._sse({'type': 'message_stop'}, 'message_stop')


class QueueFull(Exception):
    pass


# ====================================================================================================== main
def parse_args(argv=None):
    home = os.path.join(os.path.expanduser('~'), '.ornith-serve')
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=os.path.join(home, 'config.json'))
    known, _ = pre.parse_known_args(argv)
    ap = argparse.ArgumentParser(description='Ornith-1.5-35B-A3B on a GTX 1070 as an OpenAI- and Anthropic-compatible server',
                                 epilog='All options can also be set in a JSON config file (--config, keys like "ram_gb"); '
                                        'the command line wins.')
    ap.add_argument('--config', default=known.config, help='JSON file with defaults for these options (default ~/.ornith-serve/config.json)')
    ap.add_argument('--host', default='127.0.0.1', help='address to listen on (0.0.0.0 = whole network; then set --api-key)')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--api-key', default=os.environ.get('ORNITH_API_KEY', ''), help='require this key (Bearer or x-api-key); env ORNITH_API_KEY')
    ap.add_argument('--ram-gb', type=float, default=13.5, help='pinned RAM for experts, 0.5-13.5 (more = faster)')
    ap.add_argument('--free-gb', type=float, default=1.0, help='VRAM left free for Windows and other programs')
    ap.add_argument('--ctx', type=int, default=184320, help='maximum context in tokens')
    ap.add_argument('--quality', default='turbo', choices=list(QUALITY) + list(QUALITY_ALIAS), help='exact = lossless; fast/turbo skip small non-VRAM experts')
    ap.add_argument('--langs', default='auto', choices=LANGS, help='output scripts: auto = CJK/other only once they appear')
    ap.add_argument('--think', default='on', choices=['on', 'off'], help='thinking default for requests that do not say')
    ap.add_argument('--temperature', type=float, default=0.6, help='default sampling temperature')
    ap.add_argument('--max-tokens', type=int, default=32768, help='default answer limit')
    ap.add_argument('--max-queue', type=int, default=16, help='requests that may wait for the GPU; more get 429')
    ap.add_argument('--max-image-pixels', type=int, default=1024 * 1024, help='images are scaled down to this many pixels (1 MP ~ 1000 tokens)')
    ap.add_argument('--cache', default='on', choices=['on', 'off'], help='prefix cache (checkpoints, parking, snapshots); off = read every prompt in full')
    ap.add_argument('--park-gb', type=float, default=3.0, help='RAM for exact copies of recent conversations')
    ap.add_argument('--live-ckpts', type=int, default=4, help='checkpoints kept inside the current conversation')
    ap.add_argument('--snapshots', default=os.path.join(home, 'snapshots'), help="folder for compressed conversation snapshots ('' = off)")
    ap.add_argument('--snap-cap-gb', type=float, default=20.0, help='disk cap for all snapshots (oldest go first)')
    ap.add_argument('--snap-min', type=int, default=8192, help='only snapshot conversations with at least this many tokens')
    ap.add_argument('--force', action='store_true', help='start on a GPU other than a GTX 1070-class Pascal card')
    ap.add_argument('--bench', action='store_true', help='measure start-up, prompt reading and answer speed (all quality modes), then exit')
    ap.add_argument('--fake', action='store_true', help='developer mode: no GPU or weights; a fake engine answers with a fixed text')
    if os.path.isfile(known.config):
        try:
            cfg = json.load(open(known.config, encoding='utf-8'))
        except ValueError as ex:
            raise SystemExit(f'config file {known.config} is not valid JSON: {ex}')
        dests = {a.dest for a in ap._actions}
        bad = [k for k in cfg if k.replace('-', '_') not in dests]
        if bad: raise SystemExit(f'config file {known.config}: unknown option(s) {bad}')
        for k in ('think', 'cache'):
            if k in cfg and isinstance(cfg[k], bool): cfg[k] = 'on' if cfg[k] else 'off'
        ap.set_defaults(**{k.replace('-', '_'): v for k, v in cfg.items()})
    a = ap.parse_args(argv)
    if a.quality not in list(QUALITY) + list(QUALITY_ALIAS): ap.error(f'--quality must be one of {list(QUALITY)}')
    if a.langs not in LANGS: ap.error(f'--langs must be one of {list(LANGS)}')
    if a.host not in ('127.0.0.1', 'localhost', '::1') and not a.api_key:
        log(f'warning: listening on {a.host} without --api-key: everyone in the network can use the model')
    a.home = home
    a.think = a.think in ('on', True)
    a.cache = a.cache in ('on', True)
    if not a.cache: a.snapshots = ''
    a.quality = QUALITY_ALIAS.get(a.quality, a.quality)
    a.ram_gb = min(13.5, max(0.5, a.ram_gb))
    a.exit_on_error = False
    os.makedirs(home, exist_ok=True)
    return a


class HTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False                                   # Windows: SO_REUSEADDR would let a second server share the port
    daemon_threads = True

    def server_bind(self):
        import socket
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'): self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        elif os.name != 'nt': self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # POSIX: fast restart is safe
        super().server_bind()


def serve(a, engine=None, prefill=None, tok=None):
    """bind the port first (a busy port fails at once, before minutes of model loading), then load in the background"""
    handler = type('BoundHandler', (Handler,), {})
    try:
        httpd = HTTPServer((a.host, a.port), handler)
    except OSError as ex:
        raise SystemExit(f'cannot listen on {a.host}:{a.port}: {ex} (another server running? use --port)')
    if getattr(a, 'fake', False) and engine is None:
        from fake_engine import FakeEngine, FakePrefill, FakeVision, tokenizer
        tok = tokenizer(); engine = FakeEngine(tok, max_ctx=a.ctx); prefill = FakePrefill(engine)
        a.snapshots = ''
        log('developer mode: fake engine, no model')
    S = Server(a, engine, prefill, tok)
    if getattr(a, 'fake', False): S.vision = FakeVision()
    handler.S = S
    log(f'ornith-serve {__version__} listening on http://{a.host}:{a.port} (loading the model; GET /health shows progress)')
    return S, httpd


BENCH_TEXT = ('The history of railways began with wooden wagonways in the sixteenth century and grew into the steam age. ' * 40)


def bench(a):
    """--bench: start the engine, measure start-up, prompt reading and answer speed in each quality mode, print a report"""
    import platform, threading as th
    a.snapshots, a.port = '', 0
    S, httpd = serve(a)
    t0 = time.time()
    while not S.ready.is_set():
        if STATUS['state'] == 'error': raise SystemExit('start failed: ' + STATUS['what'])
        time.sleep(0.5)
    rep = {'version': __version__, 'startup_s': round(time.time() - t0, 1), 'ram_gb': a.ram_gb, 'os': platform.platform()}
    try:
        import torch
        rep['gpu'] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    noop = th.Event()

    def ask(text, n, quality):
        req = build_request(a, {'max_tokens': n, 'temperature': 0, 'reasoning_effort': 'none', 'ornith': {'quality': quality}},
                            [{'role': 'user', 'content': text}], [])
        S.render(req)
        S.call(S._prepare_fresh)
        t = time.time()
        r = S.call(S.generate, req, lambda k, x: None, noop)
        return r, time.time() - t
    ask('Hello', 8, 'exact')                                     # warm-up
    long_prompt = BENCH_TEXT + '\nSummarise this text in one sentence.'
    for q in ('exact', 'fast', 'turbo'):
        r, dt = ask('Write a long story about a lighthouse keeper.', 256, q)
        rep[f'decode_tok_s_{q}'] = round(r['completion_tokens'] / dt, 1)
        log(f'{q}: {rep[f"decode_tok_s_{q}"]} tok/s ({r["completion_tokens"]} tokens)')
    S.call(S._prepare_fresh)
    req = build_request(a, {'max_tokens': 1, 'temperature': 0, 'reasoning_effort': 'none'}, [{'role': 'user', 'content': long_prompt}], [])
    S.render(req)
    t = time.time(); S.call(S.generate, req, lambda k, x: None, noop)
    rep['prompt_tokens'] = len(req['ids']); rep['prefill_tok_s'] = round(len(req['ids']) / (time.time() - t), 1)
    print('\n' + json.dumps(rep, indent=1))
    print('\nPaste this block into an issue or discussion to compare your numbers.')
    httpd.server_close()
    S.call(S.shutdown)


def main():
    args = parse_args()
    args.exit_on_error = True
    if args.bench: return bench(args)
    srv, httpd = serve(args)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log('stopping (finishing the current answer and snapshot writes) ...')
    finally:
        httpd.server_close()
        if srv.ready.is_set(): srv.call(srv.shutdown)


if __name__ == '__main__':
    main()
