"""A fake engine with the interface of the real one, for tests and for working on the server without a GPU or the weights
(python ornith_serve.py --fake). It needs only the tokenizer files.

Its 'state' is a running hash of (position, token) and its KV cache stores the token of each position, so tests can check
that the prefix cache restored exactly the state of a conversation. It answers with the text given to say() (default: a
fixed note), token by token through the normal sampling path."""
import os, time
import torch
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
M = (1 << 61) - 1


def tokenizer():
    """the model's tokenizer: from an installation (install.json / ORNITH_MODEL), else downloaded from Hugging Face (~20 MB)"""
    dirs = [os.environ.get('ORNITH_MODEL'), os.path.join(HERE, '..', 'ornith', 'model')]
    try:
        import json
        dirs.insert(0, json.load(open(os.path.join(HERE, 'install.json'), encoding='utf-8')).get('model_dir'))
    except (OSError, ValueError):
        pass
    for d in dirs:
        if d and os.path.isfile(os.path.join(d, 'tokenizer.json')): return AutoTokenizer.from_pretrained(d)
    return AutoTokenizer.from_pretrained('ornith-ai/Ornith-1.5-35B-A3B-NVFP4')


def state_hash(ids):
    h = 0
    for p, t in enumerate(ids): h = (h * 1000003 + t * 7919 + p) % M
    return h


class FakeEngine:
    CK, RING = 4096, 6144
    DEFAULT = 'This answer comes from the fake engine of ornith-serve (no model loaded).'

    def __init__(self, tok, max_ctx=65536):
        self.tok, self.max_ctx, self.V = tok, max_ctx, len(tok)
        self.END = tok.convert_tokens_to_ids('<|im_end|>')
        self.IM_START = tok.convert_tokens_to_ids('<|im_start|>')
        self.PAD = tok.convert_tokens_to_ids('<|image_pad|>')
        self.CSB = self.CK * 4                                       # one int32 token per position
        self.arena = torch.zeros(max_ctx * 4 + self.CSB, dtype=torch.uint8)
        self.S = torch.zeros(1, dtype=torch.int64)                  # running hash of (pos, tok)
        self.conv_state = torch.zeros(1, dtype=torch.int64)         # number of tokens read
        self.Kr = torch.zeros(self.RING, dtype=torch.int64); self.Vr = self.Kr.clone()
        self.logits = torch.zeros(self.V)
        self.kv_chunks, self.steps, self.quality, self.delay = 0, 0, None, 0.0
        self.image_ropes, self.emb_rows = [], 0
        self.say('x\n</think>\n\n' + self.DEFAULT)

    def kv(self): return self.arena[:self.max_ctx * 4].view(torch.int32)

    def reset(self):
        self.S.zero_(); self.conv_state.zero_(); self.Kr.zero_(); self.kv_chunks = 0

    def ensure_kv(self, n): self.kv_chunks = max(self.kv_chunks, (n + self.CK - 1) // self.CK)

    def rope_state(self, n): return {'n': n}

    def set_rope_state(self, st): pass

    def set_quality(self, a, b): self.quality = b

    def add_image_rope(self, p0, gh, gw): self.image_ropes.append((p0, gh, gw))

    def set_langs(self, **kw): self.langs = kw

    def vram_split(self): return self.kv_chunks * self.CSB / 2 ** 30, 1.0

    def step(self, tok, pos):
        assert int(self.conv_state) == pos, f'state holds {int(self.conv_state)} tokens, asked to read position {pos}'
        self.ensure_kv(pos + 1)
        self.kv()[pos] = tok
        self.S[0] = (int(self.S) * 1000003 + tok * 7919 + pos) % M
        self.conv_state[0] = pos + 1
        self.Kr[pos % self.RING] = tok
        self.steps += 1
        if self.delay: time.sleep(self.delay)
        if tok == self.IM_START: self.gi = 0                        # a new prompt is read: answer from the start again
        elif self.gi < len(self.script) and tok == self.script[self.gi]: self.gi += 1
        self.logits.fill_(0); self.logits[self.script[min(self.gi, len(self.script) - 1)]] = 1

    def say(self, text):
        """what the fake model answers next (the prompt ends inside <think>, so start with 'x\\n</think>\\n\\n')"""
        self.script, self.gi = self.tok(text, add_special_tokens=False).input_ids + [self.END], 0
        self.logits.fill_(0); self.logits[self.script[0]] = 1


class FakePrefill:
    def __init__(self, e): self.e = e

    def run(self, ids, p0=0, log=None, emb=None):
        if emb is not None:
            assert len(emb[0]) == len(emb[1]) and all(ids[int(j)] == self.e.PAD for j in emb[0]), 'embedding rows must sit on image pads'
            self.e.emb_rows += len(emb[0])
        for i, t in enumerate(ids): self.e.step(t, p0 + i)


class FakeVision:
    """grid from the image size: (h // 100) x (w // 100) tokens"""
    calls = 0

    def encode(self, img):
        FakeVision.calls += 1
        gh, gw = max(1, img.size[1] // 100), max(1, img.size[0] // 100)
        return torch.zeros(gh * gw, 2048), (gh, gw)
