"""Compressed session snapshots: KV cache + DeltaNet state of a conversation, so reopening it takes seconds instead of a
prefill (150k tokens ~23 min). Lossy (~1-2 % perplexity at STEP 0.5).

Codec per full-attention layer, K/V and kv head: the int8 rows (x = int8 * scale per 32 dims) are un-rotated (K: RoPE),
projected on a per-session KLT basis (PCA, frozen after the first 4096 tokens), divided by their RMS, rounded with
STEP and stored as int8 component-major; the original fp16 scales are kept, so on load small errors round back to the
exact int8 value. Storage is per 4096-token chunk (like the KV arena), so after each answer only new chunks are added.
Folder <sid>.kv/: meta.json, basis.pt, c<k>.xz (full chunks), tail.xz (the last, partial chunk), state.pt (DeltaNet
state + conv state in fp16, RoPE state). The GPU does the transforms; the CPU only compresses (lzma) and writes files."""
import os, io, json, lzma, shutil, hashlib, threading, queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

STEP = float(os.environ.get('KVSNAP_STEP', 0.5))
FMT = 1
_pool = ThreadPoolExecutor(max(2, (os.cpu_count() or 4) - 2))
_jobs = queue.Queue()
_inflight = threading.BoundedSemaphore(6)          # chunk payloads waiting for compression (each ~45 MB raw)


def _worker():
    while True:
        fn, args = _jobs.get()
        try:
            fn(*args)
        except Exception:
            import traceback; traceback.print_exc()
        finally:
            _jobs.task_done()


threading.Thread(target=_worker, daemon=True).start()


def wait():
    """block until all background writes are on disk"""
    _jobs.join()


def ctx_hash(ids):
    return hashlib.sha1(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()


def _blocks(e, c):
    """(i, which, int8 values [2, CK, 256], fp16 scales [2, CK, 8]) of every full-attention layer i, K (0) / V (1), in chunk c"""
    nf, CK, CSB = e.nfull, e.CK, e.CSB
    kb, sb = 2 * CK * 256, 2 * CK * 16
    o = c * CSB
    for i in range(nf):
        for which in (0, 1):
            vo = o + which * nf * kb + i * kb
            so = o + 2 * nf * kb + which * nf * sb + i * sb
            yield i, which, e.arena[vo:vo + kb].view(torch.int8).view(2, CK, 256), e.arena[so:so + sb].view(torch.float16).view(2, CK, 8)


def _rope(e, x, rows, sign):
    """rotate (sign +1, like attn_prep) or un-rotate (sign -1) dims 0..63 of K rows x [m, 256] in place"""
    c, s = e.cos[rows], e.sin[rows]
    a, b = x[:, :32].clone(), x[:, 32:64].clone()
    x[:, :32] = a * c - sign * b * s
    x[:, 32:64] = b * c + sign * a * s
    return x


def _deq(v, s):
    return (v.float().view(-1, 8, 32) * s.float().unsqueeze(-1)).view(-1, 256)


def _basis(e, n):
    """per (layer, K/V, head): mean, KLT basis, rms from the first min(n, 16384) tokens"""
    m = min(n, 16384)
    B = {}
    acc = {}
    for c in range((m + e.CK - 1) // e.CK):
        k = min(e.CK, m - c * e.CK)
        rows = e.rp[c * e.CK:c * e.CK + k].long()
        for i, which, v, s in _blocks(e, c):
            for h in range(2):
                x = _deq(v[h, :k], s[h, :k])
                if which == 0: x = _rope(e, x, rows, -1)
                acc.setdefault((i, which, h), []).append(x)
    for key, xs in acc.items():
        x = torch.cat(xs)
        mu = x.mean(0)
        xc = x - mu
        rms = float(xc.pow(2).mean().sqrt()) or 1.0
        _, P = torch.linalg.eigh((xc.T @ xc) / len(xc))
        B[key] = (mu, P.flip(1).contiguous(), rms)
    return B


def _encode_chunk(e, B, c, k):
    """chunk c (first k tokens) -> raw bytes (q component-major int8 + fp16 scales, all blocks)"""
    rows = e.rp[c * e.CK:c * e.CK + k].long()
    parts = []
    for i, which, v, s in _blocks(e, c):
        for h in range(2):
            mu, P, rms = B[(i, which, h)]
            x = _deq(v[h, :k], s[h, :k])
            if which == 0: x = _rope(e, x, rows, -1)
            q = torch.round(((x - mu) @ P) / (rms * STEP)).clamp_(-127, 127).to(torch.int8)
            parts.append(q.T.contiguous().cpu().numpy().tobytes())
            parts.append(s[h, :k].contiguous().cpu().numpy().tobytes())
    return b''.join(parts)


def _decode_chunk(e, B, c, k, raw):
    rows = e.rp[c * e.CK:c * e.CK + k].long()
    off = 0
    for i, which, v, s in _blocks(e, c):
        for h in range(2):
            mu, P, rms = B[(i, which, h)]
            q = torch.frombuffer(bytearray(raw[off:off + 256 * k]), dtype=torch.int8).view(256, k).T.to(v.device); off += 256 * k
            sc = torch.frombuffer(bytearray(raw[off:off + 16 * k]), dtype=torch.float16).view(k, 8).to(v.device); off += 16 * k
            x = (q.float() * (rms * STEP)) @ P.T + mu
            if which == 0: x = _rope(e, x, rows, +1)
            v[h, :k] = torch.round(x.view(-1, 8, 32) / sc.float().unsqueeze(-1)).clamp_(-127, 127).to(torch.int8).view(k, 256)
            s[h, :k] = sc


def _put(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f: f.write(data)
    os.replace(tmp, path)


def _compress_write(path, raw):
    try:
        _put(path, lzma.compress(raw, preset=1))
    finally:
        _inflight.release()


def _submit(folder, name, raw, futs):
    """compress (.xz) or write a file in the pool; blocks while 6 payloads are already waiting (bounded RAM)"""
    path = os.path.join(folder, name)
    if name.endswith('.xz'):
        _inflight.acquire()
        futs.append(_pool.submit(_compress_write, path, raw))
    else:
        futs.append(_pool.submit(_put, path, raw))


def _finish_job(folder, futs, meta, remove):
    for fu in futs: fu.result()
    for name in remove:
        p = os.path.join(folder, name)
        if os.path.exists(p): os.remove(p)
    _put(os.path.join(folder, 'meta.json'), json.dumps(meta).encode())


def _read_meta(folder):
    try:
        with open(os.path.join(folder, 'meta.json')) as f:
            m = json.load(f)
        return m if m.get('fmt') == FMT and m.get('step') == STEP else None
    except (OSError, ValueError):
        return None


def save(e, folder, ids):
    """snapshot the engine state after the tokens ids (call on the engine thread, after an answer). The GPU part runs now,
    compression + writing in the background (call wait() before reading the folder from elsewhere)."""
    n = len(ids)
    if n == 0: return
    wait()                                                    # the previous snapshot of any session is complete on disk
    meta = _read_meta(folder)
    if meta and (meta['n'] > n or meta['prefix_hash'] != ctx_hash(ids[:meta['n']])):
        meta = None                                           # different / rewound conversation: start over
    if meta is None:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        meta = dict(fmt=FMT, step=STEP, n=0, chunks=0, frozen=False)
    futs = []
    if not meta['frozen']:                                    # basis from what we have; frozen after the first full chunk
        B = _basis(e, n)
        buf = io.BytesIO(); torch.save({k: (mu.cpu(), P.half().cpu(), rms) for k, (mu, P, rms) in B.items()}, buf)
        _submit(folder, 'basis.pt', buf.getvalue(), futs)
        meta['chunks'] = 0
        meta['frozen'] = n >= e.CK
        B = {k: (mu, P.half().float(), rms) for k, (mu, P, rms) in B.items()}   # encode with the basis as stored (fp16)
    else:
        B = load_basis(folder, e.arena.device)
    full = n // e.CK
    for c in range(meta['chunks'], full):
        _submit(folder, f'c{c}.xz', _encode_chunk(e, B, c, e.CK), futs)
    rest = n - full * e.CK
    remove = []
    if rest:
        _submit(folder, 'tail.xz', _encode_chunk(e, B, full, rest), futs)
    else:
        remove.append('tail.xz')
    buf = io.BytesIO()
    torch.save({'S': e.S.half().cpu(), 'conv': e.conv_state.half().cpu(), 'rope': e.rope_state(n)}, buf)
    _submit(folder, 'state.pt', buf.getvalue(), futs)
    meta.update(n=n, chunks=full, prefix_hash=ctx_hash(ids))
    _jobs.put((_finish_job, (folder, futs, dict(meta), remove)))


def load_basis(folder, device):
    d = torch.load(os.path.join(folder, 'basis.pt'))
    return {k: (mu.to(device), P.float().to(device), rms) for k, (mu, P, rms) in d.items()}


def load(e, folder, ids):
    """restore the engine state of ids from the snapshot; False if there is none that matches exactly"""
    wait()
    meta = _read_meta(folder)
    n = len(ids)
    if not meta or meta['n'] != n or meta['prefix_hash'] != ctx_hash(ids):
        return False
    dev = e.arena.device
    st = torch.load(os.path.join(folder, 'state.pt'))
    B = load_basis(folder, dev)
    e.reset(); e.ensure_kv(max(1, n))
    if st.get('rope') is not None: e.set_rope_state(st['rope'])
    e.S.copy_(st['S'].float()); e.conv_state.copy_(st['conv'].float())
    full = n // e.CK
    names = [f'c{c}.xz' for c in range(full)] + (['tail.xz'] if n - full * e.CK else [])
    raws = _pool.map(lambda nm: lzma.decompress(open(os.path.join(folder, nm), 'rb').read()), names)
    win0 = max(0, n - e.RING)                                 # positions that must also sit in the exact fp16 ring
    for c, raw in enumerate(raws):
        k = min(e.CK, n - c * e.CK)
        t0 = c * e.CK
        _decode_chunk(e, B, c, k, raw)
        lo = max(win0, t0)
        if lo < t0 + k:                                       # ring rows: the int8 values as the engine would read them
            pos = torch.arange(lo, t0 + k, device=dev) % e.RING
            for i, which, v, s in _blocks(e, c):
                ring = e.Kr if which == 0 else e.Vr
                for h in range(2):
                    ring[i, h, pos] = _deq(v[h, lo - t0:k], s[h, lo - t0:k]).half()
    torch.cuda.synchronize()
    return True


def size(folder):
    try:
        return sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder))
    except OSError:
        return 0


def drop(folder):
    """delete a snapshot (after its background writes)"""
    _jobs.put((shutil.rmtree, (folder, True)))


def enforce_cap(root, cap_bytes, keep=None):
    """delete the least recently written snapshots (*.kv folders in root) until all together fit in cap_bytes; never `keep`"""
    def job():
        snaps = []
        for f in os.listdir(root):
            p = os.path.join(root, f)
            if f.endswith('.kv') and os.path.isdir(p):
                m = os.path.join(p, 'meta.json')
                snaps.append((os.path.getmtime(m) if os.path.exists(m) else 0.0, size(p), p))
        total = sum(sz for _, sz, _ in snaps)
        for _, sz, p in sorted(snaps):
            if total <= cap_bytes: break
            if keep and os.path.normcase(os.path.abspath(p)) == os.path.normcase(os.path.abspath(keep)): continue
            shutil.rmtree(p, ignore_errors=True)
            total -= sz
    _jobs.put((job, ()))
