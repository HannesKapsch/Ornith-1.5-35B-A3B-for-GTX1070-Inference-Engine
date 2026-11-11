"""Pack all routed experts into one file: experts.bin = [layer][expert] blobs of BLOB bytes.
Blob = gate W | up W | down W | gate S | up S | down S | 3 x fp32 global scales (padded)."""
import os, sys, time
import numpy as np
import torch
from weights import Weights, LM

GW, DW = 512 * 1024, 2048 * 256          # NVFP4 bytes: gate/up [512, 2048/2], down [2048, 512/2]
GS, DS = 512 * 128, 2048 * 32            # fp8 scale bytes
OFF = dict(gw=0, uw=GW, dw=2 * GW, gs=2 * GW + DW, us=2 * GW + DW + GS, ds=2 * GW + DW + 2 * GS)
OFF['g2'] = OFF['ds'] + DS
BLOB = (OFF['g2'] + 12 + 255) // 256 * 256
L, E = 40, 256
OUT = os.environ.get('ORNITH_EXPERTS') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experts.bin')


def blob_of(W, p):
    b = np.zeros(BLOB, np.uint8)
    for part, key in (('gate_proj', 'g'), ('up_proj', 'u'), ('down_proj', 'd')):
        w = W.raw(f'{p}.{part}.weight').numpy().ravel()
        s = W.raw(f'{p}.{part}.weight_scale').view(torch.uint8).numpy().ravel()
        b[OFF[key + 'w']:OFF[key + 'w'] + w.size] = w
        b[OFF[key + 's']:OFF[key + 's'] + s.size] = s
    g2 = np.array([W.raw(f'{p}.{q}.weight_scale_2').float().item() for q in ('gate_proj', 'up_proj', 'down_proj')], np.float32)
    b[OFF['g2']:OFF['g2'] + 12] = g2.view(np.uint8)
    return b


if __name__ == '__main__':
    W = Weights()
    if os.path.exists(OUT) and os.path.getsize(OUT) == L * E * BLOB:
        print('already packed'); sys.exit()
    t0 = time.time()
    with open(OUT + '.tmp', 'wb') as f:
        for l in range(L):
            f.write(b''.join(blob_of(W, f'{LM}layers.{l}.mlp.experts.{e}').tobytes() for e in range(E)))
            print(f'{l + 1}/{L} layers packed ({time.time() - t0:.0f}s)', flush=True)
    os.replace(OUT + '.tmp', OUT)
