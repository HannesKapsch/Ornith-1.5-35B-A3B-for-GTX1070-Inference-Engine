"""Tensor access + dequantisation for Ornith-1.5-35B-A3B NVFP4 (modelopt mixed precision).

- experts / shared expert / lm_head: NVFP4 -> uint8 [N, K/2] (low nibble = even k),
  fp8-e4m3 scale [N, K/16], fp32 scale_2.   W = E2M1[code] * scale * scale_2
- attention + linear-attention projections: FP8-e4m3 [N, K] with fp32 per-tensor weight_scale
- everything else: bf16
"""
import glob, os
import torch
from safetensors import safe_open

# Hugging Face snapshot if present, else the slimmed local copy (configs + non-expert shards; experts live in experts.bin)
# ORNITH_MODEL (set by ornith-serve's installer) wins over both
PATH = (([os.environ['ORNITH_MODEL']] if os.environ.get('ORNITH_MODEL') else [])
        + glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-NVFP4/snapshots/*'))
        + [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')])[0]
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
LM = 'model.language_model.'


class Weights:
    def __init__(self, path=PATH):
        self.path = path
        self.h = [safe_open(f, framework='pt', device='cpu') for f in sorted(glob.glob(os.path.join(path, '*.safetensors')))]
        self.where = {k: h for h in self.h for k in h.keys()}

    def __contains__(self, k):
        return k in self.where

    def raw(self, k):
        return self.where[k].get_tensor(k)

    def dense(self, prefix, device='cpu'):
        """fp32 [N, K] weight for any linear layer, whatever its storage format."""
        w = self.raw(prefix + '.weight')
        if w.dtype == torch.uint8:  # NVFP4
            N, K2 = w.shape
            lo, hi = (w & 15).long(), (w >> 4).long()
            lut = E2M1.to(w.device)
            vals = torch.stack([lut[lo], lut[hi]], -1).reshape(N, K2 * 2)
            s = self.raw(prefix + '.weight_scale').float().repeat_interleave(16, 1)
            return (vals * s * self.raw(prefix + '.weight_scale_2').float()).to(device)
        if w.dtype == torch.float8_e4m3fn:
            return (w.float() * self.raw(prefix + '.weight_scale').float()).to(device)
        return w.float().to(device)
