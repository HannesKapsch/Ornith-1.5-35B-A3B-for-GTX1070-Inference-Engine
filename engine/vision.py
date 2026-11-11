"""Vision encoder of Ornith (Qwen3.5-MoE ViT, 27 blocks, 1152 wide, patch 16, 2x2 merge -> 2048) on the CPU, fp32.
The GPU stays free for the language model; an 800x600 image (~475 tokens) takes a few seconds.

encode(path) -> (embeds [n, 2048] float32, (grid_h, grid_w))   with n = grid_h * grid_w (merged grid, row-major)
image_tokens(n) -> token ids  <|vision_start|> <|image_pad|> x n <|vision_end|>
"""
import os, time, torch
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel
from PIL import Image
from weights import Weights, PATH

VISION_START, VISION_END, IMAGE_PAD = 248053, 248054, 248056


class Vision:
    def __init__(self, W=None, max_pixels=1024 * 1024, min_pixels=128 * 128, threads=None):
        t0 = time.time()
        cfg = AutoConfig.from_pretrained(PATH).vision_config
        self.model = Qwen3_5MoeVisionModel._from_config(cfg, torch_dtype=torch.float32).eval()
        W = W or Weights()
        sd = {}
        for k in self.model.state_dict():
            sd[k] = W.raw('model.visual.' + k).float()
        self.model.load_state_dict(sd)
        self.proc = AutoImageProcessor.from_pretrained(PATH, size={'longest_edge': max_pixels, 'shortest_edge': min_pixels})
        self.merge = cfg.spatial_merge_size
        if threads: torch.set_num_threads(threads)
        self.load_seconds = time.time() - t0

    @torch.no_grad()
    def encode(self, path_or_image):
        img = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
        img = img.convert('RGB')
        out = self.proc(images=[img], return_tensors='pt')
        grid = out['image_grid_thw']
        emb = self.model(out['pixel_values'].float(), grid_thw=grid)
        emb = emb.pooler_output if hasattr(emb, 'pooler_output') else emb
        t, h, w = grid[0].tolist()
        gh, gw = h // self.merge, w // self.merge
        assert emb.shape[0] == t * gh * gw, (emb.shape, grid)
        return emb.float(), (gh, gw)


def image_tokens(n):
    return [VISION_START] + [IMAGE_PAD] * n + [VISION_END]
