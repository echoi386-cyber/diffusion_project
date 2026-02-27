import math
import random
import torch
# Masking
def mask_pixels_bernoulli(x: torch.Tensor, p_zero: float = 0.5) -> torch.Tensor:
    # independent per-pixel masking
    m = (torch.rand_like(x) > p_zero).float()
    return x * m

def mask_random_block(x: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
    # zero out a random rectangular block covering roughly frac of area
    B, C, H, W = x.shape
    out = x.clone()
    block_area = int(H * W * frac)
    bh = max(1, int(math.sqrt(block_area)))
    bw = max(1, int(block_area / bh))
    for b in range(B):
        top = random.randint(0, max(0, H - bh))
        left = random.randint(0, max(0, W - bw))
        out[b, :, top:top+bh, left:left+bw] = 0.0
    return out

def mask_random_halfplane(x: torch.Tensor) -> torch.Tensor:
    # zero either left/right or top/bottom randomly
    B, C, H, W = x.shape
    out = x.clone()
    for b in range(B):
        if random.random() < 0.5:
            # vertical split
            if random.random() < 0.5:
                out[b, :, :, :W//2] = 0.0
            else:
                out[b, :, :, W//2:] = 0.0
        else:
            # horizontal split
            if random.random() < 0.5:
                out[b, :, :H//2, :] = 0.0
            else:
                out[b, :, H//2:, :] = 0.0
    return out