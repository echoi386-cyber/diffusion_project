import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlock
from .time_embedding import SinusoidalTimeEmbedding

class SimpleUNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 64, t_dim: int = 128):
        super().__init__()
        self.t_emb = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.d1 = ResBlock(in_ch, base, t_dim)
        self.d2 = ResBlock(base, base*2, t_dim)
        self.d3 = ResBlock(base*2, base*4, t_dim)
        self.pool = nn.AvgPool2d(2)
        self.u2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.u3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.m1 = ResBlock(base*4, base*4, t_dim)
        self.u_block2 = ResBlock(base*4 + base*2, base*2, t_dim)
        self.u_block1 = ResBlock(base*2 + base, base, t_dim)
        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_emb(t)
        h1 = self.d1(x, te)
        h2 = self.d2(self.pool(h1), te)
        h3 = self.d3(self.pool(h2), te)
        h = self.m1(h3, te)
        h = self.u3(h)
        h = self.u_block2(torch.cat([h, h2], dim=1), te)
        h = self.u2(h)
        h = self.u_block1(torch.cat([h, h1], dim=1), te)
        return self.out(h)