import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyMLP(nn.Module):
    """
    Predicts x0 directly from (xt, t) using x0-parameterization.
    """

    def __init__(self, dim: int = 2, t_dim: int = 64, width: int = 256, depth: int = 4):
        super().__init__()
        self.t_dim = t_dim

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        layers = []
        in_dim = dim + t_dim
        for i in range(depth):
            layers += [nn.Linear(in_dim if i == 0 else width, width), nn.SiLU()]
        layers += [nn.Linear(width, dim)]
        self.net = nn.Sequential(*layers)

    def time_embed(self, t: torch.Tensor) -> torch.Tensor:
        half = self.t_dim // 2
        device = t.device
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.t_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.t_mlp(emb)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_embed(t)
        h = torch.cat([x, te], dim=1)
        return self.net(h)