import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SignalMLP(nn.Module):
    """
    Denoiser for 1D signals: predicts x0_hat given (x_t, t).
    """
    def __init__(self, length: int, hidden: int = 256, depth: int = 4, t_dim: int = 128):
        super().__init__()
        self.length = length
        self.t_emb = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        layers = []
        in_dim = length + t_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, length)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_emb(t)
        return self.net(torch.cat([x, te], dim=1))