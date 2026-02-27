import torch
import torch.nn as nn
from .time_embedding import SinusoidalTimeEmbedding

class ToyMLP(nn.Module):
    def __init__(self, dim: int = 2, hidden: int = 256, t_dim: int = 128):
        super().__init__()
        self.t_emb = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_emb(t)
        return self.net(torch.cat([x, te], dim=1))
