import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, T: int, emb_dim: int):
        super().__init__()
        self.T = T
        self.emb = nn.Embedding(T, emb_dim)

    def forward(self, t: torch.Tensor):
        return self.emb(t)

class ToyMLP(nn.Module):
    def __init__(self, dim: int, T: int, hidden: int = 256, t_emb: int = 64):
        super().__init__()
        self.tproj = nn.Sequential(
            TimeEmbedding(T, t_emb),
            nn.Linear(t_emb, hidden),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, xt: torch.Tensor, t: torch.Tensor):
        te = self.tproj(t)  # (B, hidden)
        h = torch.cat([xt, te], dim=1)
        return self.net(h)