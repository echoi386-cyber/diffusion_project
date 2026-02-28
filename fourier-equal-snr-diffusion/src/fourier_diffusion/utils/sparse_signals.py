import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset


def rfft1(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.rfft(x, dim=-1, norm="ortho")


def irfft1(y: torch.Tensor, n: int) -> torch.Tensor:
    return torch.fft.irfft(y, n=n, dim=-1, norm="ortho")


@dataclass
class KSparseConfig:
    length: int = 256
    k: int = 8
    amp_min: float = 0.5
    amp_max: float = 1.5
    time_noise_std: float = 0.0
    seed: int = 0


class KSparseFourierDataset(Dataset):
    """
    Real x in R^N whose rFFT coefficients have exactly k nonzero bins.
    Returns:
      x: (N,) float32 in roughly [-1,1]
      support: (k,) int64 rFFT bin indices
    """
    def __init__(self, cfg: KSparseConfig, n_samples: int):
        super().__init__()
        self.cfg = cfg
        self.n_samples = n_samples
        self.N = cfg.length
        self.Nf = self.N // 2 + 1

        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(cfg.seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg

        bins = torch.randperm(self.Nf, generator=self.gen)[:cfg.k]  # (k,)
        amps = torch.rand((cfg.k,), generator=self.gen) * (cfg.amp_max - cfg.amp_min) + cfg.amp_min
        phases = torch.rand((cfg.k,), generator=self.gen) * (2.0 * math.pi)
        coeffs = amps * torch.exp(1j * phases)  # (k,) complex

        y = torch.zeros((self.Nf,), dtype=torch.complex64)
        y[bins] = coeffs

        x = irfft1(y.unsqueeze(0), n=self.N).squeeze(0).to(torch.float32)

        if cfg.time_noise_std > 0:
            x = x + cfg.time_noise_std * torch.randn_like(x, generator=self.gen)

        x = x / x.abs().max().clamp_min(1e-6)  # normalize

        return x, bins