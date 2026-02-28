import math
from dataclasses import dataclass
from typing import Literal, Dict, Tuple

import torch

from .schedules import make_beta_schedule, alphas_from_betas


def rfft1(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.rfft(x, dim=-1, norm="ortho")


def irfft1(y: torch.Tensor, n: int) -> torch.Tensor:
    return torch.fft.irfft(y, n=n, dim=-1, norm="ortho")


@torch.no_grad()
def estimate_C_diag_rfft1(loader, device: torch.device, n_batches: int = 200, clamp_min: float = 1e-8) -> torch.Tensor:
    """
    Estimate C_k = Var(y0_k) where y0 = rfft(x0).
    Returns: (Nf,) float32 on device.
    """
    sum_re = None
    sum_im = None
    sum_re2 = None
    sum_im2 = None
    n = 0

    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
        x0 = x0.to(device, non_blocking=True).float()  # (B,N)
        y0 = rfft1(x0)  # (B,Nf)
        re, im = y0.real, y0.imag

        if sum_re is None:
            sum_re = re.sum(dim=0)
            sum_im = im.sum(dim=0)
            sum_re2 = (re ** 2).sum(dim=0)
            sum_im2 = (im ** 2).sum(dim=0)
        else:
            sum_re += re.sum(dim=0)
            sum_im += im.sum(dim=0)
            sum_re2 += (re ** 2).sum(dim=0)
            sum_im2 += (im ** 2).sum(dim=0)
        n += re.shape[0]

    mean_re = sum_re / max(n, 1)
    mean_im = sum_im / max(n, 1)
    var_re = (sum_re2 / max(n, 1)) - mean_re ** 2
    var_im = (sum_im2 / max(n, 1)) - mean_im ** 2
    C = (var_re + var_im).clamp_min(clamp_min)
    return C


@dataclass
class Fourier1DLambdaConfig:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    beta_schedule: Literal["linear", "cosine"] = "cosine"
    cosine_s: float = 0.008
    lam: float = 1.0  # Sigma = C^lam


class Fourier1DLambdaForwardProcess:
    """
    Forward diffusion in rFFT domain:
      y_t = sqrt(ab_t) y0 + sqrt(1-ab_t) eps * sqrt(Sigma)
    where Sigma_k = C_k^lam.
    """

    def __init__(self, cfg: Fourier1DLambdaConfig, C_diag: torch.Tensor, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.C_diag = C_diag.to(device)

        betas = make_beta_schedule(
            T=cfg.T,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            device=device,
            schedule=cfg.beta_schedule,
            s=cfg.cosine_s,
        )
        _, alpha_bar = alphas_from_betas(betas)
        self.alpha_bar = alpha_bar.clamp(1e-8, 1.0 - 1e-8)

        self.Sigma_diag = torch.pow(self.C_diag.clamp_min(1e-12), float(cfg.lam)).clamp_min(1e-12)

    def sample_t(self, B: int) -> torch.Tensor:
        return torch.randint(1, self.cfg.T + 1, (B,), device=self.device)

    def alpha_bar_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar[t - 1]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x0: (B,N)
        returns xt: (B,N), and aux dict with y0, yt, ab.
        """
        B, N = x0.shape
        ab = self.alpha_bar_t(t).view(B, 1)

        y0 = rfft1(x0)  # (B,Nf)

        eps = (torch.randn_like(y0.real) + 1j * torch.randn_like(y0.real)) / math.sqrt(2.0)
        eps_sigma = eps * torch.sqrt(self.Sigma_diag).view(1, -1)

        yt = torch.sqrt(ab) * y0 + torch.sqrt(1.0 - ab) * eps_sigma
        xt = irfft1(yt, n=N)

        return xt, {"y0": y0, "yt": yt, "ab": ab}