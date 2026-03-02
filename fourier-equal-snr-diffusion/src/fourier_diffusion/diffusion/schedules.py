import math
from typing import Optional, Tuple

import torch

from fourier_diffusion.utils.seed import get_device


def make_beta_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device=None,
    schedule: str = "cosine",
    s: float = 0.008,
) -> torch.Tensor:
    device = device or get_device()
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T, device=device)
    if schedule != "cosine":
        raise ValueError(f"Unknown schedule: {schedule}")

    steps = torch.arange(T + 1, device=device, dtype=torch.float32)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = (f / f[0]).clamp(1e-8, 1.0)
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-8, 0.999)


def alphas_from_betas(betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bar


def snr_from_alpha_bar(
    alpha_bar: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    Sigma_diag: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # s_t(i) = alpha_bar_t * C_i / ((1-alpha_bar_t) * Sigma_ii)
    T = alpha_bar.shape[0]
    if C is None:
        C = torch.ones(1, device=alpha_bar.device)
    if Sigma_diag is None:
        Sigma_diag = torch.ones_like(C)
    ab = alpha_bar.view(T, *([1] * C.ndim))
    return (ab * C) / ((1.0 - ab) * Sigma_diag)


def calibrate_equal_snr_alpha_bar(alpha_bar_ddpm: torch.Tensor, C_diag: torch.Tensor) -> torch.Tensor:
    """
    Match average SNR across frequencies between DDPM and EqualSNR.
    Derived in your existing comment and consistent with paper calibration idea.
    """
    meanC = C_diag.mean()
    ab = alpha_bar_ddpm
    ab_eq = (ab * meanC) / ((1.0 - ab) + ab * meanC)
    return ab_eq.clamp(1e-8, 1.0 - 1e-8)