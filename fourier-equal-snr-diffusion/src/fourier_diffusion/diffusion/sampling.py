from typing import Tuple

import torch
import torch.nn as nn

from ..utils.fft import rfft2, irfft2
from .forward_process import FourierForwardProcess, sample_valid_rfft2_noise


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    fwd: FourierForwardProcess,
    shape: Tuple[int, int, int, int],  # (B,C,H,W)
    steps: int = 100,
) -> torch.Tensor:
    """
    Paper (Alg. 2): DDIM in Fourier space for EqualSNR/FlippedSNR-like variants.
    For DDPM baseline: pixel-space DDIM update.
    """
    device = fwd.device
    B, C, H, W = shape
    T = fwd.cfg.T
    ts = torch.linspace(T, 1, steps, device=device).long()

    if fwd.cfg.schedule == "ddpm" or fwd.C_diag is None:
        x = torch.randn(shape, device=device)
        for idx, t in enumerate(ts):
            t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
            ab_t = fwd.alpha_bar_t(t_batch).view(B, 1, 1, 1)
            x0_hat = model(x, t_batch)
            if idx == len(ts) - 1:
                return x0_hat
            t_prev = ts[idx + 1]
            t_prev_batch = torch.full((B,), int(t_prev.item()), device=device, dtype=torch.long)
            ab_prev = fwd.alpha_bar_t(t_prev_batch).view(B, 1, 1, 1)
            eps = (x - torch.sqrt(ab_t) * x0_hat) / torch.sqrt(1.0 - ab_t)
            x = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1.0 - ab_prev) * eps
        return x

    eps = sample_valid_rfft2_noise(B, C, H, W, device)
    y = eps * torch.sqrt(fwd.Sigma_diag).unsqueeze(0).to(eps.dtype)

    for idx, t in enumerate(ts):
        t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
        ab_t = fwd.alpha_bar_t(t_batch).view(B, 1, 1, 1)

        x = irfft2(y, H, W)
        x0_hat = model(x, t_batch)
        y0_hat = rfft2(x0_hat)

        if idx == len(ts) - 1:
            return x0_hat

        t_prev = ts[idx + 1]
        t_prev_batch = torch.full((B,), int(t_prev.item()), device=device, dtype=torch.long)
        ab_prev = fwd.alpha_bar_t(t_prev_batch).view(B, 1, 1, 1)

        y = torch.sqrt(ab_prev) * y0_hat + torch.sqrt(1.0 - ab_prev) / torch.sqrt(1.0 - ab_t) * (y - torch.sqrt(ab_t) * y0_hat)

    return irfft2(y, H, W)