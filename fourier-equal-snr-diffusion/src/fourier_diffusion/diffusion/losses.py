from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.fft import rfft2, rfft2_onesided_weights
from .forward_process import FourierForwardProcess


def loss_x0_fourier_weighted(
    model: nn.Module,
    fwd: FourierForwardProcess,
    x0: torch.Tensor,
    t: torch.Tensor,
    c_floor_rel: float = 1e-3,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    xt, aux = fwd.q_sample(x0, t)
    x0_hat = model(xt, t)

    if fwd.cfg.schedule == "ddpm" or fwd.C_diag is None:
        loss = F.mse_loss(x0_hat.float(), x0.float())
        return loss, {"loss": loss.detach()}

    y0 = aux["y0"].to(torch.complex64)
    y0_hat = rfft2(x0_hat.float()).to(torch.complex64)

    B, _, H, W = x0.shape
    C_raw = fwd.C_diag.float().unsqueeze(0)

    mult = rfft2_onesided_weights(
        H,
        W,
        device=C_raw.device,
        dtype=C_raw.dtype,
    ).unsqueeze(0).unsqueeze(0)

    weighted_bins_per_sample = mult.sum() * C_raw.shape[1]
    C_mean = (C_raw * mult).sum() / weighted_bins_per_sample

    if c_floor_rel > 0:
        C_floor = (c_floor_rel * C_mean).detach()
    else:
        C_floor = torch.tensor(1e-8, device=C_raw.device, dtype=C_raw.dtype)

    C = C_raw.clamp_min(C_floor)

    diff = (y0 - y0_hat) / torch.sqrt(C)
    sq = diff.real.square() + diff.imag.square()

    loss = (sq * mult).sum() / (weighted_bins_per_sample * B)

    return loss, {
        "loss": loss.detach(),
        "C_floor": C_floor.detach(),
    }