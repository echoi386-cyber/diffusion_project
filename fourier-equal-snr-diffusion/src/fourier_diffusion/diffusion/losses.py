from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.fft import rfft2
from .forward_process import FourierForwardProcess


def loss_x0_fourier_weighted(
    model: nn.Module,
    fwd: FourierForwardProcess,
    x0: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Paper (Alg. 1): loss is || C^{-1/2}(y0 - y0_hat) ||^2 in Fourier domain for Fourier-space noise variants.
    Baseline DDPM: pixel-space MSE on x0 for stable comparison.
    """
    xt, aux = fwd.q_sample(x0, t)
    x0_hat = model(xt, t)

    if fwd.cfg.schedule == "ddpm" or fwd.C_diag is None:
        loss = F.mse_loss(x0_hat, x0)
        return loss, {"loss": loss.detach()}

    y0 = aux["y0"]
    y0_hat = rfft2(x0_hat)
    C = fwd.C_diag.unsqueeze(0)
    diff = (y0 - y0_hat) / torch.sqrt(C)
    loss = (diff.real ** 2 + diff.imag ** 2).mean()
    return loss, {"loss": loss.detach()}