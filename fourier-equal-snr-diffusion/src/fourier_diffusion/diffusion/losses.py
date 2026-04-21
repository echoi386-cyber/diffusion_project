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
    c_floor_rel: float = 1e-3,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    xt, aux = fwd.q_sample(x0, t)
    x0_hat = model(xt, t)

    # True DDPM baseline
    if fwd.cfg.schedule == "ddpm" or fwd.C_diag is None:
        loss = F.mse_loss(x0_hat.float(), x0.float())
        return loss, {"loss": loss.detach()}

    # Safer Fourier-domain loss
    y0 = aux["y0"].to(torch.complex64)
    y0_hat = rfft2(x0_hat.float()).to(torch.complex64)

    C = fwd.C_diag.float().unsqueeze(0)
    C_floor = (c_floor_rel * C.mean()).detach()
    C = C.clamp_min(C_floor)

    diff = (y0 - y0_hat) / torch.sqrt(C)
    loss = (diff.real.square() + diff.imag.square()).mean()

    return loss, {
        "loss": loss.detach(),
        "C_floor": C_floor.detach(),
    }
