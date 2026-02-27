import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from ..utils.fft import rfft2
from .forward_process import FourierForwardProcess

# TRaining loss for Fourier weighted data prediction
def loss_x0_fourier_weighted(
    model: nn.Module,
    fwd: FourierForwardProcess,
    x0: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Predict x0_hat = model(xt, t).
    If Fourier scheme active: compute y0_hat = F(x0_hat) and MSE in Fourier scaled by C^{-1/2}.
    For DDPM baseline: reduce to pixel MSE on x0 (for stable baseline comparison).
    """
    xt, aux = fwd.q_sample(x0, t)
    x0_hat = model(xt, t)

    if fwd.cfg.schedule == "ddpm" or fwd.C_diag is None:
        loss = F.mse_loss(x0_hat, x0)
        return loss, {"loss": loss.detach()}

    y0 = aux["y0"]
    y0_hat = rfft2(x0_hat)
    # scale by C^{-1/2}
    C = fwd.C_diag.unsqueeze(0)
    diff = (y0 - y0_hat) / torch.sqrt(C)
    loss = (diff.real ** 2 + diff.imag ** 2).mean()
    return loss, {"loss": loss.detach()}
