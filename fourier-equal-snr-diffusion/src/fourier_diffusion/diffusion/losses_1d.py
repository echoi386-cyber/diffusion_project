from typing import Dict, Tuple
import torch

from .forward_process_1d import rfft1, Fourier1DLambdaForwardProcess


def loss_x0_fourier_weighted_1d(
    model,
    fwd: Fourier1DLambdaForwardProcess,
    x0: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    xt, aux = fwd.q_sample(x0, t)
    x0_hat = model(xt, t)

    y0 = aux["y0"]
    y0_hat = rfft1(x0_hat)

    C = fwd.C_diag.view(1, -1)
    diff = (y0 - y0_hat) / torch.sqrt(C)
    loss = (diff.real ** 2 + diff.imag ** 2).mean()
    return loss, {"loss": loss.detach()}