import torch
import torch.nn as nn

from .toy_forward import ToyForward


@torch.no_grad()
def ddim_sample_toy(model: nn.Module, fwd: ToyForward, n: int, steps: int) -> torch.Tensor:
    """
    DDIM update in PCA space:
      y_{t-1} = sqrt(ab_{t-1}) y0_hat + sqrt(1-ab_{t-1})/sqrt(1-ab_t) * (y_t - sqrt(ab_t) y0_hat)
    """
    device = fwd.device
    B = n

    y = torch.randn(B, 2, device=device) * torch.sqrt(fwd.Sigma).view(1, 2)
    ts = torch.linspace(fwd.T, 1, steps, device=device).long()

    for i, t_scalar in enumerate(ts):
        t = torch.full((B,), int(t_scalar.item()), device=device, dtype=torch.long)
        ab_t = fwd.alpha_bar[t - 1].view(B, 1)

        x = y @ fwd.U.t()
        x0_hat = model(x, t)
        y0_hat = x0_hat @ fwd.U

        if i == len(ts) - 1:
            return x0_hat

        t_prev = int(ts[i + 1].item())
        ab_prev = fwd.alpha_bar[t_prev - 1].view(1, 1)

        y = torch.sqrt(ab_prev) * y0_hat + (torch.sqrt(1.0 - ab_prev) / torch.sqrt(1.0 - ab_t)) * (
            y - torch.sqrt(ab_t) * y0_hat
        )

    return y @ fwd.U.t()