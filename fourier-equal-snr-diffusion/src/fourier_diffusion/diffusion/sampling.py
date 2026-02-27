import math
import torch
from ..utils.fft import rfft2, irfft2
from .forward_process import FourierForwardProcess

# DDIM in Fourier space for EqualSNR/FlippedSNR
@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    fwd: FourierForwardProcess,
    shape: Tuple[int, int, int, int],  # (B,C,H,W)
    steps: int = 100,
) -> torch.Tensor:
    """
    Deterministic DDIM in Fourier space for EqualSNR/FlippedSNR,
    and in pixel space for DDPM.
    """
    device = fwd.device
    B, C, H, W = shape
    T = fwd.cfg.T
    # choose a subset of timesteps
    ts = torch.linspace(T, 1, steps, device=device).long()

    if fwd.cfg.schedule == "ddpm" or fwd.C_diag is None:
        x = torch.randn(shape, device=device)
        for idx, t in enumerate(ts):
            t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
            ab_t = fwd.alpha_bar_t(t_batch).view(B, 1, 1, 1)
            x0_hat = model(x, t_batch)
            if idx == len(ts) - 1:
                x = x0_hat
                break
            t_prev = ts[idx + 1]
            t_prev_batch = torch.full((B,), int(t_prev.item()), device=device, dtype=torch.long)
            ab_prev = fwd.alpha_bar_t(t_prev_batch).view(B, 1, 1, 1)
            # DDIM update in pixel space:
            eps = (x - torch.sqrt(ab_t) * x0_hat) / torch.sqrt(1.0 - ab_t)
            x = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1.0 - ab_prev) * eps
        return x

    # Fourier DDIM
    # init y_T = eps * sqrt(Sigma_diag)
    Wf = W // 2 + 1
    eps = (torch.randn((B, C, H, Wf), device=device) + 1j * torch.randn((B, C, H, Wf), device=device)) / math.sqrt(2.0)
    y = eps * torch.sqrt(fwd.Sigma_diag).unsqueeze(0)

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

        # DDIM update in Fourier
        # y_{t-1} = sqrt(ab_prev) y0_hat + sqrt(1-ab_prev) / sqrt(1-ab_t) * ( y_t - sqrt(ab_t) y0_hat )
        y = torch.sqrt(ab_prev) * y0_hat + torch.sqrt(1.0 - ab_prev) / torch.sqrt(1.0 - ab_t) * (y - torch.sqrt(ab_t) * y0_hat)

    return irfft2(y, H, W)