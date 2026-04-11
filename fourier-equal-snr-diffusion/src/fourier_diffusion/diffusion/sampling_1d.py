import torch
from .forward_process_1d import rfft1, irfft1, Fourier1DLambdaForwardProcess


@torch.no_grad()
def reconstruct_from_xt_T(
    model,
    fwd: Fourier1DLambdaForwardProcess,
    xt_T: torch.Tensor,
    steps: int = 100,
) -> torch.Tensor:
    """
    Start from a provided x_T (from fwd.q_sample at t=T),
    run deterministic DDIM update in Fourier domain.
    """
    device = fwd.device
    B, N = xt_T.shape
    T = fwd.cfg.T
    ts = torch.linspace(T, 1, steps, device=device).long()

    y = rfft1(xt_T)  # initialize y_T from xt_T

    for i, t_scalar in enumerate(ts):
        t_int = int(t_scalar.item())
        t = torch.full((B,), t_int, device=device, dtype=torch.long)

        ab_t = fwd.alpha_bar[t_int - 1].view(1, 1).expand(B, 1)

        x = irfft1(y, n=N)
        x0_hat = model(x, t)
        y0_hat = rfft1(x0_hat)

        if i == len(ts) - 1:
            return x0_hat

        t_prev_int = int(ts[i + 1].item())
        ab_prev = fwd.alpha_bar[t_prev_int - 1].view(1, 1).expand(B, 1)

        y = torch.sqrt(ab_prev) * y0_hat + torch.sqrt(1.0 - ab_prev) / torch.sqrt(1.0 - ab_t) * (y - torch.sqrt(ab_t) * y0_hat)

    return irfft1(y, n=N)