import torch
from typing import Optional, Tuple

#Schedules
def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2, device=None, schedule: str = "cosine", s: float = 0.008) -> torch.Tensor:
    # Linear beta schedule
    device = device or get_device()
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T, device=device)
    if schedule != "cosine":
        raise ValueError(f"Unknown schedule: {schedule}")

    # cosine alpha_bar
    steps = torch.arrange(T+1, device=device, dtype=torch.float32)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    alpha_bar = alpha_bar.clamp(1e-8, 1.0)
    #betas from alpha_bar
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-8, 0.999)

ef alphas_from_betas(betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # betas shape (T,)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bar

def snr_from_alpha_bar(alpha_bar: torch.Tensor, C: Optional[torch.Tensor] = None, Sigma_diag: Optional[torch.Tensor] = None) -> torch.Tensor:
    # s_t(i) = alpha_bar_t * C_i / ((1-alpha_bar_t) * Sigma_ii)
    # If C is None, assume C_i = 1. If Sigma_diag is None assume Sigma_ii=1 (DDPM).
    T = alpha_bar.shape[0]
    if C is None:
        # scalar 1
        C = torch.ones(1, device=alpha_bar.device)
    if Sigma_diag is None:
        Sigma_diag = torch.ones_like(C)
    # broadcast: (T,1,...) then to match C
    ab = alpha_bar.view(T, *([1] * C.ndim))
    return (ab * C) / ((1.0 - ab) * Sigma_diag)

def calibrate_equal_snr_alpha_bar(alpha_bar_ddpm: torch.Tensor, C_diag: torch.Tensor) -> torch.Tensor:
    """
    choose alpha_bar_eq so that average SNR across frequencies matches DDPM at each t:
      alpha_eq/(1-alpha_eq) = alpha_ddpm/(1-alpha_ddpm) *  (1/d) sum_i C_i
    => alpha_eq = alpha_ddpm * meanC / ((1-alpha_ddpm) + alpha_ddpm*meanC)
    """
    meanC = C_diag.mean()
    ab = alpha_bar_ddpm
    ab_eq = (ab * meanC) / ((1.0 - ab) + ab * meanC)
    return ab_eq.clamp(1e-8, 1.0 - 1e-8)
