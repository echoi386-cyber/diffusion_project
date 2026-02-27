import math
import torch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal
from .schedules import make_beta_schedule, alphas_from_betas, calibrate_equal_snr_alpha_bar
from .covariance import make_sigma_diag
from ..utils.fft import rfft2, irfft2

#Forward process and sampling
@dataclass
class FourierDiffusionConfig:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: Literal["ddpm", "equal_snr", "flipped_snr"] = "ddpm"
    # If equal_snr or flipped_snr, use calibrated alpha_bar unless disabled:
    calibrate_alpha_bar: bool = True

class FourierForwardProcess:
    def __init__(self, cfg: FourierDiffusionConfig, C_diag: Optional[torch.Tensor], device: torch.device):
        self.cfg = cfg
        self.device = device
        betas = make_beta_schedule(cfg.T, cfg.beta_start, cfg.beta_end, device=device)
        _, alpha_bar_ddpm = alphas_from_betas(betas)

        if cfg.schedule == "ddpm" or C_diag is None:
            self.alpha_bar = alpha_bar_ddpm
            self.C_diag = None
            self.Sigma_diag = None
        else:
            self.C_diag = C_diag.to(device)
            self.Sigma_diag = make_sigma_diag(cfg.schedule, self.C_diag)
            if cfg.calibrate_alpha_bar and cfg.schedule == "equal_snr":
                self.alpha_bar = calibrate_equal_snr_alpha_bar(alpha_bar_ddpm, self.C_diag)  # App. A.3 fileciteturn1file0L14-L78
            else:
                self.alpha_bar = alpha_bar_ddpm

        self.alpha_bar = self.alpha_bar.clamp(1e-8, 1.0 - 1e-8)

    def sample_t(self, batch: int) -> torch.Tensor:
        # integer timesteps in [1..T]
        return torch.randint(1, self.cfg.T + 1, (batch,), device=self.device)

    def alpha_bar_t(self, t: torch.Tensor) -> torch.Tensor:
        # t in [1..T], returns alpha_bar[t-1]
        return self.alpha_bar[t - 1]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Return xt (pixel space) and aux dict with y0, yt, eps_y used.
        Implement Fourier-space forward:
          y_t = sqrt(ab)*y0 + sqrt(1-ab)*eps_Sigma
        where eps_Sigma ~ CN(0, Sigma) via eps * sqrt(Sigma_diag).
        """
        B, C, H, W = x0.shape
        ab = self.alpha_bar_t(t).view(B, 1, 1, 1)

        if self.cfg.schedule == "ddpm" or self.C_diag is None:
            if noise is None:
                noise = torch.randn_like(x0)
            xt = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
            return xt, {"x0": x0, "noise": noise, "alpha_bar": ab.squeeze()}

        # Fourier path
        y0 = rfft2(x0)  # (B,C,H,Wf)
        Wf = y0.shape[-1]

        if noise is None:
            # complex standard normal in rfft domain: N(0,1) for real and imaginary
            eps = (torch.randn((B, C, H, Wf), device=self.device) +
                   1j * torch.randn((B, C, H, Wf), device=self.device)) / math.sqrt(2.0)
        else:
            eps = noise  # user supplied complex eps in rfft shape

        sigma_sqrt = torch.sqrt(self.Sigma_diag).unsqueeze(0)
        eps_sigma = eps * sigma_sqrt

        ab_y = self.alpha_bar_t(t).view(B, 1, 1, 1)
        yt = torch.sqrt(ab_y) * y0 + torch.sqrt(1.0 - ab_y) * eps_sigma
        xt = irfft2(yt, H, W)
        return xt, {"y0": y0, "yt": yt, "eps_sigma": eps_sigma, "alpha_bar": ab_y.squeeze()}
