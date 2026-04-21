import torch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

from .schedules import make_beta_schedule, alphas_from_betas, calibrate_equal_snr_alpha_bar
from .covariance import make_sigma_diag
from ..utils.fft import rfft2, irfft2


def sample_valid_rfft2_noise(
    batch: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    x = torch.randn((batch, channels, height, width), device=device, dtype=torch.float32)
    return rfft2(x)


@dataclass
class FourierDiffusionConfig:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: Literal["ddpm", "equal_snr", "flipped_snr", "power_law"] = "ddpm"
    lam: float = 0.0
    calibration: Literal["fixed_trace"] = "fixed_trace"
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

            if cfg.schedule == "equal_snr":
                self.Sigma_diag = make_sigma_diag("equal_snr", self.C_diag)
                if cfg.calibrate_alpha_bar:
                    self.alpha_bar = calibrate_equal_snr_alpha_bar(alpha_bar_ddpm, self.C_diag)
                else:
                    self.alpha_bar = alpha_bar_ddpm

            elif cfg.schedule == "flipped_snr":
                self.Sigma_diag = make_sigma_diag("flipped_snr", self.C_diag)
                self.alpha_bar = alpha_bar_ddpm

            elif cfg.schedule == "power_law":
                self.Sigma_diag = make_sigma_diag(
                    "power_law",
                    self.C_diag,
                    lam=cfg.lam,
                    calibration=cfg.calibration,
                )
                self.alpha_bar = alpha_bar_ddpm

            else:
                raise ValueError(f"Unknown schedule: {cfg.schedule}")

    def sample_t(self, batch: int) -> torch.Tensor:
        return torch.randint(1, self.cfg.T + 1, (batch,), device=self.device)

    def alpha_bar_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar[t - 1]

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, C, H, W = x0.shape
        ab = self.alpha_bar_t(t).view(B, 1, 1, 1)

        if self.cfg.schedule == "ddpm" or self.C_diag is None:
            if noise is None:
                noise = torch.randn_like(x0)
            xt = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
            return xt, {"x0": x0, "noise": noise, "alpha_bar": ab.squeeze()}

        y0 = rfft2(x0)
        Wf = y0.shape[-1]

        if noise is None:
            eps = sample_valid_rfft2_noise(B, C, H, W, self.device).to(y0.dtype)
        else:
            eps = noise.to(y0.dtype)

        sigma_sqrt = torch.sqrt(self.Sigma_diag).unsqueeze(0).to(y0.dtype)
        eps_sigma = eps * sigma_sqrt

        yt = torch.sqrt(ab) * y0 + torch.sqrt(1.0 - ab) * eps_sigma
        xt = irfft2(yt, H, W)

        return xt, {
            "y0": y0,
            "yt": yt,
            "eps_sigma": eps_sigma,
            "alpha_bar": ab.squeeze(),
        }