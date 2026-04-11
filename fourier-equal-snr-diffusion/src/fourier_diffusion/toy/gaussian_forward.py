from dataclasses import dataclass

import torch

from fourier_diffusion.diffusion.schedules import (
    make_beta_schedule,
    alphas_from_betas,
    calibrate_equal_snr_alpha_bar,
)


@dataclass
class GaussianToyForward:
    schedule: str
    U: torch.Tensor
    C: torch.Tensor
    T: int
    alpha_bar: torch.Tensor
    Sigma: torch.Tensor
    device: torch.device

    @classmethod
    def build(
        cls,
        schedule: str,
        U: torch.Tensor,
        C: torch.Tensor,
        T: int,
        device: torch.device,
    ):
        betas = make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2, device=device, schedule="cosine")
        _, alpha_bar_ddpm = alphas_from_betas(betas)
        alpha_bar_ddpm = alpha_bar_ddpm.clamp(1e-8, 1.0 - 1e-8)

        if schedule == "ddpm":
            alpha_bar = alpha_bar_ddpm
            Sigma = torch.ones_like(C)
        elif schedule == "equal_snr":
            alpha_bar = calibrate_equal_snr_alpha_bar(alpha_bar_ddpm, C).clamp(1e-8, 1.0 - 1e-8)
            Sigma = C.clone()
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        return cls(
            schedule=schedule,
            U=U,
            C=C,
            T=T,
            alpha_bar=alpha_bar,
            Sigma=Sigma,
            device=device,
        )

    def sample_t(self, batch: int) -> torch.Tensor:
        return torch.randint(1, self.T + 1, (batch,), device=self.device)

    def alpha_bar_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar[t - 1]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        y0 = x0 @ self.U
        ab = self.alpha_bar_t(t).view(-1, 1)
        eps = torch.randn_like(y0) * torch.sqrt(self.Sigma).view(1, -1)
        yt = torch.sqrt(ab) * y0 + torch.sqrt(1.0 - ab) * eps
        xt = yt @ self.U.t()
        return xt, y0, yt