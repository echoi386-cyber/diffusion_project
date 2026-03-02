import torch
from dataclasses import dataclass

from fourier_diffusion.diffusion.schedules import (
    make_beta_schedule,
    alphas_from_betas,
    calibrate_equal_snr_alpha_bar,
)


@dataclass
class ToyForward:
    """
    PCA-basis analogue of the paper's Fourier-space forward process.

    In PCA domain y = x U:
      y_t = sqrt(alpha_bar_t) y_0 + sqrt(1-alpha_bar_t) eps_sigma
      eps_sigma ~ N(0, Sigma), Sigma diagonal
    """

    schedule: str
    T: int
    device: torch.device
    U: torch.Tensor          # (2,2)
    C: torch.Tensor          # (2,) eigenvalues (signal variances)
    Sigma: torch.Tensor      # (2,) diagonal noise variances
    alpha_bar: torch.Tensor  # (T,)

    @staticmethod
    def from_data(schedule: str, X: torch.Tensor, T: int, device: torch.device) -> "ToyForward":
        # PCA
        Xc = X - X.mean(0, keepdim=True)
        cov = (Xc.t() @ Xc) / Xc.shape[0]
        eigvals, eigvecs = torch.linalg.eigh(cov)
        U = eigvecs
        C = eigvals.clamp_min(1e-8)

        # Cosine DDPM schedule (repo-consistent)
        betas = make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2, device=device, schedule="cosine")
        _, alpha_bar_ddpm = alphas_from_betas(betas)
        alpha_bar_ddpm = alpha_bar_ddpm.clamp(1e-8, 1.0 - 1e-8)

        if schedule == "ddpm":
            Sigma = torch.ones_like(C)
            alpha_bar = alpha_bar_ddpm
        elif schedule == "equal_snr":
            Sigma = C.clone()
            alpha_bar = calibrate_equal_snr_alpha_bar(alpha_bar_ddpm, C).clamp(1e-8, 1.0 - 1e-8)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        return ToyForward(
            schedule=schedule,
            T=T,
            device=device,
            U=U,
            C=C,
            Sigma=Sigma,
            alpha_bar=alpha_bar,
        )

    def sample_t(self, B: int) -> torch.Tensor:
        return torch.randint(1, self.T + 1, (B,), device=self.device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Returns:
          xt : (B,2) in x-space
          y0 : (B,2) in PCA-space
          yt : (B,2) in PCA-space
        """
        B = x0.shape[0]
        ab = self.alpha_bar[t - 1].view(B, 1)
        y0 = x0 @ self.U
        eps = torch.randn_like(y0)
        eps_sigma = eps * torch.sqrt(self.Sigma).view(1, 2)
        yt = torch.sqrt(ab) * y0 + torch.sqrt(1.0 - ab) * eps_sigma
        xt = yt @ self.U.t()
        return xt, y0, yt