import torch
from dataclasses import dataclass

from fourier_diffusion.diffusion.schedules import (
    make_beta_schedule,
    alphas_from_betas,
    calibrate_equal_snr_alpha_bar,
)


@dataclass
class ToyForward:
    schedule: str
    T: int
    device: torch.device
    U: torch.Tensor          # (d,d)
    C: torch.Tensor          # (d,)
    Sigma: torch.Tensor      # (d,)
    alpha_bar: torch.Tensor  # (T,)

    @staticmethod
    def from_data(
        schedule: str,
        X: torch.Tensor,
        T: int,
        device: torch.device,
        c_floor_rel: float = 1e-3,
    ) -> "ToyForward":
        """
        c_floor_rel: floor C_i to (c_floor_rel * mean(C)) to avoid singular/near-singular whitening.
        This is a numerical stabilization, not a change of the paper's conceptual structure.
        """
        Xc = X - X.mean(0, keepdim=True)
        cov = (Xc.t() @ Xc) / Xc.shape[0]
        eigvals, eigvecs = torch.linalg.eigh(cov)

        # Sort ascending so small-variance components come first
        idx = torch.argsort(eigvals)
        C_raw = eigvals[idx]
        U = eigvecs[:, idx]

        meanC = C_raw.mean().clamp_min(1e-12)
        floor = (c_floor_rel * meanC).clamp_min(1e-12)
        C = C_raw.clamp_min(floor)

        # DDPM cosine schedule
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
        Returns: xt (B,d), y0 (B,d), yt (B,d)
        """
        B = x0.shape[0]
        ab = self.alpha_bar[t - 1].view(B, 1)
        y0 = x0 @ self.U
        eps = torch.randn_like(y0)
        eps_sigma = eps * torch.sqrt(self.Sigma).view(1, -1)
        yt = torch.sqrt(ab) * y0 + torch.sqrt(1.0 - ab) * eps_sigma
        xt = yt @ self.U.t()
        return xt, y0, yt