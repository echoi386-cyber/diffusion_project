import torch
from .schedules import alpha_bar_cosine

class ToyForwardProcess:
    def __init__(self, U: torch.Tensor, C: torch.Tensor, T: int, schedule: str, device):
        """
        U: (D,D) PCA eigenvectors
        C: (D,) PCA eigenvalues
        schedule: "ddpm" or "equal_snr" or "flipped_snr"
        """
        self.U = U
        self.C = C.clamp_min(1e-12)
        self.T = T
        self.device = device
        self.alpha_bar = alpha_bar_cosine(T, device=device)

        if schedule == "ddpm":
            self.Sigma = torch.ones_like(self.C)
        elif schedule == "equal_snr":
            # EqualSNR variance-preserving: Sigma_ii = C_i
            self.Sigma = self.C.clone()
        elif schedule == "flipped_snr":
            # Minimal, correct "flip" in coordinate index
            self.Sigma = torch.flip(self.C, dims=[0]).clamp_min(1e-8)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        """
        x0: (B,D) in data space
        t:  (B,) integer in [0,T-1]
        Returns xt (B,D) and y0 (B,D) (PCA coords)
        """
        B, D = x0.shape
        y0 = x0 @ self.U  # PCA coords

        ab = self.alpha_bar[t].view(B, 1)  # (B,1)
        # eps in PCA coords with diag Sigma
        eps = torch.randn(B, D, device=self.device) * torch.sqrt(self.Sigma).view(1, D)

        yt = torch.sqrt(ab) * y0 + torch.sqrt(1.0 - ab) * eps
        xt = yt @ self.U.T
        return xt, y0