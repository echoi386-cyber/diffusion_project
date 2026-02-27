#!/usr/bin/env python3
import argparse
import os
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fourier_diffusion.utils.seed import get_device, seed_all
from fourier_diffusion.utils.plots import plot_loss_curves, plot_radial_spectra
from fourier_diffusion.utils.fft import radial_power_spectrum
from fourier_diffusion.models.toy_mlp import ToyMLP


def make_mog(n: int, k: int, radius: float, std: float, device: torch.device):
    angles = torch.linspace(0, 2 * math.pi, k + 1, device=device)[:-1]
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    idx = torch.randint(0, k, (n,), device=device)
    return centers[idx] + std * torch.randn(n, 2, device=device)


@torch.no_grad()
def hist2d_density_torch(samples: torch.Tensor, bins: int = 128, lim: float = 8.0) -> torch.Tensor:
    # samples: (N,2) on device
    device = samples.device
    x = samples[:, 0].clamp(-lim, lim)
    y = samples[:, 1].clamp(-lim, lim)

    xi = ((x + lim) * (bins / (2.0 * lim))).long().clamp(0, bins - 1)
    yi = ((y + lim) * (bins / (2.0 * lim))).long().clamp(0, bins - 1)

    H = torch.zeros((bins, bins), device=device, dtype=torch.float32)
    H.index_put_((xi, yi), torch.ones_like(x, dtype=torch.float32), accumulate=True)

    # density normalization (optional but stable)
    bin_w = (2.0 * lim) / bins
    area = bin_w * bin_w
    H = H / (H.sum().clamp_min(1.0) * area)

    return H.view(1, 1, bins, bins)


class ToyForwardProcess:
    # PCA-basis toy analogue: low_to_high(ddpm), equal_snr, high_to_low(flipped_snr)
    def __init__(self, schedule: str, X: torch.Tensor, T: int, device: torch.device):
        self.schedule = schedule
        self.T = T
        betas = torch.linspace(1e-4, 2e-2, T, device=device)
        alphas = 1 - betas
        self.alpha_bar = torch.cumprod(alphas, 0).clamp(1e-8, 1 - 1e-8)

        Xc = X - X.mean(0, keepdim=True)
        cov = (Xc.t() @ Xc) / Xc.shape[0]
        eigvals, eigvecs = torch.linalg.eigh(cov)
        self.U = eigvecs
        self.C = eigvals.clamp_min(1e-8)

        if schedule == "ddpm":
            self.Sigma = torch.ones_like(self.C)
        elif schedule == "equal_snr":
            self.Sigma = self.C.clone()
        elif schedule == "flipped_snr":
            self.Sigma = (self.C / torch.flip(self.C, dims=[0]).clamp_min(1e-8)).clamp_min(1e-8)
        else:
            raise ValueError(schedule)

    def sample_t(self, B: int, device: torch.device):
        return torch.randint(1, self.T + 1, (B,), device=device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        B = x0.shape[0]
        ab = self.alpha_bar[t - 1].view(B, 1)
        y0 = x0 @ self.U
        eps = torch.randn_like(y0)
        eps_sigma = eps * torch.sqrt(self.Sigma).view(1, 2)
        yt = torch.sqrt(ab) * y0 + torch.sqrt(1 - ab) * eps_sigma
        xt = yt @ self.U.t()
        return xt, y0


def train_toy(schedule: str, loader: DataLoader, X_all: torch.Tensor, device: torch.device, T: int, iters: int, lr: float, log_every: int):
    model = ToyMLP(dim=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    fwd = ToyForwardProcess(schedule, X_all, T=T, device=device)
    losses = []
    it = iter(loader)

    for step in range(1, iters + 1):
        try:
            (x0_cpu,) = next(it)
        except StopIteration:
            it = iter(loader)
            (x0_cpu,) = next(it)

        x0 = x0_cpu.to(device).float()
        t = fwd.sample_t(x0.shape[0], device)

        xt, y0 = fwd.q_sample(x0, t)
        x0_hat = model(xt, t)
        y0_hat = x0_hat @ fwd.U

        diff = (y0 - y0_hat) / torch.sqrt(fwd.C).view(1, 2)
        loss = (diff ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % log_every == 0:
            losses.append(float(loss.detach().cpu()))
            print(f"[toy {schedule}] step {step}/{iters} loss {loss.item():.4f}")

    return model, fwd, losses


@torch.no_grad()
def sample_toy(model: nn.Module, fwd: ToyForwardProcess, device: torch.device, n: int = 50000, steps: int = 200):
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

        t_prev = ts[i + 1]
        ab_prev = fwd.alpha_bar[t_prev - 1].view(1, 1)
        eps = (y - torch.sqrt(ab_t) * y0_hat) / torch.sqrt(1 - ab_t)
        y = torch.sqrt(ab_prev) * y0_hat + torch.sqrt(1 - ab_prev) * eps

    return y @ fwd.U.t()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="outputs/plots/toy")
    p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()
    seed_all(args.seed)

    X = make_mog(n=200000, k=8, radius=4.0, std=0.5, device=device)
    loader = DataLoader(TensorDataset(X.detach().cpu()), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    schedules = ["ddpm", "equal_snr", "flipped_snr"]
    models = {}
    fwds = {}
    loss_hist = {}

    for s in schedules:
        m, fwd, losses = train_toy(s, loader, X, device, args.T, args.iters, args.lr, log_every=200)
        models[s], fwds[s], loss_hist[s] = m, fwd, losses

    plot_loss_curves(loss_hist, "Toy MoG: loss comparison (PCA-weighted)", os.path.join(args.outdir, "toy_loss.png"))

    # Spectrum comparison via FFT of histogram density
    specs = {}
    real_H = hist2d_density_torch(make_mog(n=50000, k=8, radius=4.0, std=0.5, device=device), bins=128, lim=8.0)
    specs["real"] = radial_power_spectrum(real_H)

    for s in schedules:
        samp = sample_toy(models[s], fwds[s], device=device, n=50000, steps=200)
        H = hist2d_density_torch(samp, bins=128, lim=8.0)
        specs[f"gen_{s}"] = radial_power_spectrum(H)

    plot_radial_spectra(specs, "Toy MoG: radial spectrum of density histogram", os.path.join(args.outdir, "toy_spectrum.png"))
    print("saved:", args.outdir)


if __name__ == "__main__":
    main()