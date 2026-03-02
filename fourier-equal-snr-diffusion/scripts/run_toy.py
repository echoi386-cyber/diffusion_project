#!/usr/bin/env python3
# scripts/run_toy.py
#
# Paper-consistent toy experiment + EMA sampling:
# - DDPM baseline: pixel-space MSE on x0
# - EqualSNR: Sigma=C in PCA domain + alpha_bar calibrated to match average SNR to DDPM
# - DDPM uses cosine schedule (repo-consistent)
# - Sampling: DDIM update in PCA domain (toy analogue of Fourier DDIM)
# - Quality: Wasserstein metrics
#   * Sliced Wasserstein-1 (fast approximation)
#   * Sinkhorn W2 (entropic OT approximation)
# - EMA: maintain exponential moving average of model parameters and sample from EMA.
#
# Default settings are chosen to run under ~10 minutes on a GPU.

import argparse
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from fourier_diffusion.utils.seed import get_device, seed_all
from fourier_diffusion.diffusion.schedules import (
    make_beta_schedule,
    alphas_from_betas,
    calibrate_equal_snr_alpha_bar,
)


# -----------------------
# Data: Mixture of Gaussians on a ring (ground truth)
# -----------------------
@torch.no_grad()
def make_mog(n: int, k: int, radius: float, std: float, device: torch.device) -> torch.Tensor:
    angles = torch.linspace(0, 2 * math.pi, k + 1, device=device)[:-1]
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)  # (k,2)
    idx = torch.randint(0, k, (n,), device=device)
    return centers[idx] + std * torch.randn(n, 2, device=device)


# -----------------------
# Toy denoiser: predict x0 from xt, t (x0-parameterization)
# -----------------------
class ToyMLP(nn.Module):
    def __init__(self, dim: int = 2, t_dim: int = 64, width: int = 256, depth: int = 4):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        layers = []
        in_dim = dim + t_dim
        for i in range(depth):
            layers += [nn.Linear(in_dim if i == 0 else width, width), nn.SiLU()]
        layers += [nn.Linear(width, dim)]
        self.net = nn.Sequential(*layers)

    def time_embed(self, t: torch.Tensor) -> torch.Tensor:
        # Sinusoidal embedding for integer t in [1..T]
        half = self.t_dim // 2
        device = t.device
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.t_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.t_mlp(emb)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_embed(t)
        h = torch.cat([x, te], dim=1)
        return self.net(h)


# -----------------------
# EMA helper
# -----------------------
class EMA:
    """
    Exponential Moving Average of parameters.

    Usage:
      ema = EMA(model, decay=0.999, warmup_steps=200, device=device)
      ...
      ema.update(model, step=global_step)
      ...
      ema_model = ema.copy_to_model(model_class_ctor)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup_steps: int = 0,
        device: Optional[torch.device] = None,
    ):
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.device = device

        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                v = p.detach().clone()
                if self.device is not None:
                    v = v.to(self.device)
                self.shadow[name] = v

        self.buffers = {}
        for name, b in model.named_buffers():
            # For buffers (e.g., running stats), keep a copy so EMA model behaves consistently.
            v = b.detach().clone()
            if self.device is not None:
                v = v.to(self.device)
            self.buffers[name] = v

    @torch.no_grad()
    def update(self, model: nn.Module, step: int):
        # Optional warmup: use smaller decay early to let EMA catch up.
        if self.warmup_steps > 0 and step < self.warmup_steps:
            # linearly ramp decay from 0 to target decay
            d = self.decay * (step / max(1, self.warmup_steps))
        else:
            d = self.decay

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            x = p.detach()
            if self.device is not None:
                x = x.to(self.device)
            self.shadow[name].mul_(d).add_(x, alpha=(1.0 - d))

        # Keep buffers synced (copy, not EMA) for determinism.
        for name, b in model.named_buffers():
            x = b.detach()
            if self.device is not None:
                x = x.to(self.device)
            self.buffers[name].copy_(x)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(self.shadow[name].to(p.device))
        for name, b in model.named_buffers():
            b.copy_(self.buffers[name].to(b.device))


# -----------------------
# Forward process in PCA domain (toy analogue of Fourier domain)
# -----------------------
class ToyForwardProcess:
    """
    x0 in R^2
    PCA basis U, eigenvalues C (signal variance per "frequency").
    Forward in PCA domain y = x U:
      y_t = sqrt(alpha_bar_t) * y_0 + sqrt(1-alpha_bar_t) * eps_sigma
      eps_sigma ~ N(0, Sigma)
    Schedules:
      - ddpm: Sigma=I, alpha_bar=cosine schedule
      - equal_snr: Sigma=C, alpha_bar calibrated to match average SNR to DDPM
    """

    def __init__(self, schedule: str, X: torch.Tensor, T: int, device: torch.device):
        self.schedule = schedule
        self.T = T
        self.device = device

        # PCA from dataset
        Xc = X - X.mean(0, keepdim=True)
        cov = (Xc.t() @ Xc) / Xc.shape[0]  # (2,2)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        self.U = eigvecs  # (2,2)
        self.C = eigvals.clamp_min(1e-8)  # (2,)

        # DDPM cosine schedule (repo-consistent)
        betas = make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2, device=device, schedule="cosine")
        _, alpha_bar_ddpm = alphas_from_betas(betas)
        alpha_bar_ddpm = alpha_bar_ddpm.clamp(1e-8, 1.0 - 1e-8)

        if schedule == "ddpm":
            self.Sigma = torch.ones_like(self.C)
            self.alpha_bar = alpha_bar_ddpm
        elif schedule == "equal_snr":
            self.Sigma = self.C.clone()
            self.alpha_bar = calibrate_equal_snr_alpha_bar(alpha_bar_ddpm, self.C).clamp(1e-8, 1.0 - 1e-8)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def sample_t(self, B: int) -> torch.Tensor:
        return torch.randint(1, self.T + 1, (B,), device=self.device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Returns xt, y0, yt
        """
        B = x0.shape[0]
        ab = self.alpha_bar[t - 1].view(B, 1)  # alpha_bar at t
        y0 = x0 @ self.U  # (B,2)
        eps = torch.randn_like(y0)
        eps_sigma = eps * torch.sqrt(self.Sigma).view(1, 2)
        yt = torch.sqrt(ab) * y0 + torch.sqrt(1.0 - ab) * eps_sigma
        xt = yt @ self.U.t()
        return xt, y0, yt


# -----------------------
# DDIM sampling in PCA domain (toy analogue of Alg. 2)
# -----------------------
@torch.no_grad()
def ddim_sample_toy(model: nn.Module, fwd: ToyForwardProcess, n: int, steps: int) -> torch.Tensor:
    device = fwd.device
    B = n

    # Initialize y_T ~ N(0, Sigma) in PCA domain
    y = torch.randn(B, 2, device=device) * torch.sqrt(fwd.Sigma).view(1, 2)

    ts = torch.linspace(fwd.T, 1, steps, device=device).long()  # includes 1

    for i, t_scalar in enumerate(ts):
        t = torch.full((B,), int(t_scalar.item()), device=device, dtype=torch.long)
        ab_t = fwd.alpha_bar[t - 1].view(B, 1)

        x = y @ fwd.U.t()
        x0_hat = model(x, t)
        y0_hat = x0_hat @ fwd.U

        if i == len(ts) - 1:
            return x0_hat

        t_prev = int(ts[i + 1].item())
        ab_prev = fwd.alpha_bar[t_prev - 1].view(1, 1)

        y = torch.sqrt(ab_prev) * y0_hat + (torch.sqrt(1.0 - ab_prev) / torch.sqrt(1.0 - ab_t)) * (
            y - torch.sqrt(ab_t) * y0_hat
        )

    return y @ fwd.U.t()


# -----------------------
# Wasserstein metrics
# -----------------------
@torch.no_grad()
def sliced_wasserstein_1(x: torch.Tensor, y: torch.Tensor, n_proj: int = 256) -> torch.Tensor:
    """
    Sliced Wasserstein-1 in R^2 (approximation).
    """
    device = x.device
    N = min(x.shape[0], y.shape[0])
    x = x[:N]
    y = y[:N]

    dirs = torch.randn(n_proj, 2, device=device)
    dirs = dirs / torch.norm(dirs, dim=1, keepdim=True).clamp_min(1e-12)

    x_proj = x @ dirs.t()  # (N, n_proj)
    y_proj = y @ dirs.t()

    x_sort, _ = torch.sort(x_proj, dim=0)
    y_sort, _ = torch.sort(y_proj, dim=0)

    return torch.mean(torch.abs(x_sort - y_sort))


@torch.no_grad()
def sinkhorn_w2_approx(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.05,
    iters: int = 200,
    max_points: int = 2048,
) -> torch.Tensor:
    """
    Entropic OT (Sinkhorn) approximation of W2.
    Returns sqrt(<pi, ||x-y||^2>) as an approximate W2.
    Uses uniform marginals and subsamples to max_points.
    """
    device = x.device
    n = min(x.shape[0], y.shape[0], max_points)
    x = x[:n]
    y = y[:n]

    a = torch.full((n,), 1.0 / n, device=device)
    b = torch.full((n,), 1.0 / n, device=device)

    x2 = (x ** 2).sum(dim=1, keepdim=True)
    y2 = (y ** 2).sum(dim=1, keepdim=True).t()
    C = (x2 + y2 - 2.0 * (x @ y.t())).clamp_min(0.0)

    K = torch.exp(-C / eps).clamp_min(1e-12)

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(iters):
        u = a / (K @ v).clamp_min(1e-12)
        v = b / (K.t() @ u).clamp_min(1e-12)

    pi = (u.view(-1, 1) * K) * v.view(1, -1)
    w2_sq = torch.sum(pi * C)
    return torch.sqrt(w2_sq.clamp_min(0.0))


# -----------------------
# Plots
# -----------------------
@torch.no_grad()
def save_scatter(real: torch.Tensor, gen: torch.Tensor, path: str, lim: float = 8.0, n_show: int = 20000):
    real = real[:n_show].detach().cpu()
    gen = gen[:n_show].detach().cpu()

    plt.figure(figsize=(6, 6))
    plt.scatter(real[:, 0], real[:, 1], s=2, alpha=0.35, label="real")
    plt.scatter(gen[:, 0], gen[:, 1], s=2, alpha=0.35, label="gen")
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    plt.gca().set_aspect("equal", "box")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# -----------------------
# Training
# -----------------------
def train_toy(
    schedule: str,
    loader: DataLoader,
    X_all: torch.Tensor,
    device: torch.device,
    T: int,
    iters: int,
    lr: float,
    log_every: int,
    amp: bool,
    ema_decay: float,
    ema_warmup: int,
):
    model = ToyMLP(dim=2, t_dim=64, width=256, depth=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    fwd = ToyForwardProcess(schedule, X_all, T=T, device=device)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))
    ema = EMA(model, decay=ema_decay, warmup_steps=ema_warmup, device=device)

    model.train()
    it = iter(loader)

    for step in range(1, iters + 1):
        try:
            (x0_cpu,) = next(it)
        except StopIteration:
            it = iter(loader)
            (x0_cpu,) = next(it)

        x0 = x0_cpu.to(device).float()
        t = fwd.sample_t(x0.shape[0])
        xt, y0, _ = fwd.q_sample(x0, t)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            x0_hat = model(xt, t)

            # Paper-consistent comparison:
            # - DDPM baseline: pixel-space MSE
            # - EqualSNR: C^{-1/2}-weighted loss in PCA domain
            if schedule == "ddpm":
                loss = F.mse_loss(x0_hat, x0)
            else:
                y0_hat = x0_hat @ fwd.U
                diff = (y0 - y0_hat) / torch.sqrt(fwd.C).view(1, 2)
                loss = (diff ** 2).mean()

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        ema.update(model, step=step)

        if step % log_every == 0:
            print(f"[toy {schedule}] step {step}/{iters} loss {loss.item():.6f}")

    # Build an EMA model for sampling (same architecture)
    ema_model = ToyMLP(dim=2, t_dim=64, width=256, depth=4).to(device)
    ema.apply_to(ema_model)
    ema_model.eval()

    return model, ema_model, fwd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="outputs/toy")
    p.add_argument("--seed", type=int, default=0)

    # Training (fast)
    p.add_argument("--iters", type=int, default=2500)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)

    # EMA
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ema_warmup", type=int, default=200)

    # Diffusion
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--ddim_steps", type=int, default=200)

    # Data
    p.add_argument("--train_n", type=int, default=200000)
    p.add_argument("--eval_n", type=int, default=50000)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--radius", type=float, default=4.0)
    p.add_argument("--std", type=float, default=0.5)

    # Wasserstein
    p.add_argument("--sw_proj", type=int, default=256)
    p.add_argument("--sinkhorn_eps", type=float, default=0.05)
    p.add_argument("--sinkhorn_iters", type=int, default=200)
    p.add_argument("--sinkhorn_max_points", type=int, default=2048)

    # Perf
    p.add_argument("--no_amp", action="store_true")

    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()
    seed_all(args.seed)

    # Dataset
    X = make_mog(args.train_n, k=args.k, radius=args.radius, std=args.std, device=device)
    loader = DataLoader(
        TensorDataset(X.detach().cpu()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    schedules = ["ddpm", "equal_snr"]

    metrics_path = os.path.join(args.outdir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("schedule\twhich\ttrain_sec\tsliced_W1\tapprox_W2_sinkhorn\n")

    for s in schedules:
        t0 = time.time()
        model, ema_model, fwd = train_toy(
            schedule=s,
            loader=loader,
            X_all=X,
            device=device,
            T=args.T,
            iters=args.iters,
            lr=args.lr,
            log_every=200,
            amp=(not args.no_amp),
            ema_decay=args.ema_decay,
            ema_warmup=args.ema_warmup,
        )
        train_sec = time.time() - t0

        # Evaluate both raw model and EMA model (EMA usually samples better)
        for which, m in [("raw", model.eval()), ("ema", ema_model.eval())]:
            with torch.no_grad():
                gen = ddim_sample_toy(m, fwd, n=args.eval_n, steps=args.ddim_steps)
                real = make_mog(args.eval_n, k=args.k, radius=args.radius, std=args.std, device=device)

                sw1 = sliced_wasserstein_1(real, gen, n_proj=args.sw_proj)
                w2 = sinkhorn_w2_approx(
                    real,
                    gen,
                    eps=args.sinkhorn_eps,
                    iters=args.sinkhorn_iters,
                    max_points=args.sinkhorn_max_points,
                )

            save_scatter(real, gen, os.path.join(args.outdir, f"scatter_{s}_{which}.png"))

            print(
                f"[toy {s} {which}] train_time={train_sec:.1f}s  SW1={sw1.item():.4f}  approxW2(Sinkhorn)={w2.item():.4f}"
            )

            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(f"{s}\t{which}\t{train_sec:.6f}\t{sw1.item():.6f}\t{w2.item():.6f}\n")

    print("saved:", args.outdir)


if __name__ == "__main__":
    main()