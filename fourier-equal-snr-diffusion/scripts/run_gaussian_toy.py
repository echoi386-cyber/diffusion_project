#!/usr/bin/env python3
import argparse
import os
import math

import torch
import matplotlib.pyplot as plt

from fourier_diffusion.utils.seed import seed_all, get_device
from fourier_diffusion.diffusion.schedules import (
    make_beta_schedule,
    alphas_from_betas,
    calibrate_equal_snr_alpha_bar,
)


def orthonormal_matrix(d: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    """
    Random orthonormal matrix via QR. Deterministic given seed.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    A = torch.randn(d, d, device=device, generator=g)
    Q, R = torch.linalg.qr(A)
    # Fix sign so it's deterministic
    diag = torch.sign(torch.diag(R))
    diag[diag == 0] = 1.0
    Q = Q * diag
    return Q


def make_spectrum(d: int, p: float, scale: float, device: torch.device) -> torch.Tensor:
    """
    C_i = scale^2 * i^{-p}, i=1..d
    """
    i = torch.arange(1, d + 1, device=device, dtype=torch.float32)
    C = (scale ** 2) * (i ** (-p))
    return C


@torch.no_grad()
def sample_diag_gaussian(n: int, C: torch.Tensor) -> torch.Tensor:
    """
    Sample y ~ N(0, diag(C)).
    Returns (n,d).
    """
    eps = torch.randn(n, C.numel(), device=C.device)
    return eps * torch.sqrt(C).view(1, -1)


def w2_diag(C: torch.Tensor, Chat: torch.Tensor) -> float:
    """
    Closed-form W2 for zero-mean diagonal Gaussians:
      W2^2 = sum_i (sqrt(C_i) - sqrt(Chat_i))^2
    """
    val = torch.sum((torch.sqrt(C) - torch.sqrt(Chat.clamp_min(1e-12))) ** 2)
    return float(torch.sqrt(val).detach().cpu())


def kl_diag(C: torch.Tensor, Chat: torch.Tensor) -> float:
    """
    KL( N(0,C) || N(0,Chat) ) for diagonal covariances:
      0.5 * sum_i ( C_i/Chat_i - 1 - log(C_i/Chat_i) )
    """
    Chat = Chat.clamp_min(1e-12)
    ratio = C / Chat
    val = 0.5 * torch.sum(ratio - 1.0 - torch.log(ratio.clamp_min(1e-12)))
    return float(val.detach().cpu())


def plot_curve(x, y_true, y_emp, title, xlabel, ylabel, outpath, logy=False):
    plt.figure(figsize=(7, 4))
    plt.plot(x, y_true, label="theory")
    plt.plot(x, y_emp, label="empirical")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_single(x, y, title, xlabel, ylabel, outpath, logy=False):
    plt.figure(figsize=(7, 4))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="outputs/gaussian_spectrum_check")
    p.add_argument("--seed", type=int, default=0)

    # spectrum / dimension
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--p_decay", type=float, default=1.5)
    p.add_argument("--scale", type=float, default=1.0)

    # sample sizes (CPU-friendly)
    p.add_argument("--n0", type=int, default=20000, help="samples for estimating base spectrum")
    p.add_argument("--nt", type=int, default=20000, help="samples per timestep for forward variance checks")

    # diffusion
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--timesteps", type=str, default="50,200,500,900")

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = get_device()
    seed_all(args.seed)

    # ----- Construct a true high-dimensional "frequency spectrum" -----
    # True diagonal signal covariance in "frequency space"
    C = make_spectrum(args.d, args.p_decay, args.scale, device=device)

    # Optional orthonormal mixing (keeps Gaussianity but makes x-space non-diagonal)
    # This mimics "pixel space" where covariance is not diagonal, while frequency space is.
    U = orthonormal_matrix(args.d, device=device, seed=args.seed)

    # ----- DDPM base schedule -----
    betas = make_beta_schedule(args.T, beta_start=1e-4, beta_end=2e-2, device=device, schedule="cosine")
    _, alpha_bar_ddpm = alphas_from_betas(betas)
    alpha_bar_ddpm = alpha_bar_ddpm.clamp(1e-8, 1.0 - 1e-8)

    # EqualSNR calibrated alpha_bar (uses your existing calibration)
    alpha_bar_eq = calibrate_equal_snr_alpha_bar(alpha_bar_ddpm, C).clamp(1e-8, 1.0 - 1e-8)

    # Noise covariances (diagonal)
    Sigma_ddpm = torch.ones_like(C)
    Sigma_eq = C.clone()

    # ----- Base sanity: does our empirical C_hat match C? -----
    y0 = sample_diag_gaussian(args.n0, C)         # frequency space
    x0 = y0 @ U.t()                                # "pixel" space (not used further, but present conceptually)

    # Estimate spectrum in frequency basis (we know basis, so just use y0)
    C_hat = y0.var(dim=0, unbiased=True).clamp_min(1e-12)

    # Closed-form distances between true and empirical (should be small with enough samples)
    w2_0 = w2_diag(C, C_hat)
    kl_0 = kl_diag(C, C_hat)

    with open(os.path.join(args.outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"d={args.d} p_decay={args.p_decay} scale={args.scale}\n")
        f.write(f"base spectrum estimate: W2_diag={w2_0:.6g}  KL_diag={kl_0:.6g}\n")

    # Plot true vs empirical spectrum
    idx = torch.arange(1, args.d + 1, device=device).detach().cpu().numpy()
    plot_curve(
        idx,
        C.detach().cpu().numpy(),
        C_hat.detach().cpu().numpy(),
        title=f"Base spectrum check: true C_i vs empirical (n={args.n0})",
        xlabel="frequency index i",
        ylabel="variance C_i",
        outpath=os.path.join(args.outdir, "spectrum_true_vs_empirical.png"),
        logy=True,
    )

    # ----- Forward-process variance + SNR checks at multiple timesteps -----
    t_list = [int(s.strip()) for s in args.timesteps.split(",") if s.strip() != ""]
    t_list = [t for t in t_list if 1 <= t <= args.T]
    if len(t_list) == 0:
        t_list = [50, 200, 500, 900]

    for t in t_list:
        ab_ddpm = alpha_bar_ddpm[t - 1].item()
        ab_eq = alpha_bar_eq[t - 1].item()

        # Draw fresh y0 each timestep check (avoid reusing same y0)
        y0t = sample_diag_gaussian(args.nt, C)

        # Sample forward y_t for each schedule (in frequency space)
        eps = torch.randn_like(y0t)

        # DDPM forward: Sigma=I
        yt_ddpm = math.sqrt(ab_ddpm) * y0t + math.sqrt(1.0 - ab_ddpm) * (eps * torch.sqrt(Sigma_ddpm).view(1, -1))
        var_emp_ddpm = yt_ddpm.var(dim=0, unbiased=True).clamp_min(1e-12)
        var_the_ddpm = (ab_ddpm * C + (1.0 - ab_ddpm) * Sigma_ddpm).clamp_min(1e-12)

        # EqualSNR forward: Sigma=C, calibrated alpha_bar
        yt_eq = math.sqrt(ab_eq) * y0t + math.sqrt(1.0 - ab_eq) * (eps * torch.sqrt(Sigma_eq).view(1, -1))
        var_emp_eq = yt_eq.var(dim=0, unbiased=True).clamp_min(1e-12)
        var_the_eq = (ab_eq * C + (1.0 - ab_eq) * Sigma_eq).clamp_min(1e-12)

        # SNR curves across i
        # s_t(i) = ab*C_i / ((1-ab)*Sigma_ii)
        snr_ddpm = (ab_ddpm * C) / ((1.0 - ab_ddpm) * Sigma_ddpm).clamp_min(1e-12)
        snr_eq = (ab_eq * C) / ((1.0 - ab_eq) * Sigma_eq).clamp_min(1e-12)

        # Plot variance match (theory vs empirical) across frequencies
        plot_curve(
            idx,
            var_the_ddpm.detach().cpu().numpy(),
            var_emp_ddpm.detach().cpu().numpy(),
            title=f"DDPM forward variance check at t={t}",
            xlabel="frequency index i",
            ylabel="Var(y_t,i)",
            outpath=os.path.join(args.outdir, f"var_match_ddpm_t{t}.png"),
            logy=True,
        )
        plot_curve(
            idx,
            var_the_eq.detach().cpu().numpy(),
            var_emp_eq.detach().cpu().numpy(),
            title=f"EqualSNR forward variance check at t={t} (calibrated)",
            xlabel="frequency index i",
            ylabel="Var(y_t,i)",
            outpath=os.path.join(args.outdir, f"var_match_equal_snr_t{t}.png"),
            logy=True,
        )

        # Plot SNR hierarchy vs flatness
        plot_single(
            idx,
            snr_ddpm.detach().cpu().numpy(),
            title=f"DDPM SNR across frequencies at t={t} (hierarchical)",
            xlabel="frequency index i",
            ylabel="SNR s_t(i)",
            outpath=os.path.join(args.outdir, f"snr_ddpm_t{t}.png"),
            logy=True,
        )
        plot_single(
            idx,
            snr_eq.detach().cpu().numpy(),
            title=f"EqualSNR SNR across frequencies at t={t} (should be ~flat)",
            xlabel="frequency index i",
            ylabel="SNR s_t(i)",
            outpath=os.path.join(args.outdir, f"snr_equal_snr_t{t}.png"),
            logy=True,
        )

        # Numeric error summaries for this timestep (cheap + objective)
        rel_err_ddpm = float(torch.mean(torch.abs(var_emp_ddpm - var_the_ddpm) / var_the_ddpm).detach().cpu())
        rel_err_eq = float(torch.mean(torch.abs(var_emp_eq - var_the_eq) / var_the_eq).detach().cpu())

        with open(os.path.join(args.outdir, "summary.txt"), "a", encoding="utf-8") as f:
            f.write(f"t={t}: rel_err_var DDPM={rel_err_ddpm:.6g}  EqualSNR={rel_err_eq:.6g}\n")

    print("Saved check outputs to:", args.outdir)
    print("Key file:", os.path.join(args.outdir, "summary.txt"))
    print("Look at: spectrum_true_vs_empirical.png and snr_*.png and var_match_*.png")


if __name__ == "__main__":
    main()