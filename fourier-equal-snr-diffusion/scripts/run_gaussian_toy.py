#!/usr/bin/env python3
import argparse
import math
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from fourier_diffusion.utils.seed import seed_all, get_device
from fourier_diffusion.utils.plots import plot_curve, plot_single
from fourier_diffusion.diffusion.schedules import (
    make_beta_schedule,
    alphas_from_betas,
    calibrate_equal_snr_alpha_bar,
)
from fourier_diffusion.toy.gaussian_utils import (
    orthonormal_matrix,
    make_spectrum,
    sample_diag_gaussian,
    w2_diag,
    kl_diag,
)
from fourier_diffusion.toy.gaussian_train_eval import train_and_evaluate_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="outputs/gaussian_toy_train")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--d", type=int, default=50)
    p.add_argument("--p_decay", type=float, default=1.5)
    p.add_argument("--scale", type=float, default=1.0)

    p.add_argument("--n0", type=int, default=20000)
    p.add_argument("--nt", type=int, default=20000)

    p.add_argument("--train_n", type=int, default=200000)
    p.add_argument("--eval_n", type=int, default=50000)

    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--c_floor_rel", type=float, default=1e-3)

    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--timesteps", type=str, default="50,200,500,900")

    p.add_argument("--sw_proj", type=int, default=256)

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = get_device()
    seed_all(args.seed)

    C = make_spectrum(args.d, args.p_decay, args.scale, device=device)
    U = orthonormal_matrix(args.d, device=device, seed=args.seed)

    betas = make_beta_schedule(args.T, beta_start=1e-4, beta_end=2e-2, device=device, schedule="cosine")
    _, alpha_bar_ddpm = alphas_from_betas(betas)
    alpha_bar_ddpm = alpha_bar_ddpm.clamp(1e-8, 1.0 - 1e-8)
    alpha_bar_eq = calibrate_equal_snr_alpha_bar(alpha_bar_ddpm, C).clamp(1e-8, 1.0 - 1e-8)

    Sigma_ddpm = torch.ones_like(C)
    Sigma_eq = C.clone()

    y0 = sample_diag_gaussian(args.n0, C)
    x0 = y0 @ U.t()
    C_hat = y0.var(dim=0, unbiased=True).clamp_min(1e-12)

    w2_0 = w2_diag(C, C_hat)
    kl_0 = kl_diag(C, C_hat)

    with open(os.path.join(args.outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"d={args.d} p_decay={args.p_decay} scale={args.scale}\n")
        f.write(f"base spectrum estimate: W2_diag={w2_0:.6g}  KL_diag={kl_0:.6g}\n")

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

    t_list = [int(s.strip()) for s in args.timesteps.split(",") if s.strip() != ""]
    t_list = [t for t in t_list if 1 <= t <= args.T]
    if len(t_list) == 0:
        t_list = [50, 200, 500, 900]

    for t in t_list:
        ab_ddpm = alpha_bar_ddpm[t - 1].item()
        ab_eq = alpha_bar_eq[t - 1].item()

        y0t = sample_diag_gaussian(args.nt, C)
        eps = torch.randn_like(y0t)

        yt_ddpm = math.sqrt(ab_ddpm) * y0t + math.sqrt(1.0 - ab_ddpm) * (eps * torch.sqrt(Sigma_ddpm).view(1, -1))
        var_emp_ddpm = yt_ddpm.var(dim=0, unbiased=True).clamp_min(1e-12)
        var_the_ddpm = (ab_ddpm * C + (1.0 - ab_ddpm) * Sigma_ddpm).clamp_min(1e-12)

        yt_eq = math.sqrt(ab_eq) * y0t + math.sqrt(1.0 - ab_eq) * (eps * torch.sqrt(Sigma_eq).view(1, -1))
        var_emp_eq = yt_eq.var(dim=0, unbiased=True).clamp_min(1e-12)
        var_the_eq = (ab_eq * C + (1.0 - ab_eq) * Sigma_eq).clamp_min(1e-12)

        snr_ddpm = (ab_ddpm * C) / ((1.0 - ab_ddpm) * Sigma_ddpm).clamp_min(1e-12)
        snr_eq = (ab_eq * C) / ((1.0 - ab_eq) * Sigma_eq).clamp_min(1e-12)

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
            title=f"EqualSNR forward variance check at t={t}",
            xlabel="frequency index i",
            ylabel="Var(y_t,i)",
            outpath=os.path.join(args.outdir, f"var_match_equal_snr_t{t}.png"),
            logy=True,
        )
        plot_single(
            idx,
            snr_ddpm.detach().cpu().numpy(),
            title=f"DDPM SNR across frequencies at t={t}",
            xlabel="frequency index i",
            ylabel="SNR",
            outpath=os.path.join(args.outdir, f"snr_ddpm_t{t}.png"),
            logy=True,
        )
        plot_single(
            idx,
            snr_eq.detach().cpu().numpy(),
            title=f"EqualSNR SNR across frequencies at t={t}",
            xlabel="frequency index i",
            ylabel="SNR",
            outpath=os.path.join(args.outdir, f"snr_equal_snr_t{t}.png"),
            logy=True,
        )

    X_train = x0 if args.train_n == args.n0 else (sample_diag_gaussian(args.train_n, C) @ U.t())
    loader = DataLoader(
        TensorDataset(X_train.detach().cpu()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    train_and_evaluate_all(
        loader=loader,
        X_train=X_train,
        U=U,
        C=C,
        device=device,
        T=args.T,
        iters=args.iters,
        lr=args.lr,
        ema_decay=args.ema_decay,
        c_floor_rel=args.c_floor_rel,
        eval_n=args.eval_n,
        sw_proj=args.sw_proj,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()