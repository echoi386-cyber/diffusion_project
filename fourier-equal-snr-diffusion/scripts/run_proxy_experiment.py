#!/usr/bin/env python3
import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import torch


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def make_spectrum(d: int, p: float, scale: float, device: torch.device) -> torch.Tensor:
    i = torch.arange(1, d + 1, device=device, dtype=torch.float64)
    return (scale ** 2) * (i ** (-p))


def make_task_weights(d: int, q: float, device: torch.device) -> torch.Tensor:
    i = torch.arange(1, d + 1, device=device, dtype=torch.float64)
    return i ** q


def unrestricted_optimum(a: torch.Tensor, S: float, tol: float = 1e-12, max_iter: int = 200) -> Tuple[torch.Tensor, float]:
    """
    Solve s*_i = (sqrt(a_i / mu) - 1)_+ with sum_i s*_i = S.
    Returns s_star and mu.
    """
    if S <= 0:
        return torch.zeros_like(a), float('inf')

    a = a.clamp_min(0.0)
    amax = float(a.max().item())
    if amax == 0.0:
        return torch.zeros_like(a), float('inf')

    def budget(mu: float) -> float:
        mu_t = torch.tensor(mu, device=a.device, dtype=a.dtype)
        s = torch.sqrt(a / mu_t).sub(1.0).clamp_min(0.0)
        return float(s.sum().item())

    # Need f(lo) >= S and f(hi) <= S
    hi = max(amax, 1.0)
    while budget(hi) > S:
        hi *= 2.0

    lo = hi
    while budget(lo) < S:
        lo *= 0.5

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        b = budget(mid)
        if abs(b - S) <= tol * max(1.0, S):
            lo = hi = mid
            break
        if b > S:
            lo = mid
        else:
            hi = mid

    mu = 0.5 * (lo + hi)
    mu_t = torch.tensor(mu, device=a.device, dtype=a.dtype)
    s_star = torch.sqrt(a / mu_t).sub(1.0).clamp_min(0.0)
    return s_star, mu


def restricted_snr_family(C: torch.Tensor, S: float, lambdas: torch.Tensor) -> torch.Tensor:
    """
    Returns s(lambda) for all lambda.
    Output shape: (n_lambda, d)
    s_i(lambda) = S * C_i^(1-lambda) / sum_j C_j^(1-lambda)
    """
    logC = torch.log(C).view(1, -1)
    expo = (1.0 - lambdas.view(-1, 1)) * logC
    raw = torch.exp(expo)
    return S * raw / raw.sum(dim=1, keepdim=True)


def bayes_risk(a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    a: (d,) or broadcastable
    s: (..., d)
    returns (...,)
    """
    return (a.view(*([1] * (s.ndim - 1)), -1) / (1.0 + s)).sum(dim=-1)


def restricted_risk_gradient(C: torch.Tensor, a: torch.Tensor, S: float, lambdas: torch.Tensor) -> torch.Tensor:
    """
    Proposition 2 from the draft.
    d/dlambda R*(lambda) = S sum_i a_i p_i(lambda)(log C_i - E_p[log C]) / (1 + S p_i(lambda))^2
    """
    logC = torch.log(C).view(1, -1)
    expo = (1.0 - lambdas.view(-1, 1)) * logC
    raw = torch.exp(expo)
    p = raw / raw.sum(dim=1, keepdim=True)
    elog = (p * logC).sum(dim=1, keepdim=True)
    centered = logC - elog
    denom = (1.0 + S * p) ** 2
    grad = S * (a.view(1, -1) * p * centered / denom).sum(dim=1)
    return grad


def empirical_bayes_mc(C: torch.Tensor, s: torch.Tensor, w: torch.Tensor, n_mc: int, device: torch.device) -> float:
    """
    Optional Monte Carlo check that empirical posterior-mean risk matches exact Bayes risk.
    x_i ~ N(0,C_i), y_i = x_i + n_i, Var(n_i)=C_i/s_i.
    Posterior mean coefficient is s_i/(1+s_i).
    """
    if n_mc <= 0:
        return float('nan')
    x = torch.randn(n_mc, C.numel(), device=device, dtype=C.dtype) * torch.sqrt(C).view(1, -1)
    tau2 = C / s.clamp_min(1e-12)
    n = torch.randn_like(x) * torch.sqrt(tau2).view(1, -1)
    y = x + n
    coeff = (s / (1.0 + s)).view(1, -1)
    xhat = coeff * y
    risk = (w.view(1, -1) * (x - xhat).pow(2)).sum(dim=1).mean()
    return float(risk.item())


@dataclass
class Row:
    d: int
    p: float
    q: float
    S: float
    lambda_emp: float
    lambda_heur: float
    lambda_abs_err: float
    risk_unrestricted: float
    risk_restricted_best: float
    rel_gap: float
    active_frac: float
    mc_risk_unrestricted: float


def save_curve_plot(path: str, lambdas: torch.Tensor, risk_curves: Dict[float, torch.Tensor], title: str) -> None:
    plt.figure(figsize=(7.5, 4.8))
    for q, risk in sorted(risk_curves.items(), key=lambda kv: kv[0]):
        plt.plot(lambdas.cpu().numpy(), risk.cpu().numpy(), label=f"q={q:g}")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$R^*(\lambda)$")
    plt.title(title)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_heatmap(path: str, rows: List[Row], d: int, S: float, value_name: str, title: str) -> None:
    ps = sorted({r.p for r in rows if r.d == d and r.S == S})
    qs = sorted({r.q for r in rows if r.d == d and r.S == S})
    grid = torch.full((len(ps), len(qs)), float('nan'), dtype=torch.float64)

    for i, p in enumerate(ps):
        for j, q in enumerate(qs):
            r = next((x for x in rows if x.d == d and x.S == S and x.p == p and x.q == q), None)
            if r is None:
                continue
            grid[i, j] = getattr(r, value_name)

    plt.figure(figsize=(6.4, 4.8))
    im = plt.imshow(grid.numpy(), aspect='auto', origin='lower')
    plt.xticks(range(len(qs)), [f"{q:g}" for q in qs])
    plt.yticks(range(len(ps)), [f"{p:g}" for p in ps])
    plt.xlabel("q")
    plt.ylabel("p")
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel(value_name, rotation=270, labelpad=16)
    finite = grid[torch.isfinite(grid)]
    vmax = float(finite.abs().max().item()) if finite.numel() > 0 else 1.0
    for i in range(len(ps)):
        for j in range(len(qs)):
            val = grid[i, j].item()
            if math.isfinite(val):
                color = 'white' if abs(val) > 0.6 * vmax else 'black'
                plt.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8, color=color)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=str, default='outputs/proxy_experiment')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--dims', type=str, default='256,512,1024,2048')
    ap.add_argument('--p_values', type=str, default='0.5,1.0,1.5,2.0')
    ap.add_argument('--S_values', type=str, default='1,4,16,64')
    ap.add_argument('--lambda_min', type=float, default=-1.0)
    ap.add_argument('--lambda_max', type=float, default=2.0)
    ap.add_argument('--lambda_steps', type=int, default=301)
    ap.add_argument('--mc_samples', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plot_dir = os.path.join(args.outdir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    device = torch.device(args.device)
    dims = parse_int_list(args.dims)
    p_values = parse_float_list(args.p_values)
    S_values = parse_float_list(args.S_values)

    rows: List[Row] = []
    lambda_grid = torch.linspace(args.lambda_min, args.lambda_max, args.lambda_steps, device=device, dtype=torch.float64)

    for d in dims:
        for p in p_values:
            C = make_spectrum(d, p, args.scale, device=device)
            q_values = [-p, 0.0, 0.5 * p, p, 1.5 * p]

            for S in S_values:
                # Figure 1: risk vs lambda for different q, at fixed p,S,d
                risk_curves: Dict[float, torch.Tensor] = {}

                for q in q_values:
                    w = make_task_weights(d, q, device=device)
                    a = w * C

                    s_star, _ = unrestricted_optimum(a, S)
                    risk_star = float(bayes_risk(a, s_star.view(1, -1)).item())

                    s_lambda = restricted_snr_family(C, S, lambda_grid)
                    risk_lambda = bayes_risk(a, s_lambda)
                    grad_lambda = restricted_risk_gradient(C, a, S, lambda_grid)

                    best_idx = int(torch.argmin(risk_lambda).item())
                    lambda_emp = float(lambda_grid[best_idx].item())
                    lambda_heur = 0.5 + q / (2.0 * p)
                    risk_best = float(risk_lambda[best_idx].item())
                    rel_gap = (risk_best - risk_star) / max(risk_star, 1e-15)
                    active_frac = float((s_star > 0).double().mean().item())
                    mc_risk = empirical_bayes_mc(C, s_star, w, args.mc_samples, device) if args.mc_samples > 0 else float('nan')

                    rows.append(Row(
                        d=d,
                        p=p,
                        q=q,
                        S=S,
                        lambda_emp=lambda_emp,
                        lambda_heur=lambda_heur,
                        lambda_abs_err=abs(lambda_emp - lambda_heur),
                        risk_unrestricted=risk_star,
                        risk_restricted_best=risk_best,
                        rel_gap=rel_gap,
                        active_frac=active_frac,
                        mc_risk_unrestricted=mc_risk,
                    ))

                    risk_curves[q] = risk_lambda.detach().cpu()

                    # Save per-setting diagnostics
                    diag_path = os.path.join(plot_dir, f'diag_d{d}_p{p:g}_q{q:g}_S{S:g}.png')
                    plt.figure(figsize=(7.5, 4.8))
                    ax1 = plt.gca()
                    ax1.plot(lambda_grid.cpu().numpy(), risk_lambda.detach().cpu().numpy(), label='R*(lambda)')
                    ax1.axvline(lambda_emp, linestyle='--', label=f'emp={lambda_emp:.3f}')
                    ax1.axvline(lambda_heur, linestyle=':', label=f'heur={lambda_heur:.3f}')
                    ax1.set_xlabel(r'$\lambda$')
                    ax1.set_ylabel(r'$R^*(\lambda)$')
                    ax1.set_title(f'd={d}, p={p:g}, q={q:g}, S={S:g}')
                    ax2 = ax1.twinx()
                    ax2.plot(lambda_grid.cpu().numpy(), grad_lambda.detach().cpu().numpy(), alpha=0.35, label='dR/dlambda')
                    ax2.set_ylabel(r'$dR^*/d\lambda$')
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
                    plt.tight_layout()
                    plt.savefig(diag_path, dpi=160)
                    plt.close()

                curve_path = os.path.join(plot_dir, f'figure1_risk_curves_d{d}_p{p:g}_S{S:g}.png')
                save_curve_plot(
                    curve_path,
                    lambda_grid,
                    risk_curves,
                    title=f'Experiment A risk curves (d={d}, p={p:g}, S={S:g})',
                )

        # Figure 2 and 3: one heatmap per budget and dimension
        for S in S_values:
            save_heatmap(
                os.path.join(plot_dir, f'figure2_lambda_phase_d{d}_S{S:g}.png'),
                rows,
                d=d,
                S=S,
                value_name='lambda_emp',
                title=f'Empirical best lambda (d={d}, S={S:g})',
            )
            save_heatmap(
                os.path.join(plot_dir, f'figure3_rel_gap_d{d}_S{S:g}.png'),
                rows,
                d=d,
                S=S,
                value_name='rel_gap',
                title=f'Restricted-vs-unrestricted relative gap (d={d}, S={S:g})',
            )

    csv_path = os.path.join(args.outdir, 'experiment_a_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'd', 'p', 'q', 'S', 'lambda_emp', 'lambda_heur', 'lambda_abs_err',
            'risk_unrestricted', 'risk_restricted_best', 'rel_gap', 'active_frac',
            'mc_risk_unrestricted',
        ])
        for r in rows:
            writer.writerow([
                r.d, r.p, r.q, r.S, r.lambda_emp, r.lambda_heur, r.lambda_abs_err,
                r.risk_unrestricted, r.risk_restricted_best, r.rel_gap, r.active_frac,
                r.mc_risk_unrestricted,
            ])

    # Also save a compact markdown summary.
    md_path = os.path.join(args.outdir, 'README_results.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Experiment A results\n\n')
        f.write(f'- Device: {device}\n')
        f.write(f'- Dims: {dims}\n')
        f.write(f'- p values: {p_values}\n')
        f.write(f'- S values: {S_values}\n')
        f.write(f'- Lambda grid: [{args.lambda_min}, {args.lambda_max}] with {args.lambda_steps} points\n')
        f.write(f'- CSV: `{os.path.basename(csv_path)}`\n')
        f.write(f'- Plot dir: `{os.path.basename(plot_dir)}`\n')
    print('Saved Experiment A outputs to:', args.outdir)
    print('Main table:', csv_path)
    print('Plots:', plot_dir)


if __name__ == '__main__':
    main()
