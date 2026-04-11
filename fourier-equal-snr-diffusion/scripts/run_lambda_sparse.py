#!/usr/bin/env python3
import argparse
import os
from typing import List

import torch
from torch.utils.data import DataLoader

from fourier_diffusion.utils.seed import get_device, seed_all
from fourier_diffusion.utils.sparse_signals import KSparseConfig, KSparseFourierDataset
from fourier_diffusion.diffusion.forward_process_1d import (
    Fourier1DLambdaConfig, Fourier1DLambdaForwardProcess, estimate_C_diag_rfft1
)
from fourier_diffusion.diffusion.losses_1d import loss_x0_fourier_weighted_1d
from fourier_diffusion.diffusion.sampling_1d import reconstruct_from_xt_T
from fourier_diffusion.models.signal_mlp import SignalMLP
from fourier_diffusion.utils.spectrum_1d import exact_k_support_recovery_rate


def parse_lams(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="outputs/lambda_sparse")
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n_train", type=int, default=200000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--test_batch", type=int, default=512)

    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--beta_schedule", type=str, default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--beta_start", type=float, default=1e-4)
    ap.add_argument("--beta_end", type=float, default=2e-2)
    ap.add_argument("--cosine_s", type=float, default=0.008)

    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log_every", type=int, default=200)

    ap.add_argument("--lams", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.3,1.5,1.7,2")
    ap.add_argument("--recon_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_C_batches", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()
    seed_all(args.seed)
    print("device:", device)

    ds_cfg = KSparseConfig(length=args.length, k=args.k, seed=args.seed)
    train_ds = KSparseFourierDataset(ds_cfg, n_samples=args.n_train)
    test_ds = KSparseFourierDataset(ds_cfg, n_samples=args.n_test)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=args.test_batch, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    # Estimate C once
    print("Estimating C_diag ...")
    C_diag = estimate_C_diag_rfft1(train_dl, device=device, n_batches=args.n_C_batches)

    lams = parse_lams(args.lams)
    results = []

    for lam in lams:
        print(f"\n=== lambda={lam} ===")

        fwd = Fourier1DLambdaForwardProcess(
            Fourier1DLambdaConfig(
                T=args.T,
                beta_start=args.beta_start,
                beta_end=args.beta_end,
                beta_schedule=args.beta_schedule,
                cosine_s=args.cosine_s,
                lam=lam,
            ),
            C_diag=C_diag,
            device=device,
        )

        model = SignalMLP(length=args.length, hidden=256, depth=4, t_dim=128).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # Train
        it = iter(train_dl)
        loss_hist = []

        for step in range(1, args.iters + 1):
            try:
                x0, _ = next(it)
            except StopIteration:
                it = iter(train_dl)
                x0, _ = next(it)

            x0 = x0.to(device, non_blocking=True).float()
            t = fwd.sample_t(x0.shape[0])

            loss, _ = loss_x0_fourier_weighted_1d(model, fwd, x0, t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % args.log_every == 0:
                loss_hist.append(float(loss.detach().cpu()))
                print(f"[lam={lam}] step {step}/{args.iters} loss {loss.item():.4f}")

        # Evaluate exact support recovery at t=T
        model.eval()
        rates = []
        with torch.no_grad():
            for x0, bins in test_dl:
                x0 = x0.to(device).float()
                bins = bins.to(device)

                tT = torch.full((x0.shape[0],), args.T, device=device, dtype=torch.long)
                xt_T, _ = fwd.q_sample(x0, tT)

                x_rec = reconstruct_from_xt_T(model, fwd, xt_T, steps=args.recon_steps)
                rates.append(exact_k_support_recovery_rate(x_rec, bins, k=args.k))

        exact_rate = float(torch.tensor(rates).mean().item())
        print(f"[lam={lam}] exact k-support recovery rate = {exact_rate:.4f}")

        torch.save(
            {"lambda": lam, "loss_hist": loss_hist, "exact_rate": exact_rate},
            os.path.join(args.outdir, f"lam_{lam}.pt"),
        )
        results.append((lam, exact_rate))

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n=== Summary (best first) ===")
    for lam, rate in results:
        print(f"lambda={lam:<6g} exact_recovery={rate:.4f}")

    print("saved:", args.outdir)


if __name__ == "__main__":
    main()