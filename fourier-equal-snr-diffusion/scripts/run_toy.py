#!/usr/bin/env python3
import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fourier_diffusion.utils.seed import get_device, seed_all
from fourier_diffusion.utils.ema import EMA
from fourier_diffusion.data.toy import make_mog_ring
from fourier_diffusion.models.toy_mlp import ToyMLP
from fourier_diffusion.diffusion.toy_forward import ToyForward
from fourier_diffusion.diffusion.toy_sampling import ddim_sample_toy
from fourier_diffusion.utils.wasserstein import sliced_wasserstein_1, sinkhorn_w2_approx
from fourier_diffusion.utils.plots import save_scatter


def train_one(
    schedule: str,
    loader: DataLoader,
    X_all: torch.Tensor,
    device: torch.device,
    T: int,
    iters: int,
    lr: float,
    amp: bool,
    ema_decay: float,
    ema_warmup: int,
):
    model = ToyMLP(dim=2, t_dim=64, width=256, depth=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    fwd = ToyForward.from_data(schedule=schedule, X=X_all, T=T, device=device)

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

    # Build EMA model for sampling
    ema_model = ToyMLP(dim=2, t_dim=64, width=256, depth=4).to(device)
    ema.apply_to(ema_model)
    ema_model.eval()

    model.eval()
    return model, ema_model, fwd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="outputs/toy")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--iters", type=int, default=12000)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ema_warmup", type=int, default=200)

    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--ddim_steps", type=int, default=600)

    p.add_argument("--train_n", type=int, default=200000)
    p.add_argument("--eval_n", type=int, default=50000)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--radius", type=float, default=4.0)
    p.add_argument("--std", type=float, default=0.5)

    p.add_argument("--sw_proj", type=int, default=256)
    p.add_argument("--sinkhorn_eps", type=float, default=0.05)
    p.add_argument("--sinkhorn_iters", type=int, default=200)
    p.add_argument("--sinkhorn_max_points", type=int, default=2048)

    p.add_argument("--no_amp", action="store_true")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()
    seed_all(args.seed)

    X = make_mog_ring(args.train_n, k=args.k, radius=args.radius, std=args.std, device=device)
    loader = DataLoader(
        TensorDataset(X.detach().cpu()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    metrics_path = os.path.join(args.outdir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("schedule\twhich\ttrain_sec\tsliced_W1\tapprox_W2_sinkhorn\n")

    for schedule in ["ddpm", "equal_snr"]:
        t0 = time.time()
        raw_model, ema_model, fwd = train_one(
            schedule=schedule,
            loader=loader,
            X_all=X,
            device=device,
            T=args.T,
            iters=args.iters,
            lr=args.lr,
            amp=(not args.no_amp),
            ema_decay=args.ema_decay,
            ema_warmup=args.ema_warmup,
        )
        train_sec = time.time() - t0

        for which, model in [("raw", raw_model), ("ema", ema_model)]:
            with torch.no_grad():
                gen = ddim_sample_toy(model, fwd, n=args.eval_n, steps=args.ddim_steps)
                real = make_mog_ring(args.eval_n, k=args.k, radius=args.radius, std=args.std, device=device)

                sw1 = sliced_wasserstein_1(real, gen, n_proj=args.sw_proj)
                w2 = sinkhorn_w2_approx(
                    real,
                    gen,
                    eps=args.sinkhorn_eps,
                    iters=args.sinkhorn_iters,
                    max_points=args.sinkhorn_max_points,
                )

            save_scatter(real, gen, os.path.join(args.outdir, f"scatter_{schedule}_{which}.png"))

            print(f"[toy {schedule} {which}] train_time={train_sec:.1f}s  SW1={sw1.item():.4f}  approxW2(Sinkhorn)={w2.item():.4f}")

            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(f"{schedule}\t{which}\t{train_sec:.6f}\t{sw1.item():.6f}\t{w2.item():.6f}\n")


if __name__ == "__main__":
    main()