#!/usr/bin/env python3
import argparse
import os
import json
from datetime import datetime

import torch
import torchvision.utils as vutils
from PIL import Image

from fourier_diffusion.utils.seed import get_device, seed_all
from fourier_diffusion.utils.fft import radial_power_spectrum

from fourier_diffusion.data.mnist import get_grayscale_mnist
from fourier_diffusion.data.cifar10 import get_cifar10

from fourier_diffusion.models.unet import SimpleUNet
from fourier_diffusion.diffusion.forward_process import FourierDiffusionConfig, FourierForwardProcess
from fourier_diffusion.diffusion.covariance import estimate_C_diag_rfft2
from fourier_diffusion.diffusion.losses import loss_x0_fourier_weighted
from fourier_diffusion.diffusion.sampling import ddim_sample


def save_image_grid(x: torch.Tensor, path: str, nrow: int = 8):
    """
    x: (B,C,H,W) in [-1,1]. Saves an RGB PNG grid.
    - For MNIST (C=1), repeats channels to RGB for consistent viewing.
    - For CIFAR (C=3), saves as RGB directly.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = x.clamp(-1, 1)
    x = (x + 1.0) * 0.5  # [0,1]
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    grid = vutils.make_grid(x, nrow=nrow, padding=2)
    arr = (grid.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    Image.fromarray(arr, mode="RGB").save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    p.add_argument("--schedule", choices=["ddpm", "equal_snr", "flipped_snr"], required=True)

    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--T", type=int, default=1000)

    p.add_argument("--n_C_batches", type=int, default=200)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--steps_ddim", type=int, default=200)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="outputs")
    args = p.parse_args()

    device = get_device()
    seed_all(args.seed)

    ckpt_dir = os.path.join(args.outdir, "checkpoints")
    plot_dir = os.path.join(args.outdir, "plots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Load data
    if args.dataset == "mnist":
        ds, dl = get_grayscale_mnist(args.batch_size, root="./data", num_workers=4)
        in_ch, H, W = 1, 28, 28
    else:
        ds, dl = get_cifar10(args.batch_size, root="./data", num_workers=4)
        in_ch, H, W = 3, 32, 32

    # Estimate C_diag only for Fourier schedules
    C_diag = None
    if args.schedule != "ddpm":
        C_diag = estimate_C_diag_rfft2(dl, device=device, n_batches=args.n_C_batches)

    fwd = FourierForwardProcess(
        FourierDiffusionConfig(T=args.T, schedule=args.schedule, calibrate_alpha_bar=True),
        C_diag=C_diag,
        device=device,
    )

    model = SimpleUNet(in_ch=in_ch, base=64, t_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    losses = []
    loader_iter = iter(dl)

    print(f"device={device} dataset={args.dataset} schedule={args.schedule}")

    last_x0 = None  # keep last batch for final spectrum plot

    for step in range(1, args.iters + 1):
        try:
            x0, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dl)
            x0, _ = next(loader_iter)

        x0 = x0.to(device, non_blocking=True).float()
        last_x0 = x0

        t = fwd.sample_t(x0.shape[0])
        loss, _ = loss_x0_fourier_weighted(model, fwd, x0, t)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0:
            losses.append(float(loss.detach().cpu()))
            print(f"[{args.dataset} {args.schedule}] step {step}/{args.iters} loss {loss.item():.4f}")

        if step % args.sample_every == 0:
            model.eval()
            with torch.no_grad():
                samp = ddim_sample(model, fwd, (64, in_ch, H, W), steps=args.steps_ddim).clamp(-1, 1)
            model.train()

            # Save sample image grid (MNIST and CIFAR)
            save_image_grid(
                samp,
                os.path.join(plot_dir, f"{args.dataset}_{args.schedule}_samples_step{step}.png"),
                nrow=8,
            )


    # Final plots + checkpoint
    tag = f"{args.dataset}_{args.schedule}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pt")

    torch.save(
        {
            "dataset": args.dataset,
            "schedule": args.schedule,
            "seed": args.seed,
            "T": args.T,
            "in_ch": in_ch,
            "H": H,
            "W": W,
            "model_state": model.state_dict(),
            "C_diag": C_diag.detach().cpu() if C_diag is not None else None,
            "losses": losses,
            "args": vars(args),
        },
        ckpt_path,
    )


    # Final sampling + save images + spectrum
    model.eval()
    with torch.no_grad():
        samp = ddim_sample(model, fwd, (64, in_ch, H, W), steps=args.steps_ddim).clamp(-1, 1)

    save_image_grid(
        samp,
        os.path.join(plot_dir, f"{tag}_samples.png"),
        nrow=8,
    )


    # Save run metadata
    with open(os.path.join(plot_dir, f"{tag}_run.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("saved checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()