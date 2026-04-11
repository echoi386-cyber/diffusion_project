#!/usr/bin/env python3
import argparse
import os
import json
from datetime import datetime

import torch
import torchvision.utils as vutils
from torchvision.utils import save_image
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

from fourier_diffusion.utils.fid import calculate_fid_given_paths


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


def save_spectrum_plot(spec: torch.Tensor, path: str, title: str):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    y = spec.detach().cpu().float()
    plt.figure(figsize=(7, 4))
    plt.plot(y)
    plt.xlabel("radial frequency bin")
    plt.ylabel("power")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


@torch.no_grad()
def save_real_images_for_fid(ds, outdir: str, num: int):
    os.makedirs(outdir, exist_ok=True)
    n = min(num, len(ds))
    for i in range(n):
        img, _ = ds[i]   # already normalized to [-1,1]
        img = img.clamp(-1, 1)
        img = (img + 1.0) * 0.5
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        save_image(img, os.path.join(outdir, f"{i:06d}.png"))


@torch.no_grad()
def save_generated_images_for_fid(model, fwd, outdir: str, total_num: int, batch_size: int, shape, steps_ddim: int):
    os.makedirs(outdir, exist_ok=True)
    saved = 0
    while saved < total_num:
        cur_bs = min(batch_size, total_num - saved)
        samp = ddim_sample(model, fwd, (cur_bs, *shape), steps=steps_ddim).clamp(-1, 1)
        samp = (samp + 1.0) * 0.5
        if samp.shape[1] == 1:
            samp = samp.repeat(1, 3, 1, 1)
        for i in range(cur_bs):
            save_image(samp[i], os.path.join(outdir, f"{saved + i:06d}.png"))
        saved += cur_bs


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

    p.add_argument("--num_fid_samples", type=int, default=10000)
    p.add_argument("--fid_batch_size", type=int, default=100)
    p.add_argument("--fid_dims", type=int, default=2048)
    p.add_argument("--fid_num_workers", type=int, default=1)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="outputs")
    args = p.parse_args()

    device = get_device()
    seed_all(args.seed)

    ckpt_dir = os.path.join(args.outdir, "checkpoints")
    plot_dir = os.path.join(args.outdir, "plots")
    fid_dir = os.path.join(args.outdir, "fid")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(fid_dir, exist_ok=True)

    # Load data
    if args.dataset == "mnist":
        ds, dl = get_grayscale_mnist(args.batch_size, root="./data", num_workers=4)
        eval_ds, _ = get_grayscale_mnist(args.batch_size, root="./data", num_workers=4)
        in_ch, H, W = 1, 28, 28
    else:
        ds, dl = get_cifar10(args.batch_size, root="./data", num_workers=4)
        eval_ds = ds
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

    # 1) dataset frequency spectrum plot
    first_batch, _ = next(iter(dl))
    first_batch = first_batch.to(device).float()
    dataset_spec = radial_power_spectrum(first_batch[:256])
    save_spectrum_plot(
        dataset_spec,
        os.path.join(plot_dir, f"{args.dataset}_{args.schedule}_dataset_spectrum.png"),
        title=f"{args.dataset} dataset frequency spectrum",
    )

    for step in range(1, args.iters + 1):
        try:
            x0, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dl)
            x0, _ = next(loader_iter)

        x0 = x0.to(device, non_blocking=True).float()

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

            # 2) sample plot during training
            save_image_grid(
                samp,
                os.path.join(plot_dir, f"{args.dataset}_{args.schedule}_samples_step{step}.png"),
                nrow=8,
            )

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

    # Final samples
    model.eval()
    with torch.no_grad():
        samp = ddim_sample(model, fwd, (64, in_ch, H, W), steps=args.steps_ddim).clamp(-1, 1)

    save_image_grid(
        samp,
        os.path.join(plot_dir, f"{tag}_samples.png"),
        nrow=8,
    )

    # 3) FID for grayscale-MNIST DDPM and EqualSNR experiments
    fid_value = None
    if args.dataset == "mnist":
        real_dir = os.path.join(fid_dir, f"{tag}_real")
        gen_dir = os.path.join(fid_dir, f"{tag}_gen")

        save_real_images_for_fid(eval_ds, real_dir, args.num_fid_samples)
        save_generated_images_for_fid(
            model=model,
            fwd=fwd,
            outdir=gen_dir,
            total_num=args.num_fid_samples,
            batch_size=args.fid_batch_size,
            shape=(in_ch, H, W),
            steps_ddim=args.steps_ddim,
        )

        fid_value = float(calculate_fid_given_paths(
            [gen_dir, real_dir],
            batch_size=args.fid_batch_size,
            device=device,
            dims=args.fid_dims,
            num=args.num_fid_samples,
            num_workers=args.fid_num_workers,
        ))
        print(f"FID ({args.schedule}) = {fid_value:.6f}")

    # Save run metadata
    run_info = vars(args).copy()
    run_info["checkpoint"] = ckpt_path
    run_info["fid"] = fid_value
    with open(os.path.join(plot_dir, f"{tag}_run.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    print("saved checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()