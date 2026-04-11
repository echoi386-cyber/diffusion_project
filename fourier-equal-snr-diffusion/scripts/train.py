#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from PIL import Image

from fourier_diffusion.data.cifar10 import get_cifar10
from fourier_diffusion.diffusion.covariance import estimate_C_diag_rfft2
from fourier_diffusion.diffusion.forward_process import FourierDiffusionConfig, FourierForwardProcess
from fourier_diffusion.diffusion.losses import loss_x0_fourier_weighted
from fourier_diffusion.diffusion.sampling import ddim_sample
from fourier_diffusion.models.diffusers_unet import DiffusersUNet
from fourier_diffusion.utils.fft import radial_power_spectrum
from fourier_diffusion.utils.seed import get_device, seed_all


def save_image_grid(x: torch.Tensor, path: str, nrow: int = 8) -> None:
    """
    x: (B,C,H,W) in [-1,1]. Saves an RGB PNG grid.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = x.clamp(-1, 1)
    x = (x + 1.0) * 0.5  # [0,1]
    grid = vutils.make_grid(x, nrow=nrow, padding=2)
    arr = (grid.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    Image.fromarray(arr, mode="RGB").save(path)


def save_loglog_spectrum_plot(
    real: torch.Tensor,
    gen: torch.Tensor,
    path: str,
    title: str = "",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    real_spec = radial_power_spectrum(real).detach().cpu().clamp_min(1e-12)
    gen_spec = radial_power_spectrum(gen).detach().cpu().clamp_min(1e-12)

    n = min(real_spec.numel(), gen_spec.numel())
    x = torch.arange(1, n + 1, dtype=torch.float32)

    plt.figure(figsize=(6, 4))
    plt.loglog(x.numpy(), real_spec[:n].numpy(), label="real")
    plt.loglog(x.numpy(), gen_spec[:n].numpy(), label="generated")
    plt.xlabel("radial frequency bin")
    plt.ylabel("power")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()

    # CIFAR-focused paper-closer route
    p.add_argument("--dataset", choices=["cifar10"], required=True)

    # forward process family
    p.add_argument(
        "--schedule",
        choices=["ddpm", "equal_snr", "flipped_snr", "power_law"],
        required=True,
    )
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--calibration", choices=["fixed_trace"], default="fixed_trace")

    # training
    p.add_argument("--iters", type=int, default=800000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--T", type=int, default=1000)

    # bookkeeping / evaluation during training
    p.add_argument("--n_C_batches", type=int, default=200)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--sample_every", type=int, default=50000)
    p.add_argument("--steps_ddim", type=int, default=200)

    # system
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--no_amp", action="store_true")

    args = p.parse_args()

    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    use_amp = (device.type == "cuda" and not args.no_amp)
    seed_all(args.seed)

    lam_str = f"{args.lam:g}"

    ckpt_dir = os.path.join(args.outdir, "checkpoints")
    plot_dir = os.path.join(args.outdir, "plots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # CIFAR-10 only
    _, dl = get_cifar10(args.batch_size, root="./data", num_workers=args.num_workers)
    in_ch, H, W = 3, 32, 32

    # Estimate Fourier variance only for non-DDPM schedules
    C_diag = None
    if args.schedule != "ddpm":
        C_diag = estimate_C_diag_rfft2(dl, device=device, n_batches=args.n_C_batches)

    fwd = FourierForwardProcess(
        FourierDiffusionConfig(
            T=args.T,
            schedule=args.schedule,
            lam=args.lam,
            calibration=args.calibration,
            calibrate_alpha_bar=True,
        ),
        C_diag=C_diag,
        device=device,
    )

    model = DiffusersUNet(
        in_ch=in_ch,
        sample_size=H,
        base_channels=128,
    ).to(device)

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    losses = []
    loader_iter = iter(dl)
    last_x0 = None

    print(
        f"device={device} dataset={args.dataset} schedule={args.schedule} "
        f"lam={lam_str} calibration={args.calibration}"
    )

    def save_ckpt(path: str, step: int) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "step": step,
                "dataset": args.dataset,
                "schedule": args.schedule,
                "lam": args.lam,
                "calibration": args.calibration,
                "seed": args.seed,
                "T": args.T,
                "in_ch": in_ch,
                "H": H,
                "W": W,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "C_diag": C_diag.detach().cpu() if C_diag is not None else None,
                "losses": losses,
                "args": vars(args),
            },
            path,
        )

    start_step = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        losses = ckpt.get("losses", [])
        start_step = int(ckpt["step"]) + 1
        print(f"Resumed from {args.resume} at step {start_step}")

    model.train()
    last_completed_step = start_step - 1

    for step in range(start_step, args.iters + 1):
        try:
            x0, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dl)
            x0, _ = next(loader_iter)

        x0 = x0.to(device, non_blocking=True).float()
        if device.type == "cuda":
            x0 = x0.contiguous(memory_format=torch.channels_last)

        last_x0 = x0.detach()
        t = fwd.sample_t(x0.shape[0])

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, _ = loss_x0_fourier_weighted(model, fwd, x0, t)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        last_completed_step = step

        if step % args.log_every == 0:
            losses.append(float(loss.detach().cpu()))
            print(
                f"[{args.dataset} {args.schedule} lam={lam_str}] "
                f"step {step}/{args.iters} loss {loss.item():.6f}"
            )

        if step % args.save_every == 0:
            tag = f"{args.dataset}_{args.schedule}_lam{lam_str}_step{step}"
            save_ckpt(os.path.join(ckpt_dir, f"{tag}.pt"), step)

        if step % args.sample_every == 0:
            model.eval()
            with torch.no_grad():
                samp = ddim_sample(
                    model,
                    fwd,
                    (64, in_ch, H, W),
                    steps=args.steps_ddim,
                ).clamp(-1, 1)
            model.train()

            save_image_grid(
                samp,
                os.path.join(
                    plot_dir,
                    f"{args.dataset}_{args.schedule}_lam{lam_str}_samples_step{step}.png",
                ),
                nrow=8,
            )

            if last_x0 is not None:
                save_loglog_spectrum_plot(
                    last_x0[: min(64, last_x0.shape[0])],
                    samp[: min(64, samp.shape[0])],
                    os.path.join(
                        plot_dir,
                        f"{args.dataset}_{args.schedule}_lam{lam_str}_spectrum_step{step}.png",
                    ),
                    title=(
                        f"{args.dataset} {args.schedule} lam={lam_str} "
                        f"log-log radial spectrum @ step {step}"
                    ),
                )

    tag = (
        f"{args.dataset}_{args.schedule}_lam{lam_str}_seed{args.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pt")
    save_ckpt(ckpt_path, last_completed_step)

    model.eval()
    with torch.no_grad():
        samp = ddim_sample(
            model,
            fwd,
            (64, in_ch, H, W),
            steps=args.steps_ddim,
        ).clamp(-1, 1)

    save_image_grid(
        samp,
        os.path.join(plot_dir, f"{tag}_samples.png"),
        nrow=8,
    )

    if last_x0 is not None:
        save_loglog_spectrum_plot(
            last_x0[: min(64, last_x0.shape[0])],
            samp[: min(64, samp.shape[0])],
            os.path.join(plot_dir, f"{tag}_spectrum_loglog.png"),
            title=f"{args.dataset} {args.schedule} lam={lam_str} log-log radial spectrum",
        )

    with open(os.path.join(plot_dir, f"{tag}_run.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("saved checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()