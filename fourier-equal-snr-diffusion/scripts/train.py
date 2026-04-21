#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
import copy

import torch
import torchvision.utils as vutils
from PIL import Image

from fourier_diffusion.utils.seed import get_device, seed_all
from fourier_diffusion.utils.fft import radial_power_spectrum
from fourier_diffusion.utils.ema import EMA

from fourier_diffusion.data.mnist import get_grayscale_mnist
from fourier_diffusion.data.cifar10 import get_cifar10

from fourier_diffusion.models.unet import SimpleUNet
from fourier_diffusion.models.diffusers_unet import DiffusersUNet

from fourier_diffusion.diffusion.forward_process import FourierDiffusionConfig, FourierForwardProcess
from fourier_diffusion.diffusion.covariance import estimate_C_diag_rfft2
from fourier_diffusion.diffusion.losses import loss_x0_fourier_weighted
from fourier_diffusion.diffusion.sampling import ddim_sample


def save_image_grid(x: torch.Tensor, path: str, nrow: int = 8) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = x.clamp(-1, 1)
    x = (x + 1.0) * 0.5
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    grid = vutils.make_grid(x, nrow=nrow, padding=2)
    arr = (grid.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    Image.fromarray(arr, mode="RGB").save(path)


def save_loglog_spectrum_plot(real: torch.Tensor, gen: torch.Tensor, path: str, title: str = "") -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    real_spec = radial_power_spectrum(real).detach().cpu().clamp_min(1e-12)
    gen_spec = radial_power_spectrum(gen).detach().cpu().clamp_min(1e-12)
    n = min(real_spec.numel(), gen_spec.numel())
    x = torch.arange(1, n + 1)

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


def get_model_name(dataset: str, init_from_hf: bool) -> str:
    if dataset == "cifar10" and init_from_hf:
        return "diffusers_unet_hf"
    if dataset == "cifar10":
        return "diffusers_unet"
    return "simple_unet"


def build_model(
    dataset: str,
    in_ch: int,
    H: int,
    W: int,
    init_from_hf: bool = False,
    hf_repo_id: str = "google/ddpm-cifar10-32",
) -> torch.nn.Module:
    if dataset == "cifar10":
        if init_from_hf:
            return DiffusersUNet(
                in_ch=in_ch,
                sample_size=H,
                base_channels=128,
                repo_id=hf_repo_id,
            )
        return DiffusersUNet(in_ch=in_ch, sample_size=H, base_channels=128)
    return SimpleUNet(in_ch=in_ch, base=64, t_dim=128)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    p.add_argument("--schedule", choices=["ddpm", "equal_snr", "flipped_snr", "power_law"], required=True)
    p.add_argument("--iters", type=int, default=800000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--calibration", choices=["fixed_trace"], default="fixed_trace")

    p.add_argument("--init_from_hf", action="store_true")
    p.add_argument("--hf_repo_id", type=str, default="google/ddpm-cifar10-32")

    p.add_argument("--n_C_batches", type=int, default=200)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--sample_every", type=int, default=5000)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--steps_ddim", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_warmup", type=int, default=2000)
    p.add_argument("--c_floor_rel", type=float, default=1e-3)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--no_amp", action="store_true")
    args = p.parse_args()

    device = get_device()
    seed_all(args.seed)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    use_amp = (device.type == "cuda" and not args.no_amp)

    ckpt_dir = os.path.join(args.outdir, "checkpoints")
    plot_dir = os.path.join(args.outdir, "plots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    if args.dataset == "mnist":
        _, dl = get_grayscale_mnist(args.batch_size, root="./data", num_workers=args.num_workers)
        in_ch, H, W = 1, 28, 28
    else:
        _, dl = get_cifar10(args.batch_size, root="./data", num_workers=args.num_workers)
        in_ch, H, W = 3, 32, 32

    model_name = get_model_name(args.dataset, args.init_from_hf)

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

    model = build_model(args.dataset, in_ch, H, W, args.init_from_hf, args.hf_repo_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = EMA(model, decay=args.ema_decay, warmup_steps=args.ema_warmup, device=device)

    losses = []
    loader_iter = iter(dl)
    last_x0 = None
    start_step = 1

    def build_ema_model() -> torch.nn.Module:
        ema_model = copy.deepcopy(model).to(device)
        ema.apply_to(ema_model)
        ema_model.eval()
        return ema_model

    def save_ckpt(path: str, step: int) -> None:
        torch.save(
            {
                "step": step,
                "dataset": args.dataset,
                "schedule": args.schedule,
                "model_name": model_name,
                "hf_repo_id": args.hf_repo_id if args.init_from_hf else "",
                "lam": args.lam,
                "calibration": args.calibration,
                "seed": args.seed,
                "T": args.T,
                "in_ch": in_ch,
                "H": H,
                "W": W,
                "model_state": model.state_dict(),
                "ema_state": ema.state_dict(),
                "opt_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "C_diag": C_diag.detach().cpu() if C_diag is not None else None,
                "losses": losses,
                "args": vars(args),
            },
            path,
        )

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        if "ema_state" in ckpt:
            ema.load_state_dict(ckpt["ema_state"])
        losses = ckpt.get("losses", [])
        start_step = int(ckpt["step"]) + 1
        print(f"resumed from {args.resume} at step {start_step}")

    print(f"device={device} dataset={args.dataset} schedule={args.schedule} model={model_name} amp={use_amp}")

    model.train()

    for step in range(start_step, args.iters + 1):
        try:
            x0, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dl)
            x0, _ = next(loader_iter)

        x0 = x0.to(device, non_blocking=True).float()
        last_x0 = x0

        t = fwd.sample_t(x0.shape[0])

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, _ = loss_x0_fourier_weighted(
                model=model,
                fwd=fwd,
                x0=x0,
                t=t,
                c_floor_rel=args.c_floor_rel,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        ema.update(model, step=step)

        if step % args.log_every == 0:
            losses.append(float(loss.detach().cpu()))
            print(f"[{args.dataset} {args.schedule}] step {step}/{args.iters} loss {loss.item():.6f}")

        if step % args.save_every == 0:
            if args.schedule == "power_law":
                tag = f"{args.dataset}_{args.schedule}_lam{args.lam}_step{step}"
            else:
                tag = f"{args.dataset}_{args.schedule}_step{step}"
            save_ckpt(os.path.join(ckpt_dir, f"{tag}.pt"), step)

        if step % args.sample_every == 0:
            ema_model = build_ema_model()
            with torch.no_grad():
                samp = ddim_sample(ema_model, fwd, (64, in_ch, H, W), steps=args.steps_ddim).clamp(-1, 1)

            save_image_grid(
                samp,
                os.path.join(plot_dir, f"{args.dataset}_{args.schedule}_samples_step{step}.png"),
                nrow=8,
            )

            if last_x0 is not None:
                if args.schedule == "power_law":
                    spec_name = f"{args.dataset}_{args.schedule}_lam{args.lam}_spectrum_step{step}.png"
                else:
                    spec_name = f"{args.dataset}_{args.schedule}_spectrum_step{step}.png"
                save_loglog_spectrum_plot(
                    last_x0[: min(64, last_x0.shape[0])],
                    samp[: min(64, samp.shape[0])],
                    os.path.join(plot_dir, spec_name),
                    title=f"{args.dataset} {args.schedule} log-log radial spectrum @ step {step}",
                )

    if args.schedule == "power_law":
        final_tag = f"{args.dataset}_{args.schedule}_lam{args.lam}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        final_tag = f"{args.dataset}_{args.schedule}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_path = os.path.join(ckpt_dir, f"{final_tag}.pt")
    save_ckpt(ckpt_path, args.iters)

    ema_model = build_ema_model()
    with torch.no_grad():
        samp = ddim_sample(ema_model, fwd, (64, in_ch, H, W), steps=args.steps_ddim).clamp(-1, 1)

    save_image_grid(
        samp,
        os.path.join(plot_dir, f"{final_tag}_samples.png"),
        nrow=8,
    )

    if last_x0 is not None:
        save_loglog_spectrum_plot(
            last_x0[: min(64, last_x0.shape[0])],
            samp[: min(64, samp.shape[0])],
            os.path.join(plot_dir, f"{final_tag}_spectrum_loglog.png"),
            title=f"{args.dataset} {args.schedule} log-log radial spectrum",
        )

    with open(os.path.join(plot_dir, f"{final_tag}_run.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("saved checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()