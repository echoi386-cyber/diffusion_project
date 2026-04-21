#!/usr/bin/env python3
import argparse
import os

import torch
from PIL import Image

from fourier_diffusion.models.unet import SimpleUNet
from fourier_diffusion.models.diffusers_unet import DiffusersUNet
from fourier_diffusion.diffusion.forward_process import FourierDiffusionConfig, FourierForwardProcess
from fourier_diffusion.diffusion.sampling import ddim_sample


def save_png_batch(x: torch.Tensor, outdir: str, start_idx: int) -> int:
    os.makedirs(outdir, exist_ok=True)
    x = x.clamp(-1, 1)
    x = (x + 1.0) * 0.5
    x = (x * 255.0).round().byte().cpu()

    saved = 0
    for i in range(x.shape[0]):
        img = x[i]
        if img.shape[0] == 1:
            arr = img[0].numpy()
            pil = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            arr = img.permute(1, 2, 0).numpy()
            pil = Image.fromarray(arr, mode="RGB")
        pil.save(os.path.join(outdir, f"{start_idx + i:06d}.png"))
        saved += 1
    return saved


def build_model(model_name: str, dataset: str, in_ch: int, H: int, W: int, hf_repo_id: str = "") -> torch.nn.Module:
    if model_name == "diffusers_unet_hf":
        return DiffusersUNet(in_ch=in_ch, sample_size=H, base_channels=128, repo_id=hf_repo_id)
    if model_name == "diffusers_unet":
        return DiffusersUNet(in_ch=in_ch, sample_size=H, base_channels=128)
    if model_name == "simple_unet":
        return SimpleUNet(in_ch=in_ch, base=64, t_dim=128)

    if dataset == "cifar10":
        if hf_repo_id:
            return DiffusersUNet(in_ch=in_ch, sample_size=H, base_channels=128, repo_id=hf_repo_id)
        return DiffusersUNet(in_ch=in_ch, sample_size=H, base_channels=128)
    return SimpleUNet(in_ch=in_ch, base=64, t_dim=128)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--num", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_raw", action="store_true")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)

    dataset = ckpt.get("dataset", "cifar10")
    schedule = ckpt["schedule"]
    model_name = ckpt.get("model_name", "diffusers_unet" if dataset == "cifar10" else "simple_unet")
    hf_repo_id = ckpt.get("hf_repo_id", "")
    T = ckpt["T"]
    in_ch = ckpt["in_ch"]
    H = ckpt["H"]
    W = ckpt["W"]
    C_diag = ckpt.get("C_diag", None)
    if C_diag is not None:
        C_diag = C_diag.to(device)

    fwd = FourierForwardProcess(
        FourierDiffusionConfig(
            T=T,
            schedule=schedule,
            lam=ckpt.get("lam", 0.0),
            calibration=ckpt.get("calibration", "fixed_trace"),
            calibrate_alpha_bar=True,
        ),
        C_diag=C_diag,
        device=device,
    )

    model = build_model(model_name, dataset, in_ch, H, W, hf_repo_id).to(device)

    if args.use_raw:
        model.load_state_dict(ckpt["model_state"], strict=True)
        print("loaded raw model weights from checkpoint")
    elif "ema_state" in ckpt:
        shadow = ckpt["ema_state"]["shadow"]
        model_sd = model.state_dict()
        for k in model_sd:
            if k in shadow and shadow[k].shape == model_sd[k].shape:
                model_sd[k] = shadow[k].to(device=device, dtype=model_sd[k].dtype)
        model.load_state_dict(model_sd, strict=True)
        print("loaded EMA weights from checkpoint")
    else:
        model.load_state_dict(ckpt["model_state"], strict=True)
        print("loaded raw model weights from checkpoint")

    model.eval()
    os.makedirs(args.outdir, exist_ok=True)

    saved = 0
    while saved < args.num:
        bsz = min(args.batch_size, args.num - saved)
        with torch.no_grad():
            samp = ddim_sample(model, fwd, (bsz, in_ch, H, W), steps=args.steps).clamp(-1, 1)
        wrote = save_png_batch(samp, args.outdir, saved)
        saved += wrote
        print(f"saved {saved}/{args.num}")

    print(f"done: wrote {saved} images to {args.outdir}")


if __name__ == "__main__":
    main()