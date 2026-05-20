#!/usr/bin/env python3
import argparse
import os
import shutil

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


def build_model(
    model_name: str,
    dataset: str,
    in_ch: int,
    H: int,
    W: int,
    hf_repo_id: str = "",
) -> torch.nn.Module:
    if model_name == "diffusers_unet_hf":
        if not hf_repo_id:
            raise ValueError("checkpoint has model_name=diffusers_unet_hf but empty hf_repo_id")

        return DiffusersUNet(
            in_ch=in_ch,
            sample_size=H,
            base_channels=128,
            repo_id=hf_repo_id,
        )

    if model_name == "diffusers_unet":
        return DiffusersUNet(
            in_ch=in_ch,
            sample_size=H,
            base_channels=128,
        )

    if model_name == "simple_unet":
        return SimpleUNet(
            in_ch=in_ch,
            base=64,
            t_dim=128,
        )

    raise ValueError(f"Unknown model_name in checkpoint: {model_name}")


@torch.no_grad()
def load_ema_strict(model: torch.nn.Module, ckpt: dict, device: torch.device) -> None:
    if "ema_state" not in ckpt or ckpt["ema_state"] is None:
        raise RuntimeError("Checkpoint has no ema_state.")

    ema_state = ckpt["ema_state"]

    if "shadow" not in ema_state or not isinstance(ema_state["shadow"], dict):
        raise RuntimeError("Checkpoint ema_state has no shadow dict.")

    shadow = ema_state["shadow"]
    buffers = ema_state.get("buffers", {})

    copied_params = 0
    total_params = 0
    missing_params = []
    mismatched_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        total_params += 1

        if name not in shadow:
            missing_params.append(name)
            continue

        src = shadow[name]

        if tuple(src.shape) != tuple(param.shape):
            mismatched_params.append((name, tuple(src.shape), tuple(param.shape)))
            continue

        param.data.copy_(src.to(device=device, dtype=param.dtype))
        copied_params += 1

    print(f"EMA parameter tensors copied: {copied_params}/{total_params}")

    if missing_params:
        print("missing EMA params first 20:", missing_params[:20])

    if mismatched_params:
        print("mismatched EMA params first 20:", mismatched_params[:20])

    if copied_params != total_params:
        raise RuntimeError("EMA parameter checkpoint/model mismatch. Refusing to export samples.")

    copied_buffers = 0
    total_buffers = 0
    missing_buffers = []
    mismatched_buffers = []

    for name, buf in model.named_buffers():
        total_buffers += 1

        if name not in buffers:
            missing_buffers.append(name)
            continue

        src = buffers[name]

        if tuple(src.shape) != tuple(buf.shape):
            mismatched_buffers.append((name, tuple(src.shape), tuple(buf.shape)))
            continue

        buf.data.copy_(src.to(device=device, dtype=buf.dtype))
        copied_buffers += 1

    print(f"EMA buffer tensors copied: {copied_buffers}/{total_buffers}")

    if total_buffers > 0 and copied_buffers != total_buffers:
        if missing_buffers:
            print("missing EMA buffers first 20:", missing_buffers[:20])

        if mismatched_buffers:
            print("mismatched EMA buffers first 20:", mismatched_buffers[:20])

        raise RuntimeError("EMA buffer checkpoint/model mismatch. Refusing to export samples.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--num", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_raw", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--force_model_name", type=str, default="")
    p.add_argument("--force_hf_repo_id", type=str, default="")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)

    dataset = ckpt.get("dataset", "cifar10")
    schedule = ckpt["schedule"]

    model_name = args.force_model_name or ckpt.get(
        "model_name",
        "diffusers_unet" if dataset == "cifar10" else "simple_unet",
    )

    hf_repo_id = args.force_hf_repo_id or ckpt.get("hf_repo_id", "")

    T = ckpt["T"]
    in_ch = ckpt["in_ch"]
    H = ckpt["H"]
    W = ckpt["W"]

    C_diag = ckpt.get("C_diag", None)
    if C_diag is not None:
        C_diag = C_diag.to(device)

    print("checkpoint:", args.ckpt)
    print("dataset:", dataset)
    print("schedule:", schedule)
    print("model_name used:", model_name)
    print("hf_repo_id used:", hf_repo_id)
    print("lam:", ckpt.get("lam", 0.0))
    print("calibration:", ckpt.get("calibration", "fixed_trace"))
    print("calibrate_alpha_bar:", ckpt.get("calibrate_alpha_bar", True))
    print("T:", T, "in_ch:", in_ch, "H:", H, "W:", W)
    print("use_raw:", args.use_raw)

    if C_diag is None:
        print("C_diag: None")
    else:
        print(
            "C_diag shape/min/mean/max:",
            tuple(C_diag.shape),
            float(C_diag.min()),
            float(C_diag.mean()),
            float(C_diag.max()),
        )

    fwd = FourierForwardProcess(
        FourierDiffusionConfig(
            T=T,
            schedule=schedule,
            lam=ckpt.get("lam", 0.0),
            calibration=ckpt.get("calibration", "fixed_trace"),
            calibrate_alpha_bar=ckpt.get("calibrate_alpha_bar", True),
        ),
        C_diag=C_diag,
        device=device,
    )

    model = build_model(model_name, dataset, in_ch, H, W, hf_repo_id).to(device)

    if args.use_raw:
        model.load_state_dict(ckpt["model_state"], strict=True)
        print("loaded raw model weights from checkpoint")
    elif "ema_state" in ckpt and ckpt["ema_state"] is not None:
        load_ema_strict(model, ckpt, device)
        print("loaded EMA weights from checkpoint")
    else:
        model.load_state_dict(ckpt["model_state"], strict=True)
        print("loaded raw model weights from checkpoint")

    model.eval()

    if args.overwrite and os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)

    os.makedirs(args.outdir, exist_ok=True)

    saved = 0

    while saved < args.num:
        bsz = min(args.batch_size, args.num - saved)

        with torch.no_grad():
            samp = ddim_sample(model, fwd, (bsz, in_ch, H, W), steps=args.steps).clamp(-1, 1)

        wrote = save_png_batch(samp, args.outdir, saved)
        saved += wrote
        print(f"saved {saved}/{args.num}", flush=True)

    print(f"done: wrote {saved} images to {args.outdir}")


if __name__ == "__main__":
    main()