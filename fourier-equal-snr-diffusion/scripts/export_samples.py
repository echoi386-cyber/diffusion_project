#!/usr/bin/env python3
import argparse
import math
import os

import torch
from PIL import Image

from fourier_diffusion.models.diffusers_unet import DiffusersUNet
from fourier_diffusion.diffusion.forward_process import FourierDiffusionConfig, FourierForwardProcess
from fourier_diffusion.diffusion.sampling import ddim_sample


@torch.no_grad()
def save_png(x: torch.Tensor, path: str):
    x = x.clamp(-1, 1)
    x = (x + 1.0) * 0.5
    x = (x * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(x).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--num", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    run_args = ckpt.get("args", {})

    H = ckpt["H"]
    W = ckpt["W"]
    in_ch = ckpt["in_ch"]

    model = DiffusersUNet(in_ch=in_ch, sample_size=H, base_channels=128).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    C_diag = ckpt.get("C_diag", None)
    if C_diag is not None:
        C_diag = C_diag.to(device)

    fwd = FourierForwardProcess(
        FourierDiffusionConfig(
            T=ckpt["T"],
            schedule=run_args.get("schedule", ckpt.get("schedule", "ddpm")),
            lam=float(run_args.get("lam", ckpt.get("lam", 0.0))),
            calibration=run_args.get("calibration", ckpt.get("calibration", "fixed_trace")),
            calibrate_alpha_bar=True,
        ),
        C_diag=C_diag,
        device=device,
    )

    n_done = 0
    while n_done < args.num:
        bsz = min(args.batch_size, args.num - n_done)
        samp = ddim_sample(model, fwd, (bsz, in_ch, H, W), steps=args.steps).clamp(-1, 1)

        for i in range(bsz):
            save_png(samp[i], os.path.join(args.outdir, f"{n_done + i:06d}.png"))

        n_done += bsz
        print(f"saved {n_done}/{args.num}")


if __name__ == "__main__":
    main()