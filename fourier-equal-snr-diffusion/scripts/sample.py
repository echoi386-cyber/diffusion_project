#!/usr/bin/env python3
# scripts/sample.py
import argparse
import os
from pathlib import Path

import torch
import torchvision.utils as vutils
from PIL import Image

from fourier_diffusion.utils.seed import get_device, seed_all
from fourier_diffusion.utils.plots import plot_radial_spectra
from fourier_diffusion.utils.fft import radial_power_spectrum

from fourier_diffusion.models.unet import SimpleUNet
from fourier_diffusion.diffusion.forward_process import FourierDiffusionConfig, FourierForwardProcess
from fourier_diffusion.diffusion.sampling import ddim_sample


def to_uint8_rgb(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,C,H,W) in [-1,1]
    returns uint8 tensor (B,3,H,W) in [0,255]
    """
    x = x.clamp(-1, 1)
    x = (x + 1.0) * 0.5  # [0,1]
    x = (x * 255.0).round().to(torch.uint8)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


def save_png_folder(img_u8: torch.Tensor, outdir: str):
    """
    img_u8: uint8 (B,3,H,W)
    saves 000000.png ... into outdir
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    b = img_u8.shape[0]
    for i in range(b):
        arr = img_u8[i].permute(1, 2, 0).contiguous().cpu().numpy()  # HWC
        Image.fromarray(arr, mode="RGB").save(os.path.join(outdir, f"{i:06d}.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--steps_ddim", type=int, default=200)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--outdir", type=str, default="outputs/samples")
    p.add_argument("--seed", type=int, default=0)

    # new
    p.add_argument("--save_png", action="store_true")
    p.add_argument("--png_dir", type=str, default="")
    p.add_argument("--grid_png", type=str, default="")  # optional grid image
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()
    seed_all(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    dataset = ckpt["dataset"]
    schedule = ckpt["schedule"]
    T = ckpt["T"]
    in_ch, H, W = ckpt["in_ch"], ckpt["H"], ckpt["W"]
    C_diag = ckpt["C_diag"]

    model = SimpleUNet(in_ch=in_ch, base=64, t_dim=128).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    fwd = FourierForwardProcess(
        FourierDiffusionConfig(T=T, schedule=schedule, calibrate_alpha_bar=True),
        C_diag=C_diag.to(device) if C_diag is not None else None,
        device=device
    )

    with torch.no_grad():
        samp = ddim_sample(model, fwd, (args.n, in_ch, H, W), steps=args.steps_ddim).clamp(-1, 1)

    # spectrum plot (existing behavior)
    spec = radial_power_spectrum(samp)
    outpath = os.path.join(args.outdir, f"spectrum_{dataset}_{schedule}.png")
    plot_radial_spectra({f"gen_{schedule}": spec}, f"Generated spectrum: {dataset} {schedule}", outpath)

    # optional tensor save (existing)
    torch.save({"samples": samp.detach().cpu()}, os.path.join(args.outdir, f"samples_{dataset}_{schedule}.pt"))

    # NEW: save images as PNGs
    if args.save_png:
        png_dir = args.png_dir.strip()
        if png_dir == "":
            png_dir = os.path.join(args.outdir, f"png_{dataset}_{schedule}")
        img_u8 = to_uint8_rgb(samp)
        save_png_folder(img_u8, png_dir)

        if args.grid_png.strip() != "":
            grid = vutils.make_grid(img_u8.float() / 255.0, nrow=8, padding=2)
            grid_arr = (grid.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
            Image.fromarray(grid_arr, mode="RGB").save(args.grid_png)

        print("saved pngs:", png_dir)

    print("saved:", outpath)


if __name__ == "__main__":
    main()