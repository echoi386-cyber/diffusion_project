#!/usr/bin/env python3
import argparse
import os

import torch

from fourier_diffusion.utils.seed import get_device, seed_all
from fourier_diffusion.utils.plots import plot_radial_spectra
from fourier_diffusion.utils.fft import radial_power_spectrum

from fourier_diffusion.models.unet import SimpleUNet
from fourier_diffusion.diffusion.forward_process import FourierDiffusionConfig, FourierForwardProcess
from fourier_diffusion.diffusion.sampling import ddim_sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--steps_ddim", type=int, default=200)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--outdir", type=str, default="outputs/samples")
    p.add_argument("--seed", type=int, default=0)
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

    spec = radial_power_spectrum(samp)

    outpath = os.path.join(args.outdir, f"spectrum_{dataset}_{schedule}.png")
    plot_radial_spectra(
        {f"gen_{schedule}": spec},
        f"Generated spectrum: {dataset} {schedule}",
        outpath
    )

    # Save tensor samples if you want (optional)
    torch.save({"samples": samp.detach().cpu()}, os.path.join(args.outdir, f"samples_{dataset}_{schedule}.pt"))

    print("saved:", outpath)


if __name__ == "__main__":
    main()