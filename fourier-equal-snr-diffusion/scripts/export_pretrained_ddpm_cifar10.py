#!/usr/bin/env python3
import argparse
import os

import torch
from diffusers import DDPMPipeline, DDIMScheduler
from PIL import Image


def save_image(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="google/ddpm-cifar10-32")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--num", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = args.device

    pipe = DDPMPipeline.from_pretrained(args.model_id)

    if args.sampler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    os.makedirs(args.outdir, exist_ok=True)

    saved = 0
    batch_idx = 0

    while saved < args.num:
        bsz = min(args.batch_size, args.num - saved)

        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + batch_idx)

        out = pipe(
            batch_size=bsz,
            generator=generator,
            num_inference_steps=args.steps,
            output_type="pil",
        )

        images = out.images
        for i, img in enumerate(images):
            save_image(img, os.path.join(args.outdir, f"{saved + i:06d}.png"))

        saved += bsz
        batch_idx += 1
        print(f"saved {saved}/{args.num}")

    print(f"done: wrote {saved} images to {args.outdir}")


if __name__ == "__main__":
    main()