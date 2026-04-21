#!/usr/bin/env python3
import argparse
import os

from PIL import Image
from torchvision.datasets import CIFAR10


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--outdir", type=str, default="data/cifar10_test_png")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ds = CIFAR10(root=args.root, train=False, download=True)

    for i, (img, label) in enumerate(ds):
        # img is already a PIL image
        fname = os.path.join(args.outdir, f"{i:05d}_label{label}.png")
        img.save(fname)

    print(f"saved {len(ds)} test images to {args.outdir}")


if __name__ == "__main__":
    main()