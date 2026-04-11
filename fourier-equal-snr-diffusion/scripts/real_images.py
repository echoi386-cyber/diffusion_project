#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image


def save_folder(dataset_name: str, outdir: str, n: int):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if dataset_name == "cifar10":
        tfm = T.Compose([T.ToTensor()])
        ds = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
    elif dataset_name == "mnist":
        # convert to RGB so Inception sees 3 channels
        tfm = T.Compose([T.ToTensor()])
        ds = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    else:
        raise ValueError(dataset_name)

    n = min(n, len(ds))
    for i in range(n):
        x, _ = ds[i]  # x in [0,1], shape CxHxW
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        x_u8 = (x.clamp(0, 1) * 255.0).round().to(torch.uint8)
        arr = x_u8.permute(1, 2, 0).cpu().numpy()
        Image.fromarray(arr, mode="RGB").save(os.path.join(outdir, f"{i:06d}.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "mnist"], required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--n", type=int, default=50000)
    args = p.parse_args()

    save_folder(args.dataset, args.outdir, args.n)
    print("saved real images:", args.outdir)


if __name__ == "__main__":
    main()