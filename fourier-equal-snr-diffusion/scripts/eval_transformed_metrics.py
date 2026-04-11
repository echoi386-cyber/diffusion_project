#!/usr/bin/env python3
import argparse
import glob
import math
import os
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d, conv2d
from pytorch_fid.inception import InceptionV3


IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "bmp", "webp")


def load_images(path: str, limit: int = 50000) -> torch.Tensor:
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(path, f"**/*.{ext}"), recursive=True))
    files = sorted(files)[:limit]

    xs = []
    for fp in files:
        img = Image.open(fp).convert("RGB")
        x = TF.to_tensor(img) * 2.0 - 1.0
        xs.append(x)
    return torch.stack(xs, dim=0)


def gaussian_kernel(size: int = 9, sigma: float = 2.0, device: str = "cpu") -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    k2 = torch.outer(g, g)
    return k2


def apply_blur(x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    device = x.device
    k = gaussian_kernel(size=9, sigma=sigma, device=device)
    k = k.view(1, 1, 9, 9).repeat(x.shape[1], 1, 1, 1)
    return conv2d(x, k, padding=4, groups=x.shape[1])


def apply_edge(x: torch.Tensor) -> torch.Tensor:
    device = x.device
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device)
    sobel_y = sobel_x.t()

    kx = sobel_x.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    ky = sobel_y.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)

    gx = conv2d(x, kx, padding=1, groups=x.shape[1])
    gy = conv2d(x, ky, padding=1, groups=x.shape[1])
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def transform_batch(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "identity":
        return x
    if mode == "blur":
        return apply_blur(x)
    if mode == "edge":
        return apply_edge(x)
    raise ValueError(mode)


@torch.no_grad()
def get_activations(x: torch.Tensor, model, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size].to(device)
        xb = (xb + 1.0) * 0.5  # inception expects [0,1]
        pred = model(xb)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        out.append(pred)
    return np.concatenate(out, axis=0)


def calc_stats(x: torch.Tensor, batch_size: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    act = get_activations(x, model, batch_size, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def frechet(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def fft_var(x: torch.Tensor) -> torch.Tensor:
    y = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    re_var = y.real.var(dim=0, unbiased=False)
    im_var = y.imag.var(dim=0, unbiased=False)
    return re_var + im_var


def make_weight_grid(H: int, W: int, mode: str, device: torch.device) -> torch.Tensor:
    Wf = W // 2 + 1

    if mode == "identity":
        return torch.ones(1, H, Wf, device=device)

    yy = torch.arange(H, device=device)
    xx = torch.arange(Wf, device=device)
    u = torch.minimum(yy, H - yy).float().view(H, 1)
    v = xx.float().view(1, Wf)
    r2 = u * u + v * v

    if mode == "edge":
        return r2.unsqueeze(0)

    if mode == "blur":
        sigma = 2.0
        # Approximate |G_hat|^2 for an isotropic Gaussian blur
        return torch.exp(-sigma * sigma * r2 / max(H * W, 1)).unsqueeze(0)

    raise ValueError(mode)


def spectral_metric(real_x: torch.Tensor, gen_x: torch.Tensor, mode: str) -> float:
    device = real_x.device
    _, C, H, W = real_x.shape

    vr = fft_var(real_x)
    vg = fft_var(gen_x)
    w = make_weight_grid(H, W, mode, device=device)
    val = (w * (vg - vr).abs()).sum().item()
    return float(val)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", type=str, required=True)
    p.add_argument("--gen_dir", type=str, required=True)
    p.add_argument("--limit", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    real_x = load_images(args.real_dir, args.limit)
    gen_x = load_images(args.gen_dir, args.limit)

    n = min(real_x.shape[0], gen_x.shape[0])
    real_x = real_x[:n]
    gen_x = gen_x[:n]

    results = {}
    for mode in ["identity", "blur", "edge"]:
        real_t = transform_batch(real_x.to(device), mode).cpu()
        gen_t = transform_batch(gen_x.to(device), mode).cpu()

        mu_r, sig_r = calc_stats(real_t, args.batch_size, device)
        mu_g, sig_g = calc_stats(gen_t, args.batch_size, device)
        fid_val = frechet(mu_r, sig_r, mu_g, sig_g)

        dw = spectral_metric(real_x.to(device), gen_x.to(device), mode)

        results[mode] = {
            "fid": fid_val,
            "Dw": dw,
        }

    for mode, vals in results.items():
        print(f"{mode}: FID={vals['fid']:.6f}  Dw={vals['Dw']:.6f}")


if __name__ == "__main__":
    main()