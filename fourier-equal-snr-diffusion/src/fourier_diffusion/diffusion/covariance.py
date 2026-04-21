import math
from typing import Literal

import torch

from ..utils.fft import rfft2


def estimate_C_diag_rfft2(
    loader,
    device: torch.device,
    n_batches: int = 200,
    clamp_min: float = 1e-8,
) -> torch.Tensor:
    """
    Estimate C_i = Var[(y0)_i] in Fourier domain using rFFT2.
    Returns C_diag with shape (C,H,W//2+1).
    """
    sum_re = None
    sum_im = None
    sum_re2 = None
    sum_im2 = None
    n = 0

    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break

        x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
        x0 = x0.to(device, non_blocking=True).float()
        y0 = rfft2(x0)  # (B,C,H,Wf)

        re = y0.real
        im = y0.imag

        if sum_re is None:
            sum_re = re.sum(dim=0)
            sum_im = im.sum(dim=0)
            sum_re2 = (re ** 2).sum(dim=0)
            sum_im2 = (im ** 2).sum(dim=0)
        else:
            sum_re += re.sum(dim=0)
            sum_im += im.sum(dim=0)
            sum_re2 += (re ** 2).sum(dim=0)
            sum_im2 += (im ** 2).sum(dim=0)

        n += re.shape[0]

    mean_re = sum_re / max(n, 1)
    mean_im = sum_im / max(n, 1)
    var_re = (sum_re2 / max(n, 1)) - mean_re ** 2
    var_im = (sum_im2 / max(n, 1)) - mean_im ** 2

    C_diag = (var_re + var_im).clamp_min(clamp_min)
    return C_diag


def make_sigma_diag(
    scheme: Literal["ddpm", "equal_snr", "flipped_snr", "power_law"],
    C_diag: torch.Tensor,
    lam: float = 0.0,
    calibration: str = "fixed_trace",
) -> torch.Tensor:
    """
    Return Sigma_ii for forward noise eps ~ CN(0, Sigma) with diagonal Sigma.

    - DDPM:       Sigma = I
    - EqualSNR:   Sigma = C
    - FlippedSNR: radial flip construction
    - power_law:  Sigma_i ∝ C_i^lam, then calibrated
    """
    if scheme == "ddpm":
        return torch.ones_like(C_diag)

    if scheme == "equal_snr":
        return C_diag.clone().clamp_min(1e-8)

    if scheme == "flipped_snr":
        # Radial flip of the average spectrum
        C, H, Wf = C_diag.shape
        device = C_diag.device

        yy = torch.arange(H, device=device)
        xx = torch.arange(Wf, device=device)

        u = torch.minimum(yy, H - yy).float().view(H, 1)
        v = xx.float().view(1, Wf)
        r = torch.sqrt(u ** 2 + v ** 2)
        r_max = r.max()
        r_flip = (r_max - r).clamp_min(0.0)

        nbins = int(max(H, Wf))
        bins = torch.linspace(0.0, float(r_max), nbins, device=device)

        idx = torch.bucketize(r.flatten(), bins) - 1
        idx = idx.clamp(0, nbins - 1).view(H, Wf)

        idx_flip = torch.bucketize(r_flip.flatten(), bins) - 1
        idx_flip = idx_flip.clamp(0, nbins - 1).view(H, Wf)

        Cbin = torch.zeros(C, nbins, device=device)
        counts = torch.zeros(nbins, device=device)

        for b in range(nbins):
            mask = idx == b
            counts[b] = mask.sum().float().clamp_min(1.0)
            Cbin[:, b] = C_diag[:, mask].sum(dim=1) / counts[b]

        C_flip = Cbin[:, idx_flip]  # (C,H,Wf)
        Sigma = (C_diag / C_flip.clamp_min(1e-8)).clamp_min(1e-8)
        return Sigma

    if scheme == "power_law":
        Sigma = C_diag.clamp_min(1e-8).pow(lam)

        if calibration == "fixed_trace":
            # Match total trace to DDPM baseline, where Sigma = 1 everywhere.
            target_trace = float(C_diag.numel())
            Sigma = Sigma * (target_trace / Sigma.sum().clamp_min(1e-8))
            return Sigma.clamp_min(1e-8)

        raise ValueError(f"Unsupported calibration for power_law: {calibration}")

    raise ValueError(f"Unknown scheme: {scheme}")