import torch


def rfft2(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")


def irfft2(y: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return torch.fft.irfft2(y, s=(h, w), dim=(-2, -1), norm="ortho")


def rfft2_onesided_weights(
    height: int,
    width: int,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Multiplicity weights for a real-valued 2D rFFT stored as shape (H, W//2+1).

    Interior frequency columns represent both +kx and -kx and therefore get weight 2.
    DC and Nyquist columns (when width is even) get weight 1.
    """
    wf = width // 2 + 1
    w = torch.ones((height, wf), device=device, dtype=dtype)

    if width % 2 == 0:
        if wf > 2:
            w[:, 1:-1] = 2.0
    else:
        if wf > 1:
            w[:, 1:] = 2.0

    return w


def complex_var(y: torch.Tensor, dim: int = 0, eps: float = 1e-12) -> torch.Tensor:
    re = y.real
    im = y.imag
    return re.var(dim=dim, unbiased=False) + im.var(dim=dim, unbiased=False) + eps


def radial_power_spectrum(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    x: (B,C,H,W)
    Returns radial average of |FFT|^2 averaged over batch and channels,
    corrected for one-sided rFFT column multiplicity.
    """
    device = x.device
    _, _, H, W = x.shape
    y = rfft2(x)

    power = (y.real.square() + y.imag.square()).mean(dim=(0, 1))
    Wf = power.shape[-1]

    mult = rfft2_onesided_weights(H, W, device=device, dtype=power.dtype)

    yy = torch.arange(H, device=device)
    xx = torch.arange(Wf, device=device)
    u = torch.minimum(yy, H - yy).float().view(H, 1)
    v = xx.float().view(1, Wf)
    r = torch.sqrt(u ** 2 + v ** 2)
    r_max = r.max()

    nbins = int(max(H, Wf))
    bins = torch.linspace(0.0, r_max + eps, nbins + 1, device=device)

    idx = torch.bucketize(r.flatten(), bins) - 1
    idx = idx.clamp(0, nbins - 1)

    spec = torch.zeros(nbins, device=device, dtype=power.dtype)
    counts = torch.zeros(nbins, device=device, dtype=power.dtype)

    pflat = power.flatten()
    wflat = mult.flatten()

    for b in range(nbins):
        m = idx == b
        counts[b] = wflat[m].sum().clamp_min(1.0)
        spec[b] = (pflat[m] * wflat[m]).sum() / counts[b]

    return spec