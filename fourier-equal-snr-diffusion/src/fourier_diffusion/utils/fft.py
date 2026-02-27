import torch
#Fourier transform utils for real-valued images: rfft2 and irfft2 + complex variance
def rfft2(x: torch.Tensor) -> torch.Tensor:
    # x: (B,C,H,W) real
    return torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")

def irfft2(y: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # y: (B,C,H,W//2+1) complex
    return torch.fft.irfft2(y, s=(h, w), dim=(-2, -1), norm="ortho")

def complex_var(y: torch.Tensor, dim: int = 0, eps: float = 1e-12) -> torch.Tensor:
    # Var(Re)+Var(Im) along dim
    re = y.real
    im = y.imag
    return re.var(dim=dim, unbiased=False) + im.var(dim=dim, unbiased=False) + eps
# Frequency analysis and plots
def radial_power_spectrum(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    x: (B,C,H,W) in [-1,1] or [0,1]
    Returns radial average of |FFT|^2 averaged over batch and channels.
    """
    device = x.device
    B, C, H, W = x.shape
    y = rfft2(x)  # (B,C,H,Wf)
    power = (y.real ** 2 + y.imag ** 2).mean(dim=(0, 1))  # (H,Wf)
    Wf = power.shape[-1]
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
    spec = torch.zeros(nbins, device=device)
    counts = torch.zeros(nbins, device=device)
    pflat = power.flatten()
    for b in range(nbins):
        m = (idx == b)
        counts[b] = m.sum().float().clamp_min(1.0)
        spec[b] = pflat[m].sum() / counts[b]
    return spec