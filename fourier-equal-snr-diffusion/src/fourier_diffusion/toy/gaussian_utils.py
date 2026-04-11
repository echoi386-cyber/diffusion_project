import math
import torch


def orthonormal_matrix(d: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    A = torch.randn(d, d, device=device, generator=g)
    Q, R = torch.linalg.qr(A)
    diag = torch.sign(torch.diag(R))
    diag[diag == 0] = 1.0
    Q = Q * diag
    return Q


def make_spectrum(d: int, p: float, scale: float, device: torch.device) -> torch.Tensor:
    i = torch.arange(1, d + 1, device=device, dtype=torch.float32)
    return (scale ** 2) * (i ** (-p))


@torch.no_grad()
def sample_diag_gaussian(n: int, C: torch.Tensor) -> torch.Tensor:
    eps = torch.randn(n, C.numel(), device=C.device)
    return eps * torch.sqrt(C).view(1, -1)


def w2_diag(C: torch.Tensor, Chat: torch.Tensor) -> float:
    val = torch.sum((torch.sqrt(C) - torch.sqrt(Chat.clamp_min(1e-12))) ** 2)
    return float(torch.sqrt(val).detach().cpu())


def kl_diag(C: torch.Tensor, Chat: torch.Tensor) -> float:
    Chat = Chat.clamp_min(1e-12)
    ratio = C / Chat
    val = 0.5 * torch.sum(ratio - 1.0 - torch.log(ratio.clamp_min(1e-12)))
    return float(val.detach().cpu())


@torch.no_grad()
def sliced_wasserstein_1(x: torch.Tensor, y: torch.Tensor, n_proj: int = 256) -> float:
    device = x.device
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]

    dirs = torch.randn(n_proj, x.shape[1], device=device)
    dirs = dirs / torch.norm(dirs, dim=1, keepdim=True).clamp_min(1e-12)

    x_proj = x @ dirs.t()
    y_proj = y @ dirs.t()

    x_sort, _ = torch.sort(x_proj, dim=0)
    y_sort, _ = torch.sort(y_proj, dim=0)

    return float(torch.mean(torch.abs(x_sort - y_sort)).detach().cpu())