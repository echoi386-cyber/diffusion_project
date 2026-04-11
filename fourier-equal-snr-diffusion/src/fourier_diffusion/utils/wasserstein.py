import torch


@torch.no_grad()
def sliced_wasserstein_1(x: torch.Tensor, y: torch.Tensor, n_proj: int = 256) -> torch.Tensor:
    """
    Sliced Wasserstein-1 in R^2 (approximation):
    average 1D W1 over random directions.
    """
    device = x.device
    N = min(x.shape[0], y.shape[0])
    x = x[:N]
    y = y[:N]

    dirs = torch.randn(n_proj, 2, device=device)
    dirs = dirs / torch.norm(dirs, dim=1, keepdim=True).clamp_min(1e-12)

    x_proj = x @ dirs.t()
    y_proj = y @ dirs.t()

    x_sort, _ = torch.sort(x_proj, dim=0)
    y_sort, _ = torch.sort(y_proj, dim=0)

    return torch.mean(torch.abs(x_sort - y_sort))


@torch.no_grad()
def sinkhorn_w2_approx(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.05,
    iters: int = 200,
    max_points: int = 2048,
) -> torch.Tensor:
    """
    Entropic OT (Sinkhorn) approximation of W2.
    Returns sqrt(<pi, ||x-y||^2>) as approximate W2.
    """
    device = x.device
    n = min(x.shape[0], y.shape[0], max_points)
    x = x[:n]
    y = y[:n]

    a = torch.full((n,), 1.0 / n, device=device)
    b = torch.full((n,), 1.0 / n, device=device)

    x2 = (x ** 2).sum(dim=1, keepdim=True)
    y2 = (y ** 2).sum(dim=1, keepdim=True).t()
    C = (x2 + y2 - 2.0 * (x @ y.t())).clamp_min(0.0)

    K = torch.exp(-C / eps).clamp_min(1e-12)

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(iters):
        u = a / (K @ v).clamp_min(1e-12)
        v = b / (K.t() @ u).clamp_min(1e-12)

    pi = (u.view(-1, 1) * K) * v.view(1, -1)
    w2_sq = torch.sum(pi * C)
    return torch.sqrt(w2_sq.clamp_min(0.0))