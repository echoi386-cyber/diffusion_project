import torch

@torch.no_grad()
def make_rotated_gaussian_dataset(n: int, d: int, C_true: torch.Tensor, Q: torch.Tensor, device):
    """
    y0 ~ N(0, diag(C_true)), x0 = y0 Q^T
    Returns X0: (n, d)
    """
    y = torch.randn(n, d, device=device) * torch.sqrt(C_true).view(1, d)
    x = y @ Q.T
    return x