import torch
from .sparse_signals import rfft1


@torch.no_grad()
def exact_k_support_recovery_rate(x_hat: torch.Tensor, true_bins: torch.Tensor, k: int) -> float:
    """
    x_hat: (B,N)
    true_bins: (B,k)
    """
    y = rfft1(x_hat)
    mag = torch.sqrt(y.real ** 2 + y.imag ** 2)  # (B,Nf)
    pred = torch.topk(mag, k=k, dim=1).indices   # (B,k)

    pred_sorted = torch.sort(pred, dim=1).values
    true_sorted = torch.sort(true_bins, dim=1).values
    return float((pred_sorted == true_sorted).all(dim=1).float().mean().item())