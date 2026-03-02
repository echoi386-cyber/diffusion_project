import torch

def pca_from_data(X: torch.Tensor):
    """
    X: (N, D) zero-mean preferred (we'll center anyway)
    Returns:
      U: (D, D) eigenvectors (columns) sorted by descending eigenvalue
      C: (D,) eigenvalues (descending)
      mu: (D,)
    """
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    # covariance
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)  # ascending
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evecs, evals.clamp_min(1e-12), mu.squeeze(0)