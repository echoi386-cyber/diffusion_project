import os
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def save_scatter(
    real: torch.Tensor,
    gen: torch.Tensor,
    path: str,
    title: str = "",
    subtitle: str = "",
    lim: float = 8.0,
    n_show: int = 20000,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    real = real[:n_show].detach().cpu()
    gen = gen[:n_show].detach().cpu()

    plt.figure(figsize=(6, 6))
    plt.scatter(real[:, 0], real[:, 1], s=2, alpha=0.35, label="real")
    plt.scatter(gen[:, 0], gen[:, 1], s=2, alpha=0.35, label="gen")
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    plt.gca().set_aspect("equal", "box")
    plt.legend(loc="upper right")

    if title:
        plt.title(title)
    if subtitle:
        plt.gcf().text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()