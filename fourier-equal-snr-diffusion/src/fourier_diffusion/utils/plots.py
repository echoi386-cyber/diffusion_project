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

def plot_curve_multi(x, ys, labels, title, xlabel, ylabel, outpath, logy=False):
    plt.figure(figsize=(7, 4))
    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_curve(x, y_true, y_emp, title, xlabel, ylabel, outpath, logy=False):
    plt.figure(figsize=(7, 4))
    plt.plot(x, y_true, label="theory")
    plt.plot(x, y_emp, label="empirical")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_single(x, y, title, xlabel, ylabel, outpath, logy=False):
    plt.figure(figsize=(7, 4))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_scatter_2d(real: torch.Tensor, gen: torch.Tensor, outpath: str, title: str):
    real = real.detach().cpu()
    gen = gen.detach().cpu()
    plt.figure(figsize=(6, 6))
    plt.scatter(real[:, 0], real[:, 1], s=2, alpha=0.35, label="real")
    plt.scatter(gen[:, 0], gen[:, 1], s=2, alpha=0.35, label="gen")
    plt.legend()
    plt.title(title)
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()