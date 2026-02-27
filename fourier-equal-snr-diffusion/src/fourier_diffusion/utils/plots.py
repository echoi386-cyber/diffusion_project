import os
from typing import Dict
import matplotlib.pyplot as plt
import torch

# Plotting
def plot_loss_curves(loss_hist: Dict[str, list], title: str, save_path: str):
    plt.figure()
    for k, v in loss_hist.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.xlabel("log step")
    plt.ylabel("loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_radial_spectra(specs: Dict[str, torch.Tensor], title: str, save_path: str):
    plt.figure()
    for k, v in specs.items():
        plt.plot(v.detach().cpu().numpy(), label=k)
    plt.yscale("log")
    plt.xlabel("radial frequency bin")
    plt.ylabel("power")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()