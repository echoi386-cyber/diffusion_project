from typing import Tuple
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


def get_grayscale_mnist(
    batch_size: int,
    root: str = "./data",
    num_workers: int = 4,
) -> Tuple[torch.utils.data.Dataset, DataLoader]:
    tfm = T.Compose([
        T.ToTensor(),                          # [0,1], (1,28,28)
        T.Lambda(lambda x: x * 2.0 - 1.0),     # [-1,1], (1,28,28)
    ])
    ds = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return ds, dl