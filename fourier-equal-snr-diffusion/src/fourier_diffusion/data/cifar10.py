import typing
import torch
from typing import Tuple
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


def get_cifar10(batch_size: int, root: str = "./data", num_workers: int = 4) -> Tuple[torch.utils.data.Dataset, DataLoader]:
    tfm = T.Compose([
        T.ToTensor(),  # [0,1], (3,32,32)
        T.Lambda(lambda x: x * 2.0 - 1.0),  # [-1,1]
    ])
    ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return ds, dl