"""Dataset helpers for white-box experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets, transforms


def load_one_sample(
    data_dir: str | Path,
    index: int = 0,
    train: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one MNIST image as (1, 1, 28, 28) float in [0, 1] and label tensor."""
    ds = datasets.MNIST(
        data_dir,
        train=train,
        download=False,
        transform=transforms.ToTensor(),
    )
    x, y = ds[index]
    return x.unsqueeze(0), torch.tensor([y])
