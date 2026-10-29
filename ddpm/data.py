"""MNIST images and labels for the pixel-space diffusion exercises."""

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def make_mnist_loader(
    data_dir: str | Path = "data",
    *,
    batch_size: int = 64,
    train: bool = True,
    shuffle: bool | None = None,
    seed: int = 0,
    download: bool = True,
) -> DataLoader:
    """Yield CPU images [B, 1, 28, 28] in [-1, 1] and integer labels [B]."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    transform = transforms.Compose(
        [
            # uint8 [0, 255], [H, W] -> float32 [0, 1], [1, H, W].
            transforms.ToTensor(),
            # (x - 0.5) / 0.5 = 2*x - 1, so [0, 1] -> [-1, 1].
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    dataset = datasets.MNIST(root=data_dir, train=train, transform=transform, download=download)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train if shuffle is None else shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=False,
    )


def denormalize(images: Tensor) -> Tensor:
    """Map clean MNIST tensors from [-1, 1] back to [0, 1] for display."""
    return (images + 1) / 2
