"""
Module for data and torch tensor processing.
"""

from typing import Callable
from pathlib import Path

from tqdm import tqdm
import torch
from torch.types import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as ts
from torchvision.transforms import Compose


__all__ = [
    'center_tensor',
    'normalise',
    'process_data',
    'get_dataloaders',
]


def center_tensor(tensor: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Centers given tensor of shape `[C, H, W]` per-channel
    by subtracting the mean and dividing by the std.
    """
    dims = (1, 2)
    tensor_ = tensor.to(torch.float)
    mu = tensor_.mean(dims, keepdim=True)
    std = tensor_.std(dims, keepdim=True)
    centered = (tensor_ - mu) / (std + eps)
    return centered


def normalise(tensor: Tensor, norm_range: str = 'unilateral') -> Tensor:
    """Normalises given tensor in the range [0, 1] or [-1, 1]."""
    if norm_range not in ['unilateral', 'bilateral']:
        raise ValueError(f"Invalid 'norm_range' {norm_range}.")
    
    normalised = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    if norm_range == 'bilateral':
        normalised = 2 * normalised - 1.0

    return normalised


def process_data(
    data_paths: list[str | Path],
    open_with: Callable,
    transform: Compose | None,
) -> list[Tensor]:
    """
    Processes the given data by first opening the file and then applying a list of PyTorch
    `Transform` objs or custom-made Callable objs (if `None`, default is `PILToTensor`).
    """
    transform_ = transform if transform is not None else ts.PILToTensor()
    processed = [transform_(open_with(f)) for f in tqdm(data_paths, desc='Processing data')]
    return processed


def get_dataloaders(
    dataset: Dataset,
    batch_size: int,
    valid_size: float = 0.0,
    **kwargs,
) -> tuple[DataLoader, DataLoader | None]:
    """Training and Validation dataset loaders generation."""
    if valid_size and not (0.0 <= valid_size < 1.0):
        raise ValueError(f"Invalid 'valid_size' value {valid_size}, must be in [0, 1).")
    
    print('Baking DataLoaders...')    
    if valid_size > 0:
        vlen = int(valid_size * len(dataset))
        tlen = len(dataset) - vlen
        train, valid = random_split(dataset, [tlen, vlen])
        train_dl = DataLoader(train, batch_size, shuffle=True, **kwargs)
        valid_dl = DataLoader(valid, batch_size, shuffle=False, **kwargs)
    else:
        train_dl = DataLoader(dataset, batch_size, shuffle=True, **kwargs)
        valid_dl = None
    print('DataLoaders ready-to-go!')

    return train_dl, valid_dl


# end