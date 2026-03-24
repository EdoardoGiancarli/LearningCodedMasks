"""
Module for data and torch tensor handling.
"""

from typing import Any, OrderedDict
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset


__all__ = [
    'get_data_filespaths',
    'save_dataset',
    'load_dataset',
    'save_model',
    'load_model',
]


def get_data_filespaths(dirpath: str | Path, data_frmt: str = 'png', shuffle: bool = False) -> list[str]:
    """Groups all the data file-paths (of the given format) inside the specified directory."""
    dirpath_ = Path(dirpath)
    paths_list: list[str] = [str(path) for path in dirpath_.glob(f'*.{data_frmt}')]
    if shuffle: random.shuffle(paths_list)
    return paths_list


def save_dataset(
    dataset: Dataset,
    save_to: str | Path,
    overwrite: bool = False,
    **kwargs,
) -> None:
    """
    Saves given dataset to '.pt' file.
    """
    if Path(save_to).exists() and not overwrite:
        print("Dataset already saved!")
        return
    print("Saving dataset...")
    torch.save(dataset, save_to, **kwargs)
    print("Dataset saved!")
    return


def load_dataset(filepath: str | Path, **kwargs) -> Dataset:
    """
    Load given dataset from '.pt' file.
    """
    print("Loading dataset...")
    dataset = torch.load(filepath, weights_only=False, **kwargs)
    print("Dataset loaded!")
    return dataset


def save_model(
    state_dict: OrderedDict,
    save_to: str | Path,
    info: dict[str, Any] | None = None,
    overwrite: bool = False,
    **kwargs,
) -> None:
    """Saves given model and its state to '.pt' file, plus other info."""
    if Path(save_to).exists() and not overwrite:
        print("Model already saved!")
        return
    print("Saving model...")
    data = info if info is not None else {}
    data['state_dict'] = state_dict
    torch.save(data, save_to, **kwargs)
    print("Model saved!")
    return


def load_model(filepath: str | Path, **kwargs) -> dict[str, Any]:
    """Load given model, its state and saved info from '.pt' file."""
    print("Loading model...")
    model_state: dict = torch.load(filepath, weights_only=False, **kwargs)
    print("Model loaded!")
    return model_state



# end