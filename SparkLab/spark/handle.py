"""
Module for data and torch tensor handling.
"""

from pathlib import Path
import random
from typing import Any, OrderedDict
import warnings

import torch
from torch.utils.data import Dataset


__all__ = [
    'gather_data_filepaths',
    'gather_data_labels',
    'save_dataset',
    'save_model',
    'load_dataset',
    'load_model',
]


def gather_data_filepaths(dirpath: str | Path, data_frmt: str = 'png', shuffle: bool = False) -> list[str]:
    """Groups all the data file-paths (of the given format) inside the specified directory."""
    dirpath_ = Path(dirpath)
    paths_list: list[str] = [str(path) for path in dirpath_.glob(f'*.{data_frmt}')]
    if shuffle: random.shuffle(paths_list)
    return paths_list


def gather_data_labels(
    data_paths: list[str | Path],
    class_lbls: dict[int, str],
) -> list[int]:
    """Gathers the data labels from given data filepaths."""
    lbls: list[int] = []
    
    for path in data_paths:
        file_name = Path(path).name
        match = next(
            (id_ for id_, cls_name in class_lbls.items() if cls_name in file_name), 
            None,
        )        
        if match is None:
            warnings.warn(f"Could not find a valid class name in: {file_name}")
        lbls.append(match)
        
    return lbls


def save_dataset(
    dataset: Dataset,
    save_to: str | Path,
    overwrite: bool = False,
    **kwargs,
) -> None:
    """Saves given dataset to '.pt' file."""
    if Path(save_to).exists() and not overwrite:
        print("Dataset already saved!")
        return
    print("Saving dataset...")
    torch.save(dataset, save_to, **kwargs)
    print("Dataset saved!")
    return


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


def load_dataset(filepath: str | Path, **kwargs) -> Dataset:
    """Load given dataset from '.pt' file."""
    print("Loading dataset...")
    dataset = torch.load(filepath, weights_only=False, **kwargs)
    print("Dataset loaded!")
    return dataset


def load_model(filepath: str | Path, **kwargs) -> dict[str, Any]:
    """Load given model, its state and saved info from '.pt' file."""
    print("Loading model...")
    model_state: dict = torch.load(filepath, weights_only=False, **kwargs)
    print("Model loaded!")
    return model_state


# end