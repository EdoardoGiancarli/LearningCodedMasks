"""
Module with useful func and possible `spark` material.
"""

from functools import wraps
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.types import Tensor
from torch.utils.data import DataLoader


# _________________________________  ROUTINE TESTS  _________________________________ #

def test_model_works(
    model: nn.Module,
    data_shape: tuple[int, int, int],
    batches: int = 5,
) -> list[Tensor]:
    """Tests if model is working properly when applied to mock dataset."""
    dataset = torch.rand((batches, *data_shape))
    out: list[Tensor] = []
    for batch in dataset:
        out.append(model(batch.unsqueeze(0)))
    print("It's alive!!!")
    return out


def extract_batches(dataloader: DataLoader) -> list[Tensor]:
    """Extracts the batches from given dataloader."""
    batches: list = []
    for batch in dataloader:
        batches.append(batch)
    return batches


# ___________________ TESTING OPs ___________________ #
from spark.inspect import errors_handler

def test_hook_works(container: list, hook_fn: Callable, n: int = 5) -> None:
    for idx in range(n):
        t = torch.rand((10, 10))
        hook_fn(t)
        print(
            f'At step {idx + 1}, {len(container)=}\n'
            f'Appended correctly: {torch.allclose(t, container[-1])}'
        )
    return


@errors_handler
def handler_test(raise_err: bool) -> None:
    if raise_err:
        raise ValueError('testing decorator')
    print('executing...')
    return 1


# end