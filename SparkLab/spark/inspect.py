"""
Module for pipeline inspection tools.
"""

from functools import wraps
from contextlib import contextmanager
from typing import Any, Callable, Generator

import torch
from torch.types import Tensor
import torch.nn as nn


__all__ = [
    'OutputManager',
    'errors_handler',
    'forward_data_capture',
]


class OutputManager:
    """
    Manager for a model output data object. This class acts as a manager for data,
    and when called stores module output tensor, detaching it from the operations
    graph and moving it to the CPU to avoid GPU memory RAM overload.
    NOTE:
        * As of now, only Tensor data type handling is supported as module output.
    """
    def __init__(self, storage: list[Tensor] | None = None) -> None:
        self.storage = storage if storage is not None else []
    
    def __call__(self, module: nn.Module, in_data: Any, out_data: Tensor) -> None:
        # # NOTE: manage PyTorch multi-tensor outputs
        # if isinstance(out_data, tuple):
        #     out_data = out_data[0]
        
        if not isinstance(out_data, Tensor):
            raise ValueError(
                f"Invalid 'out_data' type {type(out_data)}, must be 'torch.Tensor'."
            )
        self.storage.append(out_data.detach().cpu())
    
    def clear(self) -> None:
        """Clears the whole storage content."""
        self.storage = []
    
    def merge(self) -> Tensor:
        """Merges the storage content to single `torch.Tensor`."""
        if not self.storage:
            return torch.empty(0)
        return torch.cat(self.storage, dim=0)


def errors_handler(func: Callable):
    """Error handler and cleanup reporter."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        error: str = ''
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            # do something? like logging / wandb.log({"status": "failed", "error": error_name})
            error = type(e).__name__
            raise
        finally:
            fn_name = func.__name__
            out_msg = (
                f"Finished executing '{fn_name}'." if not error
                else f"Found {error} in '{fn_name}'."
            )
            print(out_msg)
    return wrapper


@contextmanager
def forward_data_capture(module: nn.Module, hook_fn: Callable) -> Generator[None, None, None]:
    """Context manager for safely using a hook to capture a model's module input/output."""
    module_name: str = module.__class__.__name__
    # safely attach hook
    handle = module.register_forward_hook(hook_fn)
    print(f"Hook attached to {module_name}.")
    try:
        yield
    finally:
        # remove hook and cleanup memory
        handle.remove()
        print(f"Hook removed from {module_name}.")


# end