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
    'errors_handler',
    'link_hook',
    'forward_data_capture',
    'OutputManager',
    'config_out_storing_hook',
]


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


def link_hook(module: nn.Module, hook_fn: Callable):
    """
    Safely links and manages the lifecycle of a hook during a function call.
    NOTE: a decorator like this is active during the whole func call, and
          does not support ON/OFF switch.
          This makes it unstable and prone to produce bugs when saving/loading
          data (e.g., when saving checkpoints during training.)
          Two other choices are: use a context manager (see `forward_data_capture`),
          or use a state-aware `HookManager` obj, with the possibility
          to choose to activate/deactivate the hook data capture.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # safely attach hook
            handle = module.register_forward_hook(hook_fn)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # remove hook and cleanup memory
                handle.remove()
                print(f"Hook removed from {module.__class__.__name__}.")
        return wrapper
    return decorator


@contextmanager
def forward_data_capture(
    module: nn.Module,
    hook_fn: Callable,
    verbose: bool = False,
) -> Generator[None, None, None]:
    """Context manager for safely using a hook to capture a model's module input/output."""
    module_name: str = module.__class__.__name__
    # safely attach hook
    handle = module.register_forward_hook(hook_fn)
    if verbose: print(f"Hook attached to {module_name}.")
    try:
        yield
    finally:
        # remove hook and cleanup memory
        handle.remove()
        if verbose: print(f"Hook removed from {module_name}.")


class OutputManager:
    """
    Manager for a model output data object. This class acts as a manager for data,
    and when called stores module output tensor, detaching it from the operations
    graph and moving a copy to the CPU to avoid GPU memory RAM overload.
    It is also possible to store the output data labels for benchmark routines.
    NOTE:
        * As of now, only Tensor data type handling is supported. If the model's
          output is a Tensor, it works as it is. If the model's output is a tuple
          of Tensors, ONLY the FIRST element of the sequence is considered, and
          it must be a Tensor obj.
    """
    def __init__(
        self,
        output: list[Tensor] | None = None,
        labels: list[Tensor] | None = None,
    ) -> None:
        self.output = output if output is not None else []
        self.labels = labels if labels is not None else []
    
    def __call__(self, module: nn.Module, in_data: Any, out_data: Tensor | tuple[Tensor, ...]) -> None:
        if isinstance(out_data, tuple):
            out_data = out_data[0]
        if not isinstance(out_data, Tensor):
            raise ValueError(
                f"Invalid 'out_data' type {type(out_data)}, must be 'torch.Tensor'."
            )
        self.output.append(out_data.detach().cpu())
    
    def add_labels(self, data_labels: Tensor):
        """Adds the data labels to intern labels storage."""
        self.labels.append(data_labels.detach().cpu())
    
    def clear(self) -> None:
        """Clears the whole storage content."""
        self.output = []
        self.labels = []
    
    def merge(self) -> tuple[Tensor, Tensor]:
        """Merges the output and labels storage content to `torch.Tensor`s."""
        if not self.output:
            return torch.empty(0), torch.empty(0)
        return torch.cat(self.output, dim=0), torch.cat(self.labels, dim=0)


def config_out_storing_hook(storage: list[Tensor] | None = None) -> tuple[Callable[[Any], None], list[Tensor]]:
    """Initialises a storage hook for the model output in the feed-forward step."""
    storage = storage if storage is not None else []

    def hook(module: nn.Module, in_data: tuple, out_data: Tensor) -> None:
        """Hook for dynamic forward storage for output data during training."""
        storage.append(out_data.detach().cpu())
        return
    
    return hook, storage


# end