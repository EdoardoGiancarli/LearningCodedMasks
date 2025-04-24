"""
Support methods for analyses.
"""

from typing import Generator, Any
from contextlib import contextmanager
import time
from datetime import datetime

__all__ = [
    "timer",
]


@contextmanager
def timer(
    name: str,
    units: str = "s",
) -> Generator[None, Any, None]:
    """
    Timer context manager.

    Args:
        name (str):
            Name of the task the timer is handling.
        units (str, optional (default='s')):
            Units of the measured time (seconds 's', minutes 'min' or hours 'h').
    
    Raises:
        ValueError: For invalid 'units' input.
        Exception: Encoutered exception within the algorithm (if any).
    
    TODO: insert units for final elapsed time.
    """
    if units not in ["s", "min", "h"]:
        raise ValueError(f"Invalid 'units' {units}, choose between 's', 'min' or 'h'.")

    start_time = time.perf_counter()
    error = False

    print(f"    # Starting '{name}' at {datetime.now():%H:%M:%S}.")
    try:
        yield
    except Exception as e:
        error = type(e).__name__
        raise
    finally:
        end_time = time.perf_counter()
        mess = f"    # Finished '{name}' in {end_time - start_time:.3f}s"
        print(mess + ".\n" if not error else mess + f" with {error}.\n")


# end