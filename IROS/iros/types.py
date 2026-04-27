"""
Custom data types and containers for the LEM-X cameras IROS pipeline.
"""

from typing import NamedTuple
from numpy.typing import NDArray


class OptResult(NamedTuple):
    """Optimisation results."""
    params: NDArray
    covar: NDArray


class Source(NamedTuple):
    """
    Source candidate parameters container.

    Attributes:
        shift_x (float):
            Coded-mask camera local frame sky-coord along the x-axis [mm].
        shift_y (float):
            Coded-mask camera local frame sky-coord along the y-axis [mm].
        fluence (float):
            Observed candidate fluence [ph].
        snr (float):
            Source significance [adim].
    """
    shift_x: float
    shift_y: float
    fluence: float
    snr: float


# end