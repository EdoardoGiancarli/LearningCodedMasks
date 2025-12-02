"""
Module with fields.
"""

from typing import NamedTuple
from dataclasses import dataclass

from bloodmoon.types import CoordEquatorial


type Coords = tuple[float, float]


class Source(NamedTuple):
    """
    Source field with ID, local frame coded-mask
    camera angular coordinates and flux.

    Args:
        ID (str): Source name.
        angle_x (float): Angular coord along x-axis [deg]
        angle_y (float): Angular coord along y-axis [deg]
        flux (float): Source incoming flux [Crab]
    """
    ID: str
    angle_x: float
    angle_y: float
    flux: float


@dataclass(frozen=True)
class CameraPointer:
    """Instance with LEM-X coded-mask cameras pointings."""
    CAMZRA: float
    CAMZDEC: float
    CAMXRA: float
    CAMXDEC: float

    @property
    def pointings(self) -> dict[str, CoordEquatorial]:
        """
        Camera axis pointing information in equatorial frame.
        Angles are expressed in degrees.
        """
        return {
            "z": CoordEquatorial(ra=self.CAMZRA, dec=self.CAMZDEC),
            "x": CoordEquatorial(ra=self.CAMXRA, dec=self.CAMXDEC),
        }


# end