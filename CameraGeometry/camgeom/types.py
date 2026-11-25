"""
Module for `camgeom` custom types.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimType:
    """
    Camera geometry simulation type to be performed by WISEMAN.

    Attributes:
        level (str):
            Target subsystem (`'mask'` or `'sdd'`).
        stype (str):
            Geometry perturbation type (e.g., `'rots'`, `'shifts'`, `'warps'`,
            `'plane_expans'`, `'plane_deforms'`, `'plane_inrots'`, `'plane_warps'`).
        axis (str):
            Reference axis or mode (`'x'`, `'y'`, `'z'`, or `'radial'`).
        value (tuple[float, str]):
            Perturbation magnitude and unit as `(amount, unit)`. Supported units
            are `'deg'`, `'arcmin'`, `'mm'`, and `'micron'`.
        sddID (str, optional=`'00'`):
            Target SDD identifier (`'00'`, `'10'`, `'01'`, `'11'`) for single-SDD
            operations (`'rots'`, `'shifts'`).
        sddplane_deform (str, optional=`'external'`):
            Detector plane deformation pattern (`'external'`, `'internal'`,
            `'slide'`, or `'trapez'`) used when `stype='plane_deforms'`.
        rm_sdd_inrots (tuple[str, ...], optional=`()`):
            Sign combinations to exclude when `stype='plane_inrots'`, formatted
            as comma-separated strings (e.g., `('1,1,1,1', '-1,-1,-1,-1')`)
            mapping `-1`/`+1` to clockwise/anti-clockwise rotations across
            `('00', '10', '01', '11')`.
        mask_fpath (str | Path, optional=`""`):
            Optional override path to the coded-mask pattern FITS file used
            when `level='mask'` and `stype='warps'`. If empty, falls back to
            `mask.maskPatternFile` in the baseline JSON.

    Raises:
        ValueError: If `value[1]` is not one of the supported physical units.
    """
    level: str
    stype: str
    axis: str
    value: tuple[float, str]
    sddID: str = '00'
    sddplane_deform: str = 'external'
    rm_sdd_inrots: tuple[str, ...] = ()
    mask_fpath: str | Path = ""

    def __post_init__(self) -> None:
        avail_units = ('deg', 'arcmin', 'mm', 'micron')
        if self.value[-1] not in avail_units:
            raise ValueError(f"Invalid effect unit '{self.value[-1]}', must be in {avail_units}")


# end