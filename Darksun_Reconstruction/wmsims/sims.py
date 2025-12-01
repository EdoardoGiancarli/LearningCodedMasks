"""
Module with support objects and funcs.
"""

from typing import Any, Callable, NamedTuple
from pathlib import Path
from dataclasses import dataclass
from random import choices

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from bloodmoon.types import CoordEquatorial
from bloodmoon.coords import angle2shift, shift2equatorial
from bloodmoon.mask import CodedMaskCamera, codedmask

import darksun as ds
from darksun.types import LogEntry
from darksun.data import Log


# --- SETUP ---
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


def define_unit_pointings(
    z_axis_RA: float,
    z_axis_DEC: float,
    x_axis_RA: float,
    x_axis_DEC: float,
) -> CameraPointer:
    """LEM-X Unit pointing initialisation."""
    return CameraPointer(
        z_axis_RA, z_axis_DEC, x_axis_RA, x_axis_DEC,
    )


def init_cameras_mask(path: str | Path) -> CodedMaskCamera:
    """LEM-X Unit CodedMaskCamera instance initialisation."""
    # here init to upscaling (fine, coarse) = (5, 1)
    # just to have a nicely dense binning structure
    UPS_X: int = 5
    UPS_Y: int = 1
    return codedmask(path, UPS_X, UPS_Y)


def config_pdf(
    pdf: Callable[[Any], NDArray],
    *args: Any,
    **kwargs: Any,
) -> Callable[[NDArray], NDArray]:
    """Configures a PDF with the given parameters."""
    
    def f(x: NDArray) -> NDArray:
        """Sky-field sources parameter distribution, with explicit unit."""
        return pdf(x, *args, **kwargs)
    
    return f


# --- SOURCE SIM ---
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


def simul_coords(
    n_sources: int,
    *,
    fov_along_x: tuple[int | float, int | float] = (-45, 45),
    fov_along_y: tuple[int | float, int | float] = (-45, 45),
    pdf: Callable[[NDArray], NDArray] | None = None,
    sampling: int = 10_000,
) -> tuple[tuple[float, float], ...]:
    """
    Simulates sources camera local-frame angular coords
    in [deg] (also from custom PDF, if given).
    """
    def extract_coords(fov: tuple[float, float]) -> tuple[float, ...]:
        """Returns sources local-frame angular coords along single axis."""
        ground: NDArray = np.linspace(*fov, sampling + 1)
        weights: NDArray | None = pdf(ground) if pdf is not None else None
        return tuple(choices(ground, weights, k=n_sources))

    # NOTE: here is a little tricky, since this logic assumes a 1D PDF
    #   - should be generalised to a 2D PDF (where the axes are linked...)
    #   - otherwise, should insert as input `pdf_along_x`, and `pdf_along_y`
    txs: tuple[float, ...] = extract_coords(fov_along_x)
    tys: tuple[float, ...] = extract_coords(fov_along_y)
    return tuple((tx, ty) for tx, ty in zip(txs, tys))


def simul_fluxes(
    n_sources: int,
    f_min: int | float,
    f_max: int | float,
    pdf: Callable[[NDArray], NDArray] | None = None,
    sampling: int = 10_000,
) -> tuple[float, ...]:
    """Simulates sources fluxes in [Crab] (also from custom PDF, if given)."""
    f: NDArray = np.linspace(f_min, f_max, sampling + 1)
    weights: NDArray | None = pdf(f) if pdf is not None else None
    return tuple(choices(f, weights, k=n_sources))


def get_sources(
    coords: tuple[tuple[float, float], ...],
    fluxes: tuple[float, ...],
    IDs: tuple[str, ...] | None = None,
) -> tuple[Source, ...]:
    """
    Simulates the sky-field sources with IDs from input
    fluxes and camera local-frame angular coords values.
    """
    if not len(coords) == len(fluxes):
        raise ValueError("Input coords and fluxes tuples must have same length.")
    if IDs is None:
        IDs = tuple(f's{idx}' for idx in range(len(coords)))
    
    sources: tuple[Source, ...] = tuple(
        Source(id_, thx, thy, f) for id_, (thx, thy), f in zip(IDs, coords, fluxes)
    )
    return sources


# --- DATABASE HANDLING ---
def gen_data_log(sources: tuple[Source, ...]) -> Log:
    """Generates the Log structure for the simulated sources."""
    params: tuple[LogEntry, ...] = (
        LogEntry('ID', 'A20', ''),
        LogEntry('angle_x', 'f8', 'deg'),
        LogEntry('angle_y', 'f8', 'deg'),
    )
    log: Log = ds.create_log(params)
    log.add_entry_values('ID', [s.ID for s in sources])
    log.add_entry_values('angle_x', [s.angle_x for s in sources])
    log.add_entry_values('angle_y', [s.angle_y for s in sources])
    return log


def build_record(
    sources: tuple[Source, ...],
    sdl: CameraPointer,
    camera: CodedMaskCamera,
) -> np.recarray:
    """
    Builds the sources record structure.
    """
    def angle2equatorial(anglex, angley) -> CoordEquatorial:
        """Converts camera local-frame angular coords in RA/Dec coords."""
        sx, sy = map(lambda x: angle2shift(camera, x), (anglex, angley))
        return shift2equatorial(sdl, camera, sx, sy)

    rec = np.rec.array(
        obj=[
            (name, tx, ty, *angle2equatorial(tx, ty), f)
            for (name, tx, ty, f) in sources
        ],
        dtype=[
            ('ID', 'U10'), ('ANGLE_X', 'f8'), ('ANGLE_Y', 'f8'),
            ('RA', 'f8'), ('DEC', 'f8'), ('FLUX', 'f8'),
        ],
    )
    return rec


def make_catalog(
    sources: tuple[Source, ...],
    sdl: CameraPointer,
    camera: CodedMaskCamera,
    save_to: str | Path | None = None,
) -> None:
    """Creates the catalog for the WISEMAN simulator."""
    record: np.recarray = build_record(sources, sdl, camera)
    df: DataFrame = DataFrame(record)
    if save_to is not None:
        df.to_csv(path_or_buf=save_to, index=False)
    return df


# end