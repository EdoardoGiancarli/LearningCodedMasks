"""
Module for false catalogue generation as input to the WISEMAN Monte Carlo simulator [1].

Ref:
    [1] Ceraudo, F. et al. Development of the end-to-end simulator of the WFM camera,
        Vol. 13093 of Society of Photo-Optical Instrumentation Engineers (SPIE)
        Conference Series, 130936T (2024)
"""

import os
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

####################################################################################
# --- plotting setup
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.ioff()


# --- module funcs (setup)
@dataclass(frozen=True)
class MiniSDL:
    """Test SimulationDataLoader instance with camera pointing."""
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

def init_cameras_mask(path: str | Path) -> CodedMaskCamera:
    """LEM-X module cameras mask initialisation."""
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


# --- module funcs (sources sim)
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


# --- module funcs (handle dataframe with catalogue)
def _build_record(
    sources: tuple[Source, ...],
    sdl: MiniSDL,
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
    sdl: MiniSDL,
    camera: CodedMaskCamera,
    save_to: str | Path | None = None,
) -> None:
    """Creates the catalog for the WISEMAN simulator."""
    record: np.recarray = _build_record(sources, sdl, camera)
    df: DataFrame = DataFrame(record)
    if save_to is not None:
        df.to_csv(path_or_buf=save_to, index=False)
    return df


# --- module funcs (flux PDF func)
from scipy.special import gamma

def mod_kernel(
    x: NDArray,
    alpha: float,
    k: float,
    beta: float,
) -> NDArray:
    """Custom exp - power law probability distribution function."""
    f = (alpha + 1) / beta
    a = beta * np.pow(k, f) / gamma(f)
    t = - k * np.pow(x, beta)
    return np.abs(a * np.pow(x, alpha) * np.exp(t))


####################################################################################


def main() -> None:
    # - Cameras mask and pointings setup
    base_path: str = "..."
    MASK_FITS: str = "wfm_mask_NTHT_20250725.fits"
    wfm: CodedMaskCamera = init_cameras_mask(f"{base_path}/{MASK_FITS}")
    sdl: MiniSDL = MiniSDL(
        CAMZRA=266.4,
        CAMZDEC=-28.94,
        CAMXRA=266.4,
        CAMXDEC=61.06,
    )

    # - Catalogue setup
    n_sources: int = 40
    fov: tuple[float, float] = (-40.0, 40.0)   # [deg]
    f_min: int | float = 2.5e-2                # [Crab]
    f_max: int | float = 10                    # [Crab]

    # - Sim sources
    factors = (2.0, 4.0, 2.0)
    pdf: Callable[[NDArray], NDArray] = flux_pdf(mod_kernel, *factors)

    coords: tuple[tuple[float, float], ...] = simul_coords(n_sources, fov=fov)
    fluxes: tuple[float, ...] = simul_fluxes(
        n_sources=n_sources,
        f_min=f_min,
        f_max=f_max,
        pdf=pdf,
    )
    sources: tuple[Source, ...] = get_sources(
        coords=coords,
        fluxes=fluxes,
    )

    # - Gen Log structure and show sky-field grid
    params: tuple[LogEntry, ...] = (
        LogEntry('ID', 'A20', ''),
        LogEntry('angle_x', 'f8', 'deg'),
        LogEntry('angle_y', 'f8', 'deg'),
    )
    log: Log = ds.create_log(params)
    log.add_entry_values('ID', [s.ID for s in sources])
    log.add_entry_values('angle_x', [s.angle_x for s in sources])
    log.add_entry_values('angle_y', [s.angle_y for s in sources])


    SAVE_TO: str = '/home/edoardo/Desktop'
    #SAVE_TO: str = '/mnt/d/PhD_AASS/Coding/Images_fits'
    ds.skyfield_map(log, wfm, show_IDs=True, save_to=ds.savefig_to(SAVE_TO, 'skymap', overwrite=True))

    # - Gen catalogue dataframe
    df: DataFrame = make_catalog(sources, sdl, wfm, f'{SAVE_TO}/catalogue.csv')

    return None




if __name__ == '__main__':
    main()


# end