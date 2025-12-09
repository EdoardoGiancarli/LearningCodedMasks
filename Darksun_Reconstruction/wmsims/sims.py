"""
Module with support objects and funcs.
"""

from typing import Any, Callable
from pathlib import Path
import random

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from bloodmoon.types import CoordEquatorial
from bloodmoon.coords import angle2shift, shift2pos, shift2equatorial
from bloodmoon.mask import CodedMaskCamera, codedmask
from bloodmoon.mask import solid_angle_profile
from bloodmoon.mask import decode, variance, snratio
from bloodmoon.optim import model_shadowgram

import darksun as ds
from darksun.analyze import get_effective_area
from darksun.types import LogEntry
from darksun.data import Log

from .fields import CameraPointer
from .fields import Source
from .fields import Coords


# --- SETUP ---
def define_unit_pointings(
    z_axis_RA: float,
    z_axis_DEC: float,
    x_axis_RA: float,
    x_axis_DEC: float,
) -> CameraPointer:
    """LEM-X Unit pointing initialisation (axes RA/Dec coords in [deg])."""
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
def simul_coords(
    n_sources: int,
    *,
    fov_along_x: tuple[float, float] = (-45.0, 45.0),
    fov_along_y: tuple[float, float] = (-45.0, 45.0),
) -> tuple[Coords, ...]:
    """
    Simulates sources camera local-frame angular coords
    from uniform distribution, in [deg].
    """
    txs, tys = map(
        lambda fov: np.random.uniform(*fov, n_sources),
        (fov_along_x, fov_along_y)
    )
    return tuple((tx, ty) for tx, ty in zip(txs, tys))


def simul_fluxes(
    n_sources: int,
    f_min: float,
    f_max: float,
    pdf: Callable[[NDArray], NDArray] | None = None,
    sampling: int = 10_000,
) -> tuple[float, ...]:
    """Simulates sources fluxes in [Crab] (also from custom PDF, if given)."""
    f: NDArray = np.linspace(f_min, f_max, sampling + 1)
    weights: NDArray | None = pdf(f) if pdf is not None else None
    return tuple(random.choices(f, weights, k=n_sources))


def handmade_fluxes(
    n_sources: tuple[int, ...],
    flux_lims: tuple[tuple[float, float], ...],
    shuffle: bool = True,
) -> tuple[float, ...]:
    """
    Custom handmade sources flux simul from uniform PDF.
    Flux limits must be inserted in [Crab] unit.
    """
    # adjust inputs for lazy people (like me, lol)
    n_sources_ = (n_sources,) if isinstance(n_sources, int) else n_sources
    flux_lims_ = (flux_lims,) if isinstance(flux_lims[0], (int, float)) else flux_lims

    if not len(n_sources_) == len(flux_lims_):
        raise ValueError("Input 'n_sources' and 'flux_lims' must have same length.")
    
    fluxes: list[float] = []
    for idx, n in enumerate(n_sources_):
        fluxes += list(simul_fluxes(n, *flux_lims_[idx]))
    
    if shuffle: random.shuffle(fluxes)
    
    return tuple(fluxes)


def get_sources(
    coords: tuple[Coords, ...],
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


def sky_significance(
    camera: CodedMaskCamera,
    sources: Source | tuple[Source, ...],
    exposure: float,
    vignetting: bool = True,
    psfy: bool = True,
) -> NDArray:
    """
    Computes the simulated sky-field significance (with coding noise and CXB).
    """
    # flux values for LEM-X single coded-mask camera, from WM sims
    # https://github.com/yuri-evangelista/CodedMasks/blob/main/mask_050_1040x17/Sensitivity_cross_correlation.ipynb
    SETUP: dict[str, float] = {
        #'bkg_instr_fluence': 6.3799,    # bkg instr. flux [ph/cm2/s] (for on-axis eff area)
        #'crab_instr_fluence': 2.5737,   # 1 Crab instr. flux [ph/cm2/s] (for on-axis eff area)
        'bkg_instr_fluence': 6.7,    # bkg instr. flux [ph/cm2/s] (for on-axis eff area)
        'crab_instr_fluence': 2.5,   # 1 Crab instr. flux [ph/cm2/s] (for on-axis eff area)
    }

    def model_single_source_sg(source: Source) -> NDArray:
        """Computes the shadowgram for a single source."""
        shift_x, shift_y = map(
            lambda x: angle2shift(camera, x),
            (source.angle_x, source.angle_y),
        )
        off_axis_eff_area: float = get_effective_area(camera, shift_x, shift_y, vignetting)
        counts: float = (
            source.flux * SETUP['crab_instr_fluence'] * off_axis_eff_area * exposure
        )
        source_sg: NDArray = model_shadowgram(
            camera, shift_x, shift_y, vignetting, psfy,
        )
        return source_sg * counts

    # bkg fluence (cts / detector unit area) and dome-shaped bkg profile
    bkg_fluence: float = SETUP['bkg_instr_fluence'] * exposure
    on_axis_eff_area: float = get_effective_area(camera, 0.0, 0.0, vignetting)
    omega: NDArray = solid_angle_profile(camera)
    bkg: NDArray = (
        bkg_fluence * on_axis_eff_area * omega / omega.sum()
    )

    # compute sources shadowgrams
    sources_: tuple = (sources,) if isinstance(sources, Source) else sources
    sources_sgs: NDArray = np.zeros_like(bkg)
    for source in sources_:
        sources_sgs += model_single_source_sg(source)

    # compute significance
    detector: NDArray = bkg + sources_sgs
    sky: NDArray = decode(camera, detector)
    varmap: NDArray = variance(camera, detector)
    snrmap: NDArray = snratio(sky, varmap)

    return snrmap


def get_source_snr(
    camera: CodedMaskCamera,
    source: Source,
    snrmap: NDArray,
) -> float:
    """Returns the significance of the source from the sky-field SNR map."""
    shift_x, shift_y = map(
        lambda x: angle2shift(camera, x),
        (source.angle_x, source.angle_y),
    )
    pos: tuple[int, int] = shift2pos(camera, shift_x, shift_y)
    return snrmap[*pos]


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
    snrmap: NDArray,
    sdl: CameraPointer,
    camera: CodedMaskCamera,
) -> np.recarray:
    """
    Builds the sources record structure.
    """
    def angle2equatorial(anglex: float, angley: float) -> CoordEquatorial:
        """Converts camera local-frame angular coords in RA/Dec coords."""
        sx, sy = map(lambda x: angle2shift(camera, x), (anglex, angley))
        return shift2equatorial(sdl, camera, sx, sy)
    
    snrvalues = tuple(
        get_source_snr(camera, s, snrmap) for s in sources
    )
    rec = np.rec.array(
        obj=[
            (name, tx, ty, *angle2equatorial(tx, ty), f, snr)
            for (name, tx, ty, f), snr in zip(sources, snrvalues)
        ],
        dtype=[
            ('ID', 'U10'), ('ANGLE_X', 'f8'), ('ANGLE_Y', 'f8'),
            ('RA', 'f8'), ('DEC', 'f8'), ('FLUX', 'f8'), ('SNR', 'f16'),
        ],
    )
    return rec


def make_catalog(
    sources: Source | tuple[Source, ...],
    snrmap: NDArray,
    sdl: CameraPointer,
    camera: CodedMaskCamera,
    save_to: str | Path | None = None,
) -> DataFrame:
    """Creates the catalog for the WISEMAN simulator."""
    sources_: tuple[Source, ...] = (sources,) if isinstance(sources, Source) else sources
    record: np.recarray = build_record(sources_, snrmap, sdl, camera)
    df: DataFrame = DataFrame(record)
    if save_to is not None:
        df.to_csv(path_or_buf=save_to, index=False)
    return df


# end