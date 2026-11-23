"""
Coded-mask camera geometric configuration analyser.
This module is intended for benchmarking the performance of a LEM-X camera accounting for assembling systematics,
such as mask and/or SDDs (or even whole detector plane) rotations and/or shifts with respect to the nominal design.
"""

from functools import partial
import multiprocessing
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from tqdm import tqdm

from bloodmoon.coords import equatorial2pos, equatorial2shift, shift2angle
from bloodmoon.images import argmax
from bloodmoon.mask import CodedMaskCamera, codedmask
from bloodmoon.mask import count, decode, variance
from bloodmoon.types import CoordEquatorial
import darksun as ds
from darksun.benchmarking import source_catalogue_data, crop_source_psf
from darksun.data import DataLoader, CatalogueLoader

from IROSrec.iros.optim import default_optimiser
from IROSrec.iros.types import OptResult


# ------   I/O operations   ------

def check_and_pick(parent: Path, pattern: str) -> Path:
    matches = tuple(parent.glob(pattern))
    if not matches:
        raise ValueError(f"A file matching the pattern {str(parent / pattern)} is expected but missing.")
    f, *extra_matches = matches
    if extra_matches:
        raise ValueError(
            f"Found unexpected extra matches for glob pattern {str(parent / pattern)}."
            f"File with pattern {pattern} should be unique"
        )
    return f

def df_to_csv(df: pd.DataFrame, save_to: str | Path) -> None:
    """Saves input dataframe to `.csv` file."""
    kws = {'index': False,'float_format': '%.6f'}
    df.to_csv(save_to, **kws)
    return

# ------   helpers   ------

def config_fitter(
    optimiser: Callable[[NDArray, tuple[int, int]], OptResult],
) -> Callable[[tuple[int, int], NDArray], tuple[NDArray, NDArray]]:
    """
    Configures source candidates parameters fitter.
    """
    def fitter(arg_sky: tuple[int, int], sky: NDArray) -> tuple[NDArray, NDArray]:
        """Performs the optimisation of the source candidate params."""
        try:
            params: OptResult = optimiser(
                sky=sky,
                arg_sky=arg_sky,
            )
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}") from e
        
        return params.params, params.covar
    
    return fitter

# ------   coords handling   ------

def get_source_coords(sourceID: str, catalogue: CatalogueLoader) -> CoordEquatorial:
    """Extracts the source RA/Dec coords from catalogue."""
    data = source_catalogue_data(sourceID, catalogue.DLdata)
    return CoordEquatorial(data['RA'], data['DEC'])

def equatorial2angle(
    sdl: DataLoader,
    camera: CodedMaskCamera,
    ra: float,
    dec: float,
) -> tuple[float, float]:
    """Converts RA/Dec coords to camera local-frame angular coords."""
    shifts = equatorial2shift(sdl, camera, ra, dec)
    return tuple(shift2angle(camera, s) for s in shifts)

# ------   array operations   ------

def find_boxmax(sky: NDArray, centre: tuple[int, int], boxsize: int = 5) -> tuple[int, int]:
    """Finds the argmax of given sky in a box centered around `pos`."""
    rcut, ccut = map(lambda idx: slice(idx - boxsize, idx + boxsize + 1), centre)
    p, q = centre
    n, m = argmax(sky[rcut, ccut])
    return p - boxsize + n, q - boxsize + m

# ------   output data logging   ------



# ------   analysis   ------




# end