"""
IROS procedure operations.
"""

from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from bloodmoon.coords import angle2shift
from bloodmoon.mask import CodedMaskCamera
import darksun as ds
from darksun.data import Log
from darksun.data import create_log
from darksun.data import DataLoader
from darksun.data import CatalogueLoader
from darksun.types import LogEntry

from .optim import iros_singleCAM
from .types import Source


def get_coord_errors(camera: CodedMaskCamera) -> tuple[float, float]:
    """
    Computes the camera local-frame coords NOMINAL* errors along
    the system fine and coarse directions angular resolution,
    taking into account the chosen camera upscaling.

    *For now, we are considering a proxy for the camera angular
     resolution along the fine and coarse directions.
    
    At upscaling (upx, upy) = (1, 1) we select:
        - dthetax = 5 arcmin along the fine direction
        - dthetay = 60 arcmin along the coarse direction
    """
    def arcmin2deg(angle: float) -> float:
        """Converts angle from [arcmin] to [deg]."""
        return angle / 60
    
    UPX, UPY = camera.upscale_f
    # camera angular resolution at given upscaling [arcmin]
    dthetax, dthetay = 5.0 / UPX, 60.0 / UPY
    # local-frame sky-shifts errors [mm]
    dsx, dsy = map(
        lambda x: abs(angle2shift(camera, arcmin2deg(x))),
        (dthetax, dthetay),
    )
    return dsx, dsy


def run_IROS_loop(*args, **kwargs) -> tuple[Source, ...]:
    """Runs the IROS loop and returns found candidate sources."""
    print("# Running loop...")
    loop = iros_singleCAM(*args, **kwargs)
    cands = tuple(c for c, _ in tqdm(loop))
    print("# End loop...\n")
    return cands


def run_IROS(
    camera: CodedMaskCamera,
    loop: Iterable[tuple[Source, NDArray]],
    cameraID: str | None = None,
) -> tuple[Log, NDArray]:
    """
    Runs the IROS (Iterative Removal of Sources) loop and stores the output for the
    chosen coded-mask camera of the LEM-X observatory.

    This wrapper iteratively removes the detected sources candidates from the sky
    until either the maximum number of iterations is reached or the sky significance
    threshold is met. At each iteration, the log for the chosen coded-mask cameras
    is updated with the following candidates estimated parameters:

    * camera local frame sky-coordinates shifts along the (x, y) axes wrt the
      coded-mask camera optical axis, in [mm] [1]
    * fluence, in [ph] [2]
    * significance at the selection
    
    [1] The candidates shifts errors at upscaling `(x, y)=(1, 1)` are assumed to be
        `5 arcmin` along x and `60 arcmin` along y.\n
    [2] The candidates fluence is assumed to follow a Poissonian statistics, so the
        fluence error is the square root of the fluence.
    """
    # get camera local-frame coords sensitivity along the axes
    DSX, DSY = get_coord_errors(camera)
    # generate IROS output log
    params = (
        LogEntry('shift_x', 'D', 'mm'), LogEntry('dshift_x', 'D', 'mm'),
        LogEntry('shift_y', 'D', 'mm'), LogEntry('dshift_y', 'D', 'mm'),
        LogEntry('fluence', 'D', 'ph'), LogEntry('dfluence', 'D', 'ph'),
        LogEntry('snr', 'D', ''),
    )
    cam_log = create_log(params, cameraID)

    def callback(output: Source) -> tuple[float, ...]:
        """Manage IROS candidate output parameters."""
        sx, sy, f, signf = output
        return sx, DSX, sy, DSY, f, np.sqrt(f), signf
    
    print("# Looping around the FOV...")
    for cand, residual in tqdm(loop, desc='IROS sky-reconstruction'):
        cam_log.update(
            tuple((p.entry, val) for p, val in zip(params, callback(cand)))
        )

    return cam_log, residual


def get_sources_database(
    camera: CodedMaskCamera,
    sdl: DataLoader,
    catalogue: CatalogueLoader,
    log: Log,
    vignetting: bool | Callable = True,
    screening: bool = True,
) -> Log:
    """
    Computes sources additional parameters from IROS
    output data and performs catalogue association.
    """
    log_ = ds.compute_parameters(
        log, camera, sdl, vignetting=vignetting,
    )
    log_ = ds.catalogue_comparison(
        log_, catalogue, sdl, camera, screening=screening,
    )
    return log_


# end