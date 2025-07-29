from typing import Iterable
import warnings

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from bloodmoon.coords import angle2shift
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import count
from bloodmoon.mask import decode
from bloodmoon.mask import snratio
from bloodmoon.mask import variance
from bloodmoon.optim import optimize, model_sky

from darksun.types import LogEntry
from darksun.data import Log
from darksun.data import create_log
from darksun.data import DataLoader


def iros_singleCAM(
    camera: CodedMaskCamera,
    sdl: DataLoader,
    max_iterations: int,
    snr_threshold: float = 0.0,
    vignetting: bool = True,
    psfy: bool = True,
    sky_start: NDArray | None = None,
) -> Iterable:
    """

    """    
    def find_candidate(sky: NDArray, snr: NDArray, batch: int = 1000) -> tuple:
        """Returns candidate."""
        reservoir = np.array([np.unravel_index(id, sky.shape) for id in np.argsort(sky, axis=None)[-batch:]])[::-1]
        for pos in reservoir:
            if (snr[*pos] > snr_threshold):
                return tuple(pos)
        return tuple()

    def subtract(
        candidate: tuple[int, int],
        sky: NDArray,
        snr: NDArray,
    ) -> tuple[tuple[float, float, float, float], NDArray]:
        """Runs optimizer and subtract source."""
        try:
            shiftx, shifty, fluence = optimize(
                camera=camera,
                sky=sky,
                arg_sky=candidate,
                vignetting=vignetting,
                psfy=psfy,
            )
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}") from e

        significance = float(snr[*candidate])
        model = model_sky(
            camera=camera,
            shift_x=shiftx,
            shift_y=shifty,
            fluence=fluence,
            vignetting=vignetting,
            psfy=psfy,
        )
        residual = sky - model
        return (shiftx, shifty, fluence, significance), residual
    
    detector = count(camera, sdl.DLdata)[0]
    var_ = variance(camera, detector)
    skymap = decode(camera, detector) if sky_start is None else sky_start
    for i in range(max_iterations):
        snrmap = snratio(skymap, np.clip(var_, a_min=1, a_max=None))
        candidate = find_candidate(skymap, snrmap)
        if not candidate:
            break
        try:
            source, skymap = subtract(candidate, skymap, snrmap)
        except RuntimeError as e:
            warnings.warn(f"Optimizer failed at iteration {i}:\n\n{e}")
            continue
        yield (source, skymap)


def run_IROS(
    IDcam: str,
    camera: CodedMaskCamera,
    sdl: DataLoader,
    max_iterations: int,
    snr_threshold: float = 0.0,
    vignetting: bool = True,
    psfy: bool = True,
    sky_start: NDArray | None = None,
) -> tuple[Log, NDArray]:
    """
    """
    # coded-mask sensitivity along the (x, y) axis
    # - TODO: insert correct camera sensitivity estimation (this is a proxy,
    #         dthetax = 5 arcmin, dthetay = 60 arcmin at (upx, upy) = (1, 1))
    UPX, UPY = camera.upscale_f
    DTHETA_X = 5.0 / UPX / 60                  # [deg] PN: `/ 60` is for arcmin -> deg
    DTHETA_Y = 60.0 / UPY / 60                 # [deg]
    # errors for sky-coords shifts
    DSX = abs(angle2shift(camera, DTHETA_X))   # [mm]
    DSY = abs(angle2shift(camera, DTHETA_Y))   # [mm]

    def callback(output: tuple[float]) -> tuple[float]:
        """Manage IROS candidate output parameters."""
        sx, sy, f, signf = output
        df = np.sqrt(f)
        return sx, DSX, sy, DSY, f, df, signf
    
    # generate IROS output log
    params = (
        LogEntry('shift_x', 'D', 'mm'), LogEntry('dshift_x', 'D', 'mm'),
        LogEntry('shift_y', 'D', 'mm'), LogEntry('dshift_y', 'D', 'mm'),
        LogEntry('fluence', 'D', 'ph'), LogEntry('dfluence', 'D', 'ph'),
        LogEntry('snr', 'D', ''),
    )
    cam_log = create_log(params, IDcam)

    print("# Initializing Loop...")
    loop = iros_singleCAM(
        camera=camera,
        sdl=sdl,
        max_iterations=max_iterations,
        snr_threshold=snr_threshold,
        vignetting=vignetting,
        psfy=psfy,
        sky_start=sky_start,
    )
    print("# Looping around the FOV...")
    for candidate, residual in tqdm(loop):
        cam_log.update(
            tuple((p.entry, val) for p, val in zip(params, callback(candidate)))
        )

    return cam_log, residual


# end    