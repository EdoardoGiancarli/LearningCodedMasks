from typing import Iterable
import warnings

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from bloodmoon.coords import angle2shift
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import count
from bloodmoon.mask import decode
from bloodmoon.mask import variance
from bloodmoon.optim import optimize
from bloodmoon.optim import model_shadowgram
from bloodmoon.optim import model_sky

from darksun.types import LogEntry
from darksun.data import Log
from darksun.data import create_log
from darksun.data import DataLoader
from darksun.optim import bkg_smoothing


def iros_singleCAM(
    skymap: NDArray,
    varmap: NDArray,
    camera: CodedMaskCamera,
    max_iterations: int,
    snr_threshold: float = 0.0,
    vignetting: bool = True,
    psfy: bool = True,
) -> Iterable:
    """

    """    
    def find_candidate(sky: NDArray, snr: NDArray, batch: int = 1000) -> tuple:
        """Returns candidate."""
        reservoir = np.array(
            [np.unravel_index(id, sky.shape) for id in np.argsort(sky, axis=None)[-batch:]]
        )
        for pos in reservoir[::-1]:
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
    
    for i in range(max_iterations):
        snrmap = skymap / np.sqrt(varmap)
        candidate = find_candidate(skymap, snrmap)
        if not candidate:
            print("\nNo candidates left...")
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
) -> tuple[Log, NDArray]:
    """
    Runs the IROS (Iterative Removal of Sources) loop and stores the output for the
    chosen coded-mask camera of the Wide-Field Monitor.

    This wrapper iteratively removes the detected sources candidates from the sky
    until either the maximum number of iterations is reached or the sky significance
    threshold is met. At each iteration, the log for the chosen coded-mask cameras
    is updated with the following candidates estimated parameters:

        - camera local frame sky-coordinates shifts along the (x, y)
          axes wrt the coded-mask camera optical axis, in [mm]*
        - fluence, in [ph]**
        - significance at the selection
    
    * The candidates shifts errors at upscaling `(x, y)=(1, 1)` are assumed to be
      `5 arcmin` along x and `60 arcmin` along y.
    ** The candidates fluence is assumed to follow a Poissonian statistics, so the
       fluence error is the square root of the fluence.
    
    Args:
        IDcam (str | None, optional (default=`None`)):
                WFM coded-mask camera ID (for the Log).
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        sdl (DataLoader):
            DataLoader instance with the observed data.
        max_iterations (int):
            Maximum number of iterations for the IROS loop.
        snr_threshold (int | float, optional (default=`0.0`)):
            Minimum SNR value required to continue the iterative source removal process.
        vignetting (bool, optional (default=`True`)):
            If `True`, the model used for optimization will simulate vignetting.
        psfy (bool, optional (default=`True`)):
            If `True`, the model used for optimization will simulate detector
            position reconstruction effects.

    Returns:
        output (tuple[Log, NDArray]):
            - log (Log):
                Coded-camera log with metadata and results from IROS.
            - residual (tuple[NDArray, NDArray]):
                Coded-camera residual sky after IROS.
    
    TODO:
        * add smoothing doc
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
    
    # define significance threshold for detector smoothing
    SMOOTHING_THRESH = 25.0
    
    # generate IROS output log
    params = (
        LogEntry('shift_x', 'D', 'mm'), LogEntry('dshift_x', 'D', 'mm'),
        LogEntry('shift_y', 'D', 'mm'), LogEntry('dshift_y', 'D', 'mm'),
        LogEntry('fluence', 'D', 'ph'), LogEntry('dfluence', 'D', 'ph'),
        LogEntry('snr', 'D', ''),
    )
    cam_log = create_log(params, IDcam)

    # generating detector image
    detector = count(camera, sdl.DLdata)[0]
    skymap = decode(camera, detector)
    varmap = np.clip(variance(camera, detector), a_min=1e-8, a_max=detector.sum())

    # define unframe edges to remove sky and significance boundaries
    # TODO:
    #   - setup: done
    #   - implement unframing factors (possible criteria: where not Poisson variance)
    #   - implement shifts offset (due to framing) in IROS
    # TODO:
    #   - OR BETTER: implement sky mask from variance `* (variance > n)`
    UNFR_X, UNFR_Y = None, None    # 100 * UPX, 70 * UPY

    # performing IROS to remove the brightest sources (SNR > SMOOTHING_THRESH)
    print("# Running first loop...")
    first_loop = iros_singleCAM(
        skymap=skymap,
        varmap=varmap,
        camera=camera,
        max_iterations=max_iterations,
        snr_threshold=SMOOTHING_THRESH,
        vignetting=vignetting,
        psfy=psfy,
    )
    candidates = tuple(c for c, _ in tqdm(first_loop))
    
    # perform detector smoothing and run again IROS on the processed data;
    # to do that, we first remove the stored sources from the original
    # detector, and then we perform the smoothing
    def retrieve_detector(candidates: tuple[tuple[float]]) -> NDArray:
        """Generates detector image from retrieved candidates."""
        img = np.zeros(camera.shape_detector)
        for (sx, sy, f, _) in candidates:
            shadowgram = model_shadowgram(
                camera=camera,
                shift_x=sx,
                shift_y=sy,
                vignetting=vignetting,
                psfy=psfy,
            )
            img += (f * shadowgram)
        return img

    smoothed_detector = bkg_smoothing(
        detector=detector - retrieve_detector(candidates),
        camera=camera,
    )
    smoothed_skymap = decode(camera, detector - smoothed_detector)
    print("# Initializing second loop with smoothed detector...")
    second_loop = iros_singleCAM(
        skymap=smoothed_skymap,
        varmap=varmap,
        camera=camera,
        max_iterations=max_iterations,
        snr_threshold=snr_threshold,
        vignetting=vignetting,
        psfy=psfy,
    )
    print("# Looping around the FOV...")
    for candidate, residual in tqdm(second_loop):
        cam_log.update(
            tuple((p.entry, val) for p, val in zip(params, callback(candidate)))
        )

    return cam_log, residual


# end    