from typing import Iterable
import warnings

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from bloodmoon.coords import angle2shift
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import count
from bloodmoon.mask import decode
# from bloodmoon.mask import variance
from bloodmoon.optim import optimize
from bloodmoon.optim import model_shadowgram
from bloodmoon.optim import model_sky

from darksun.types import LogEntry
from darksun.data import Log
from darksun.data import create_log
from darksun.data import DataLoader
from darksun.optim import bkg_smoothing

from var import sky_variance as variance


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
    Performs the Iterative Removal of Sources (IROS) algorithm for a single coded-mask
    camera of the Wide Field Monitor observations.

    This function implements an iterative source detection and removal procedure.
    For each iteration, it:
    1. Ranks source candidates by peak intensity
    2. Validates candidates by significance
    3. Fits source parameters
    4. Removes fitted sources from the sky image
    5. Repeats until no significant sources remain or max iterations reached

    Args:
        ...

    Yields:
        TODO: update here!
        output (tuple):
            - aaa (bbb):
                Candidate local-frame sky-shift coords, fluence and significance.
            - aaa (bbb):
                Coded-camera residual sky after removing the current candidate.

    Raises:
        RuntimeError: If source parameter optimization fails (with detailed error message)

    ## Notes:
        Performance Considerations:
        - Computation scales with mask resolution. Keep upscaling factors low
          (upscale_x * upscale_y ~< 10) for reasonable performance

        Algorithm Details:
        - Optimizes source parameters in local windows around candidates
        - When using reconstructed data, accounts for vignetting and PSF effects

    Example: TODO: update here!
    >>> for sources, residuals in iros(camera, sdl_cam1a, sdl_cam1b, max_iterations=2):
    >>>     source_1a, source_1b = sources
    >>>     residual_1a, residual_1b = residuals
    >>>     ...
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


def camera_angular_resolution(camera: CodedMaskCamera) -> tuple[float, float]:
    """
    Computes the camera angular resolution along the axes, in [arcmin].

    Args:
        camera (CodedMaskCamera):
            Instance with info on the camera system geometry.
    
    Returns:
        output (tuple[float, float]):
            Camera angular resolution along the (x, y) axes in [arcmin].
    
    ## Notes:
        * From: Skinner, G.K., 2008. Sensitivity of coded mask telescopes.
          Applied optics, 47(15), pp.2739-2749.
    """
    def angular_resolution(m_pitch: float, d_pitch: float, dist: float) -> float:
        """
        Computes the camera angular resolution along the axis, in [arcmin].

        Args:
            m_pitch (float): Mask element pitch.
            d_pitch (float): Detector element resolution pitch.
            dist (float): Mask - Detector distance.
        """
        dtheta_rad = np.sqrt(
            np.square(m_pitch / dist) + np.square(d_pitch / dist)
        )
        dtheta_arcmin = np.rad2deg(dtheta_rad) * 60
        return dtheta_arcmin
    
    p = camera.specs['mask_detector_distance']
    mx, my = (
        camera.specs['slit_deltax'],
        camera.specs['slit_deltay'],
    )
    dx, dy = (
        camera.specs['...'],
        camera.specs['...'],
    )
    return (
        angular_resolution(mx, dx, p),
        angular_resolution(my, dy, p),
    )


def camera_skycoords_errors(camera: CodedMaskCamera) -> tuple[float, float]:
    """
    Computes the camera local-frame coords NOMINAL* errors along
    the axes, taking into account the chosen camera upscaling.

    *For now, we are considering a proxy for the camera angular
     resolution along the fine and coarse directions.

    Args:
        camera (CodedMaskCamera):
            Instance with info on the camera system geometry.
    
    Returns:
        output (tuple[float, float]):
            Camera local-frame cartesian coords errors along
            the (x, y) axes in [mm].
    """
    def arcmin2deg(angle: float) -> float:
        """Converts angle from [arcmin] to [deg]."""
        return angle / 60
    
    UPX, UPY = camera.upscale_f
    ang_res_x, ang_res_y = camera_angular_resolution(camera)
    dsx = abs(angle2shift(camera, arcmin2deg(ang_res_x / UPX)))  # [mm]
    dsy = abs(angle2shift(camera, arcmin2deg(ang_res_y / UPY)))  # [mm]
    return dsx, dsy


def shifts_errors(camera: CodedMaskCamera) -> tuple[float, float]:
    """
    Computes the camera local-frame coords NOMINAL* errors along
    the axes, taking into account the chosen camera upscaling.

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
            - residual (NDArray):
                Coded-camera residual sky after IROS.
    
    TODO:
        * add smoothing doc
    """    
    # generate IROS output log
    params = (
        LogEntry('shift_x', 'D', 'mm'), LogEntry('dshift_x', 'D', 'mm'),
        LogEntry('shift_y', 'D', 'mm'), LogEntry('dshift_y', 'D', 'mm'),
        LogEntry('fluence', 'D', 'ph'), LogEntry('dfluence', 'D', 'ph'),
        LogEntry('snr', 'D', ''),
    )
    cam_log = create_log(params, IDcam)

    # get camera local-frame coords sensitivity along the axes
    DSX, DSY = shifts_errors(camera)
    
    # define significance threshold for detector smoothing
    SMOOTHING_THRESH = 25.0

    # generating detector image
    detector = count(camera, sdl.DLdata)[0]
    skymap = decode(camera, detector)
    varmap = variance(camera, detector)
    # varmap = np.clip(variance(camera, detector), a_min=1e-8, a_max=detector.sum())

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
    def callback(output: tuple[float]) -> tuple[float]:
        """Manage IROS candidate output parameters."""
        sx, sy, f, signf = output
        df = np.sqrt(f)
        return sx, DSX, sy, DSY, f, df, signf
    
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

    smoothed_res_detector = bkg_smoothing(
        detector=detector - retrieve_detector(candidates),
        camera=camera,
    )
    smoothed_skymap = decode(
        camera,
        np.clip(detector - smoothed_res_detector, a_min=0.0, a_max=detector.sum()),
    )
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