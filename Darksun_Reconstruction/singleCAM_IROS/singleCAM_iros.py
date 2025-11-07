from typing import Iterable, NamedTuple
import warnings

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
# from scipy.ndimage import center_of_mass

from bloodmoon.coords import angle2shift
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import count
from bloodmoon.mask import decode
from bloodmoon.mask import variance
from bloodmoon.mask import snratio
from bloodmoon.optim import model_shadowgram, model_sky
from bloodmoon.optim import optimize

from darksun.types import LogEntry
from darksun.data import Log
from darksun.data import create_log
from darksun.data import DataLoader
from darksun.optim import bkg_smoothing


class Candidate(NamedTuple):
    """
    Source candidate main info container.

    Attributes:
        shift_x (float):
            Coded-mask camera local frame sky-coord along the x-axis [mm].
        shift_y (float):
            Coded-mask camera local frame sky-coord along the y-axis [mm].
        fluence (float):
            Observed candidate fluence [ph].
        snr (float):
            Candidate significance [adim].
    """
    shift_x: float
    shift_y: float
    fluence: float
    snr: float


def iros_singleCAM(
    detector: NDArray,
    camera: CodedMaskCamera,
    max_iterations: int = 40,
    snr_threshold: float = 0.0,
    vignetting: bool = True,
    psfy: bool = True,
    varmap: NDArray | None = None,
) -> Iterable[tuple[Candidate, NDArray]]:
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
            - residual (NDArray):
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
    def find_candidate(
        sky: NDArray,
        snr: NDArray,
        batch: int = 1000,
    ) -> tuple[int, int] | bool:
        """
        Returns the position of a valid IROS candidate inside the sky image.
        """
        reservoir = np.array(
            [np.unravel_index(id_, sky.shape) for id_ in np.argsort(sky, axis=None)[-batch:]]
        )
        for pos in reservoir[::-1]:
            if (snr[*pos] > snr_threshold):
                return tuple(pos)
        return False

    def fit_candidate_params(
        candidate_pos: tuple[int, int],
        sky: NDArray,
        snr: NDArray,
    ) -> Candidate:
        """Performs the optimisation of the source candidate params."""
        try:
            shift_x, shift_y, fluence = optimize(
                camera=camera,
                sky=sky,
                arg_sky=candidate_pos,
                vignetting=vignetting,
                psfy=psfy,
            )
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}") from e
        
        significance = float(snr[*candidate_pos])
        return Candidate(shift_x, shift_y, fluence, significance)

    def subtract(
        candidate: Candidate,
        detector: NDArray,
    ) -> NDArray:
        """Subtracts candidate from detector image."""
        sg_model = model_shadowgram(
            camera=camera,
            shift_x=candidate.shift_x,
            shift_y=candidate.shift_y,
            vignetting=vignetting,
            psfy=psfy,
        )
        residual = detector - candidate.fluence * sg_model
        return residual
    
    detector_ = detector.copy()
    skymap = decode(camera, detector)
    varmap = (
        varmap if varmap is not None
        else variance(camera, detector)
    )
    
    for i in range(max_iterations):
        snrmap = snratio(skymap, varmap)
        candidate_pos = find_candidate(skymap, snrmap)

        if not candidate_pos:
            print("\nNo candidates left...")
            break

        try:
            source = fit_candidate_params(candidate_pos, skymap, snrmap)
        except RuntimeError as e:
            warnings.warn(f"Optimizer failed at iteration {i}:\n\n{e}")
            continue

        detector_ = subtract(source, detector_)
        skymap = decode(camera, detector_)
        yield (source, skymap)


#def iros_singleCAM(
#    skymap: NDArray,
#    varmap: NDArray,
#    camera: CodedMaskCamera,
#    max_iterations: int = 40,
#    snr_threshold: float = 0.0,
#    vignetting: bool = True,
#    psfy: bool = True,
#) -> Iterable[tuple[Candidate, NDArray]]:
#    """
#    Performs the Iterative Removal of Sources (IROS) algorithm for a single coded-mask
#    camera of the Wide Field Monitor observations.
#
#    This function implements an iterative source detection and removal procedure.
#    For each iteration, it:
#    1. Ranks source candidates by peak intensity
#    2. Validates candidates by significance
#    3. Fits source parameters
#    4. Removes fitted sources from the sky image
#    5. Repeats until no significant sources remain or max iterations reached
#
#    Args:
#        ...
#
#    Yields:
#        TODO: update here!
#        output (tuple):
#            - aaa (bbb):
#                Candidate local-frame sky-shift coords, fluence and significance.
#            - residual (NDArray):
#                Coded-camera residual sky after removing the current candidate.
#
#    Raises:
#        RuntimeError: If source parameter optimization fails (with detailed error message)
#
#    ## Notes:
#        Performance Considerations:
#        - Computation scales with mask resolution. Keep upscaling factors low
#          (upscale_x * upscale_y ~< 10) for reasonable performance
#
#        Algorithm Details:
#        - Optimizes source parameters in local windows around candidates
#        - When using reconstructed data, accounts for vignetting and PSF effects
#
#    Example: TODO: update here!
#    >>> for sources, residuals in iros(camera, sdl_cam1a, sdl_cam1b, max_iterations=2):
#    >>>     source_1a, source_1b = sources
#    >>>     residual_1a, residual_1b = residuals
#    >>>     ...
#    """
#    def find_candidate(
#        sky: NDArray,
#        snr: NDArray,
#        batch: int = 1000,
#    ) -> tuple[int, int] | bool:
#        """
#        Returns the position of a valid IROS candidate inside the sky image.
#        """
#        reservoir = np.array(
#            [np.unravel_index(id_, sky.shape) for id_ in np.argsort(sky, axis=None)[-batch:]]
#        )
#        for pos in reservoir[::-1]:
#            if (snr[*pos] > snr_threshold):
#                return tuple(pos)
#        return False
#
#    def fit_candidate_params(
#        candidate_pos: tuple[int, int],
#        sky: NDArray,
#        snr: NDArray,
#    ) -> Candidate:
#        """Performs the optimisation of the source candidate params."""
#        try:
#            shift_x, shift_y, fluence = optimize(
#                camera=camera,
#                sky=sky,
#                arg_sky=candidate_pos,
#                vignetting=vignetting,
#                psfy=psfy,
#            )
#        except Exception as e:
#            raise RuntimeError(f"Optimization failed: {str(e)}") from e
#        
#        significance = float(snr[*candidate_pos])
#        return Candidate(shift_x, shift_y, fluence, significance)
#
#    def subtract(
#        candidate: Candidate,
#        sky: NDArray,
#    ) -> NDArray:
#        """Subtracts candidate from sky image."""
#        model = model_sky(
#            camera=camera,
#            shift_x=candidate.shift_x,
#            shift_y=candidate.shift_y,
#            fluence=candidate.fluence,
#            vignetting=vignetting,
#            psfy=psfy,
#        )
#        residual = sky - model
#        return residual
#    
#    for i in range(max_iterations):
#        snrmap = snratio(skymap, varmap)
#        candidate_pos = find_candidate(skymap, snrmap)
#
#        if not candidate_pos:
#            print("\nNo candidates left...")
#            break
#
#        try:
#            source = fit_candidate_params(candidate_pos, skymap, snrmap)
#        except RuntimeError as e:
#            warnings.warn(f"Optimizer failed at iteration {i}:\n\n{e}")
#            continue
#
#        skymap = subtract(source, skymap)
#        yield (source, skymap)




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


def iros_pre_smoothing(*args, **kwargs) -> tuple[Candidate, ...]:
    """
    Performs the IROS loop for detector smoothing.
    """
    print("# Running pre-process loop...")
    loop = iros_singleCAM(*args, **kwargs)
    cands = tuple(c for c, _ in tqdm(loop))
    print("# End pre-process loop...\n")
    return cands


def retrieve_detector(
    candidates: tuple[Candidate, ...],
    camera: CodedMaskCamera,
    vignetting: bool,
    psfy: bool,
) -> NDArray:
    """Generates detector image from retrieved candidates."""
    detector = np.zeros(camera.shape_detector)
    for (sx, sy, f, _) in candidates:
        sg = model_shadowgram(
            camera=camera,
            shift_x=sx,
            shift_y=sy,
            vignetting=vignetting,
            psfy=psfy,
        )
        detector += (f * sg)
    return detector


def detector_smoothing(
    detector: NDArray,
    candidates: tuple[Candidate, ...],
    camera: CodedMaskCamera,
    vignetting: bool,
    psfy: bool,
) -> NDArray:
    """
    Process the observed detector image by applying
    a median smoothing of the background.
    """
    KERNEL_SIZE = {
        'y': 11,
        'x': 7,
    }
    # get residual detector image
    retrieved = retrieve_detector(
        candidates=candidates,
        camera=camera,
        vignetting=vignetting,
        psfy=psfy,
    )
    res_detector = detector - retrieved
    # perform smoothing on residual detector image
    res_smoothed = bkg_smoothing(
        detector=res_detector,
        camera=camera,
        kernelsize_y=KERNEL_SIZE['y'],
        kernelsize_x=KERNEL_SIZE['x'],
    )
    # get smoothed detector image
    smoothed = np.clip(detector - res_smoothed, a_min=0.0, a_max=None)
    return smoothed


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
    SMOOTHING_THRESH = 15.0

    # IROS procedure body
    def callback(output: Candidate) -> tuple[float, ...]:
        """Manage IROS candidate output parameters."""
        sx, sy, f, signf = output
        df: float = np.sqrt(f)
        return sx, DSX, sy, DSY, f, df, signf
    
    # generating detector and sky images + variance map
    detector = count(camera, sdl.DLdata)[0]
###########    skymap = decode(camera, detector)
    varmap = variance(camera, detector)
    
    # performing IROS to remove the brightest sources (SNR > SMOOTHING_THRESH)
    brightest_cands = iros_pre_smoothing(
        detector,
        camera,
        snr_threshold=SMOOTHING_THRESH,
        vignetting=vignetting,
        psfy=psfy,
        varmap=varmap,
    )

    # perform detector smoothing and run again IROS on the processed data;
    # to do that, we first remove the stored sources from the original
    # detector, and then we perform the smoothing
    smoothed = detector_smoothing(
        detector=detector,
        candidates=brightest_cands,
        camera=camera,
        vignetting=vignetting,
        psfy=psfy,
    )
###########    smoothed_skymap = decode(camera, smoothed)
    
    print("# Initialising loop with smoothed detector...")
    loop = iros_singleCAM(
        detector=smoothed,
        camera=camera,
        max_iterations=max_iterations,
        snr_threshold=snr_threshold,
        vignetting=vignetting,
        psfy=psfy,
        varmap=varmap,
    )
    print("# Looping around the FOV...")
    for candidate, residual in tqdm(loop):
        cam_log.update(
            tuple((p.entry, val) for p, val in zip(params, callback(candidate)))
        )

    return cam_log, residual


# end