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
from darksun.types import Candidate
from darksun.data import Log
from darksun.data import create_log
from darksun.data import DataLoader
from darksun.optim import retrieve_detector
from darksun.optim import detector_smoothing

# from new_optimiser import optimize


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
    SETUP = {
        'slit_mask_fine': int(
            camera.specs.slit_deltax * camera.upscale_f.x / camera.specs.mask_deltax
        ) // 2,
        'slit_mask_coarse': int(
            camera.specs.slit_deltay * camera.upscale_f.y / camera.specs.mask_deltay
        ) // 2,
        'skymap_mask': np.ones(camera.shape_sky, dtype=int),
    }

    def _update_skymap_mask(pos: tuple[int, int]) -> None:
        """Updates the skymap mask with the new candidate position."""
        SETUP['skymap_mask'][
            pos[0] - SETUP['slit_mask_coarse'] : pos[0] + SETUP['slit_mask_coarse'] + 1,
            pos[1] - SETUP['slit_mask_fine'] : pos[1] + SETUP['slit_mask_fine'] + 1,
        ] = 0
        return None

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
            if (snr[*pos] > snr_threshold) and SETUP['skymap_mask'][*pos]:
                _update_skymap_mask(pos)
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


def shifts_errors(camera: CodedMaskCamera) -> tuple[float, float]:
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


def iros_pre_smoothing(*args, **kwargs) -> tuple[Candidate, ...]:
    """
    Performs the IROS loop for detector smoothing.
    """
    print("# Running pre-process loop...")
    loop = iros_singleCAM(*args, **kwargs)
    cands = tuple(c for c, _ in tqdm(loop))
    print("# End pre-process loop...\n")
    return cands


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
            LEM-X module coded-mask camera ID (for the data Log).
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
    varmap = variance(camera, detector)
    
    # performing IROS to remove the brightest sources (SNR > SMOOTHING_THRESH)
############    brightest_cands = iros_pre_smoothing(
############        detector,
############        camera,
############        snr_threshold=SMOOTHING_THRESH,
############        vignetting=vignetting,
############        psfy=psfy,
############        varmap=varmap,
############    )
############
############    # perform detector smoothing and run again IROS on the processed data;
############    # to do that, we first remove the stored sources from the original
############    # detector, and then we perform the smoothing
############    smoothed = detector_smoothing(
############        detector=detector,
############        candidates=brightest_cands,
############        camera=camera,
############        vignetting=vignetting,
############        psfy=psfy,
############    )
############    
############    print("# Initialising loop with smoothed detector...")
    loop = iros_singleCAM(
        detector=detector,
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