from typing import Callable, Iterable, Literal
import warnings

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from bloodmoon.coords import angle2shift
from bloodmoon.coords import shift2pos
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import count
from bloodmoon.mask import decode
# from bloodmoon.mask import variance
# from bloodmoon.optim import optimize

from scipy.optimize import minimize
from scipy.ndimage import center_of_mass
#from bloodmoon.optim import _ModelShiftFluence, _ModelShiftFluenceUncached, _Loss
from bloodmoon.optim import _Loss
from bloodmoon.mask import interpmax

#from bloodmoon.optim import model_shadowgram
#from bloodmoon.optim import model_sky

from darksun.types import LogEntry
from darksun.data import Log
from darksun.data import create_log
from darksun.data import DataLoader
from darksun.optim import bkg_smoothing

from var import sky_variance as variance
from fract_shift2 import model_shadowgram, model_sky


def _ModelShiftFluenceUncached(  # noqa
    camera: CodedMaskCamera,
    vignetting: bool = True,
    psfy: bool = True,
) -> tuple[Callable, Callable]:
    """
    A slow, vanilla implementation of the model for both direction and fluence optimization.
    Intended for debugging and benchmarking.
    """

    def f(shift_x: float, shift_y: float, fluence: float) -> NDArray:
        """
        A simple, slow version of the model for both direction and fluence optimization.

        Args:
            shift_x: Source position x-coordinate in sky-shift space (mm)
            shift_y: Source position y-coordinate in sky-shift space (mm)
            fluence: Source intensity/fluence value

        Returns:
            2D array representing the modeled sky reconstruction
        """
        return model_sky(camera, shift_x, shift_y, fluence, vignetting=vignetting, psfy=psfy)

    # there is no cache here, hence no need to clean anything.
    # we return a lambda anyway for compatibility with the other models
    return f, lambda: None


def optimize(
    camera: CodedMaskCamera,
    sky: NDArray,
    arg_sky: tuple[int, int],
    vignetting: bool = True,
    psfy: bool = True,
    #model: Literal["fast", "accurate"] = "fast",
) -> tuple[float, float, float]:
    """
    Performs the optimization to fit a point source model to sky image data.

    This function performs the optimization by simultaneously fit the candidate
    position and fluence. The starting position is inferred by interpolating the
    candidate shifts in an upscaled grid (9, 9), while the starting fluence is
    represented by the counts at the candidate extracted pixel indexes.
    The model is cached to balance speed and accuracy.

    Args:
        camera: CodedMaskCamera instance containing detector and mask parameters
        sky: 2D array of the reconstructed sky image to fit
        arg_sky: Initial guess for source position as (row, col) indices
        vignetting: If true, the model used for optimization will simulate vignetting.
        psfy: If true, the model used for optimization will simulate detector position
        reconstruction effects.

    Returns:
        Tuple containing the best-fit parameters `(x, y, fluence)` where:
                - x, y are the optimized sky-shift coordinates
                - fluence is the optimized source intensity

    Notes:
        - Initial position is refined using interpolation
        - Bounds are set based on initial guess and physical constraints
    """
    # - initialize the function to fluence and position dependent shadowgram model.
    # - it leverages caches to reduce the number of cross-correlation computations,
    #   and it is our responsibility to free memory after we will be done.
    #if model == "fast":
    #    model_shift_flux, model_shift_flux_clear = _ModelShiftFluence(camera, vignetting, psfy)
    #elif model == "accurate":
    #    model_shift_flux, model_shift_flux_clear = _ModelShiftFluenceUncached(camera, vignetting, psfy)
    #else:
    #    raise ValueError("Model value not supported. The `model` arguments should be `fast` or `accurate`.")
    
    model_shift_flux, model_shift_flux_clear = _ModelShiftFluenceUncached(camera, vignetting, psfy)
    
    sx_start, sy_start = interpmax(camera, arg_sky, sky)
    
    #i, j = arg_sky
    #yslit, xslit = (
    #    int(camera.specs['slit_deltay'] * camera.upscale_f.y / camera.specs['mask_deltay'] + 5),
    #    int(camera.specs['slit_deltax'] * camera.upscale_f.x / camera.specs['mask_deltax'] + 5),
    #)
    #labels = np.zeros(camera.shape_sky)
    #labels[i - yslit : i + yslit + 1 , j - xslit : j + xslit + 1] = 1
    #i_cm, j_cm = center_of_mass(sky, labels=labels, index=1)
    #sx0, sy0 = camera.bins_sky.x[0], camera.bins_sky.y[0]
    ypxdim, xpxdim = (
        camera.specs['mask_deltay'] / camera.upscale_f.y,
        camera.specs['mask_deltax'] / camera.upscale_f.x,
    )
    #sx_start, sy_start = sx0 + xpxdim * j_cm, sy0 + ypxdim * i_cm
    
    #fluence_start = sky[*shift2pos(camera, sx_start, sy_start)]
    fluence_start = sky[*arg_sky]
    print(
        f"\nFLUENCE START: {fluence_start}\n"
        f"SHIFTS START: {sx_start, sy_start}\n" #, pos: {i_cm, j_cm}\n"
        f"{arg_sky=}, fluence arg_sky: {sky[*arg_sky]}\n"
    )
    loss = _Loss(model_shift_flux)
    results = minimize(
        lambda args: loss((args[0], args[1], args[2]), sky, arg_sky, camera),
        x0=np.array((sx_start, sy_start, fluence_start)),
        method="Nelder-Mead",
        bounds=[
            (
                max(sx_start - 5 * xpxdim, camera.bins_sky.x[0]),
                min(sx_start + 5 * xpxdim, camera.bins_sky.x[-1]),
            ),
            (
                max(sy_start - 5 * ypxdim, camera.bins_sky.y[0]),
                min(sy_start + 5 * ypxdim, camera.bins_sky.y[-1]),
            ),
            (0.9 * fluence_start, 1.1 * fluence_start),
        ],
        options={
            "xatol": 1e-6,
        },
    )
    # store the final optimized positions and fluence.
    sx, sy, fluence = map(float, results.x[:3])
    print(
        f"FINAL OPTIMIZED FLUENCE: {fluence}\n"
        f"FLUENCE GAIN: {(fluence - fluence_start) * 100 / fluence_start:.3f}%\n"
        f"FINAL OPTIMIZED SHIFTS: {sx, sy}\n"
        f"SHIFTX GAIN: {(sx - sx_start) * 100 / sx_start:.3f}%\n"
        f"SHIFTY GAIN: {(sy - sy_start) * 100 / sy_start:.3f}%\n"
    )
    # releases model cache memory.
    model_shift_flux_clear()
    return sx, sy, fluence


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
        ## account for low-counts level and non-Poisson distr. (assuming Poisson if > 25 counts / px)
        #skymap_ = skymap[(varmap > 25)]

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
#####    first_loop = iros_singleCAM(
#####        skymap=skymap,
#####        varmap=varmap,
#####        camera=camera,
#####        max_iterations=max_iterations,
#####        snr_threshold=SMOOTHING_THRESH,
#####        vignetting=vignetting,
#####        psfy=psfy,
#####    )
#####    candidates = tuple(c for c, _ in tqdm(first_loop))
    
    # perform detector smoothing and run again IROS on the processed data;
    # to do that, we first remove the stored sources from the original
    # detector, and then we perform the smoothing
    def callback(output: tuple[float]) -> tuple[float]:
        """Manage IROS candidate output parameters."""
        sx, sy, f, signf = output
        df = np.sqrt(f)
        return sx, DSX, sy, DSY, f, df, signf
    
#####    def retrieve_detector(candidates: tuple[tuple[float]]) -> NDArray:
#####        """Generates detector image from retrieved candidates."""
#####        img = np.zeros(camera.shape_detector)
#####        for (sx, sy, f, _) in candidates:
#####            shadowgram = model_shadowgram(
#####                camera=camera,
#####                shift_x=sx,
#####                shift_y=sy,
#####                vignetting=vignetting,
#####                psfy=psfy,
#####            )
#####            img += (f * shadowgram)
#####        return img
#####
#####    smoothed_res_detector = bkg_smoothing(
#####        detector=detector - retrieve_detector(candidates),
#####        camera=camera,
#####    )
#####    smoothed_skymap = decode(
#####        camera,
#####        np.clip(detector - smoothed_res_detector, a_min=0.0, a_max=detector.sum()),
#####    )
#####    print("# Initializing second loop with smoothed detector...")
    second_loop = iros_singleCAM(
        skymap=skymap,
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