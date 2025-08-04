from typing import Iterable
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from tqdm import tqdm

from bloodmoon.coords import angle2shift
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import count
from bloodmoon.mask import decode
from bloodmoon.mask import snratio
from bloodmoon.mask import variance
from bloodmoon.optim import optimize
from bloodmoon.optim import model_shadowgram
from bloodmoon.optim import model_sky

from darksun.types import LogEntry
from darksun.data import Log
from darksun.data import create_log
from darksun.data import DataLoader


def bkg_smoothing(
    detector: NDArray,
    camera: CodedMaskCamera,
    *,
    kernelsize_y: int = 11,
    kernelsize_x: int = 7,
) -> NDArray:
    """
    Performs a smoothing of the residual background from the coded-mask camera detector image.

    The smoothing is performed on the (y, x) axes independently. First, the detector image
    is collapsed along a direction, and then a 1D median filter is applied to remove the
    residual high frequencies on the collapsed array. The median filter is applied ignoring
    the detector sensitivity array zeroes to avoid boundary effects.

    The kernel has a default size along `(y, x)` of `(11 x 7)` at upscaling `(1, 1)`, equal
    to a physical size of `(11 * 0.4, 7 * 0.25) mm` for the Wide Field Monitor cameras.
    Inside the method, the kernel size is automatically adjusted to the camera upscaling. 

    This smoothing should be applied after removing the brightest sources from the original
    detector image (e.g., by processing the observed sky with the IROS algorithm).
    While the remaining (weaker) sources will be affected by the smoothing, it has been
    tested that their significance is reduced by a factor lower than `10%`, for both on-
    and off-axis sources with SNR between `5` and `50` sigmas.

    Args:
        detector (NDArray):
            Input coded-mask camera detector image.
        camera (CodedMaskCamera):
            Instance with detector geometry info.
        kernelsize_y (int, optional (default=`7`)):
            Kernel size along the y axis (upscaling 1).
        kernelsize_x (int, optional (default=`11`)):
            Kernel size along the x axis (upscaling 1).
    
    Returns:
        output (NDArray):
            Smoothed detector image. The array is rescaled to have
            the same counts of the original input detector image.
    
    ## Notes:
        - CFR with url: [`bkg_fitting_v3.ipynb`](
        https://github.com/yuri-evangelista/CodedMasks/blob/main/notebooks/bkg_fitting_v3.ipynb
        ).
    """
    # define median filter kernel size at given camera upscaling
    UPX, UPY = camera.upscale_f
    KERNEL_SIZE = {
        'y': int(kernelsize_y * UPY),
        'x': int(kernelsize_x * UPX),
    }
    
    def apply_filter(axis: int, size: int) -> NDArray:
        """
        Collapses the detector along the specified axis and
        applies a 1D median filter of the given size.
        """
        # collapse detector and bulk
        collapsed_det = detector.sum(axis=axis)
        collapsed_bulk = camera.bulk.sum(axis=axis)
        # bulk zeros are ignored to avoid boundary effects
        bulk_mask = (collapsed_bulk > 0)
        filtered = collapsed_det.copy()
        filtered[bulk_mask] = median_filter(
            collapsed_det[bulk_mask], size=size, mode='nearest',
        )
        return filtered
    
    # apply filter along the two axes independently
    # ! the smoothing is performed by collapsing the
    #   detector along the opposing direction
    smooth_y = apply_filter(axis=1, size=KERNEL_SIZE['y'])
    smooth_x = apply_filter(axis=0, size=KERNEL_SIZE['x'])

    # restore 2D profile through broadcasting (as suggested by np.tile doc)
    # - the smoothed array is masked with the bulk to remove artefacts
    # - the filtered array is rescaled to conserve the original counts
    smoothed = smooth_y[:, np.newaxis] * smooth_x[np.newaxis, :]
    smoothed *= (camera.bulk > 0)
    smoothed *= (detector.sum() / smoothed.sum())

    return smoothed


def iros_singleCAM(
    detector: NDArray,
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
    
    skymap = decode(camera, detector)
    varmap = variance(camera, detector)
    for i in range(max_iterations):
        snrmap = snratio(skymap, np.clip(varmap, a_min=1, a_max=None))
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

    # performing IROS to remove the brightest sources (SNR > SMOOTHING_THRESH)
    print("# Running first loop...")
    first_loop = iros_singleCAM(
        detector=detector,
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
    def subtract(
        detector: NDArray,
        candidates: tuple[tuple[float]],
    ) -> NDArray:
        """Subtracts retrieved sources from original detector."""
        residual = detector.copy()
        for (sx, sy, f) in candidates:
            shadowgram = model_shadowgram(
                camera=camera,
                shift_x=sx,
                shift_y=sy,
                fluence=f,
                vignetting=vignetting,
                psfy=psfy,
            )
            residual -= shadowgram
        return residual

    residual_detector = subtract(detector, candidates)
    smoothed_detector = bkg_smoothing(residual_detector, camera)      # hmmm, wouldn't the remaining sources be subtracted in this way?
    print("# Initializing second loop with smoothed detector...")
    second_loop = iros_singleCAM(
        detector=detector - smoothed_detector,
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