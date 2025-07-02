from functools import lru_cache
from typing import Callable, Iterable, Literal
import warnings

from numpy import typing as npt
import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve

from mbloodmoon.coords import shift2pos

from mbloodmoon.images import _erosion
from mbloodmoon.images import _rbilinear
from mbloodmoon.images import _rbilinear_relative
from mbloodmoon.images import _shift
from mbloodmoon.images import argmax
from mbloodmoon.io import SimulationDataLoader
from mbloodmoon.mask import _bisect_interval
from mbloodmoon.mask import _detector_footprint
from mbloodmoon.mask import CodedMaskCamera
from mbloodmoon.mask import count
from mbloodmoon.mask import cutout
from mbloodmoon.mask import decode
from mbloodmoon.mask import interpmax
from mbloodmoon.mask import snratio
from mbloodmoon.mask import variance

from mbloodmoon.optim import _ModelShiftFluence, _ModelShiftFluenceUncached, _Loss, model_shadowgram


def _Loss(model_f: Callable) -> Callable:  # noqa
    """
    Returns a loss function for source parameter optimization, given a routine for computing models.

    Args:
        model_f: Callable that generates model predictions. Expected to have signature:
            model_f(shift_x: float, shift_y: float, fluence: float, camera: CodedMaskCamera) -> np.array

    Returns:
        Callable that computes the loss with signature:
            f(args: np.array, truth: np.array, camera: CodedMaskCamera) -> float
        where:
            - args is [shift_x, shift_y, fluence]
            - truth is the observed sky image
    """

    def f(args: npt.NDArray, truth: npt.NDArray, pos: tuple[int, int], camera: CodedMaskCamera) -> float:
        """
        Compute MSE loss between model prediction and truth.

        Args:
            args: Array of [shift_x, shift_y, fluence] parameters to evaluate
            truth: Full observed sky image to compare against
            camera: CodedMaskCamera instance containing geometry information
                    No need for this, but we take the parameter for compatibility with
                    optimization model interfaces.

        Returns:
            float: Mean Squared Error between model and truth in local window
        """
        from mbloodmoon.iros_management.show import crop
        upx, upy = camera.upscale_f
        cutx, cuty = (
            int(camera.specs["slit_deltax"] * upx / camera.specs["mask_deltax"] + 5),
            int(camera.specs["slit_deltay"] * upy / camera.specs["mask_deltay"] + 5),
        )

        model = model_f(*args)
        mse = np.mean(
            np.square(
                crop(model - truth, pos, (cuty, cutx), False)
            )
        )
        return float(mse)

    return f


def optimize(
    camera: CodedMaskCamera,
    sky: npt.NDArray,
    arg_sky: tuple[int, int],
    vignetting: bool = True,
    psfy: bool = True,
    model: Literal["fast", "accurate"] = "fast",
) -> tuple[float, float, float]:
    """
    Perform two-stage optimization to fit a point source model to sky image data.

    This function performs a two-stage optimization:
    1. Coarse optimization of fluence only, keeping position fixed
    2. Fine, simultaneous optimization of position and fluence.
       This step is warm-started with the flux value inferred from the coarse step.

    The process uses different model at each stage to balance speed and accuracy.

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
    if model == "fast":
        model_shift_flux, model_shift_flux_clear = _ModelShiftFluence(camera, vignetting, psfy)
    elif model == "accurate":
        model_shift_flux, model_shift_flux_clear = _ModelShiftFluenceUncached(camera, vignetting, psfy)
    else:
        raise ValueError("Model value not supported. The `model` arguments should be `fast` or `accurate`.")
    
    sx_start, sy_start = interpmax(camera, arg_sky, sky)
    fluence_start = sky[*arg_sky]
    loss = _Loss(model_shift_flux)
    results = minimize(
        lambda args: loss((args[0], args[1], args[2]), sky, arg_sky, camera),
        x0=np.array((sx_start, sy_start, fluence_start)),
        method="Nelder-Mead",
        bounds=[
            (
                max(sx_start - camera.mdl["slit_deltax"], camera.bins_sky.x[0]),
                min(sx_start + camera.mdl["slit_deltax"], camera.bins_sky.x[-1]),
            ),
            (
                max(sy_start - camera.mdl["slit_deltay"], camera.bins_sky.y[0]),
                min(sy_start + camera.mdl["slit_deltay"], camera.bins_sky.y[-1]),
            ),
            (0.9 * fluence_start, 1.1 * fluence_start),
        ],
        options={
            "xatol": 1e-6,
        },
    )
    # store the final optimized positions and fluence.
    sx, sy, fluence = map(float, results.x[:3])
    # releases model cache memory.
    model_shift_flux_clear()
    return sx, sy, fluence



# TODO:
#   - distance up to top mask
#   - remove `-1 *` from `red_factor` in `apply_vignetting()`