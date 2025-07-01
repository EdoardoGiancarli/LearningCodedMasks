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


def optimize_base_tf(
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
    # initialize the function to fine coarse, fluence and position dependent shadowgram model.
    # this is slower to compute and requires more memory. again it leverages caches to reduce
    # the number of cross-correlation computations, and it is our responsibility to free
    # memory after we will be done.
    if model == "fast":
        model_shift_flux, model_shift_flux_clear = _ModelShiftFluence(camera, vignetting, psfy)
    elif model == "accurate":
        model_shift_flux, model_shift_flux_clear = _ModelShiftFluenceUncached(camera, vignetting, psfy)
    else:
        raise ValueError("Model value not supported. The `model` arguments should be `fast` or `accurate`.")
    
    sx_start, sy_start = interpmax(camera, arg_sky, sky)
    fluence_start = sky[*arg_sky] # sky.max()
    print(
        f"\nFLUENCE START: {fluence_start}\n"
        f"SHIFTS START: {sx_start, sy_start}\n"
    )
    loss = _Loss(model_shift_flux)
    results = minimize(
        lambda args: loss((args[0], args[1], args[2]), sky, camera),
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


def optimize_tf4(
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
    # initialize the function to fine coarse, fluence and position dependent shadowgram model.
    # this is slow to compute and requires more memory. again it leverages caches to reduce
    # the number of cross-correlation computations, and it is our responsibility to free
    # memory after we will be done.
    if model == "fast":
        model_shift_flux, model_shift_flux_clear = _ModelShiftFluence(camera, vignetting, psfy)
    elif model == "accurate":
        model_shift_flux, model_shift_flux_clear = _ModelShiftFluenceUncached(camera, vignetting, psfy)
    else:
        raise ValueError("Model value not supported. The `model` arguments should be `fast` or `accurate`.")
    
    sx_start, sy_start = interpmax(camera, arg_sky, sky)
    s_off = model_shadowgram(
        camera=camera,
        shift_x=sx_start,
        shift_y=sy_start,
        vignetting=vignetting,
        psfy=psfy,
    )
    s_on = model_shadowgram(
        camera=camera,
        shift_x=0.0,
        shift_y=0.0,
        vignetting=vignetting,
        psfy=psfy,
    )
    #fluence_start = sky[*arg_sky] # sky.max()
    fluence_start = sky[*arg_sky] * s_on.sum() / s_off.sum()     # take good initial fluence estimate
    print(
        f"\nFLUENCE START: {fluence_start}\n"
        f"SHIFTS START: {sx_start, sy_start}\n"
    )
    loss = _Loss(model_shift_flux)
    results = minimize(
        lambda args: loss((args[0], args[1], args[2]), sky, camera),
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
            (0.96 * fluence_start, 1.04 * fluence_start),        # fluence boundary to 4%
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