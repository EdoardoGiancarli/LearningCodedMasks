"""
Module with new optimiser based on scipy's `curve_fit`.
"""

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from bloodmoon.coords import pos2shift
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.optim import model_sky


def process_skyimg(
    camera: CodedMaskCamera,
    sky: npt.NDArray,
    pos: tuple[int, int],
) -> npt.NDArray:
    """
    Processes the sky image for optimisation.
    """
    cropy, cropx = (
        int(camera.specs.slit_deltay * camera.upscale_f.y / camera.specs.mask_deltay) + 5,
        int(camera.specs.slit_deltax * camera.upscale_f.x / camera.specs.mask_deltax) + 7,
    )
    i, j = pos
    slicey, slicex = (
        slice(i - cropy, i + cropy + 1),
        slice(j - cropx, j + cropx + 1),
    )
    cropped = sky[slicey, slicex]
    return cropped.flatten()


def _ModelShiftFluence(
    camera: CodedMaskCamera,
    pos: tuple[int, int],
    vignetting: bool = True,
    psfy: bool = True,
) -> Callable[[npt.NDArray, float, float, float], npt.NDArray]:
    """
    A slow, vanilla implementation of the model for both direction and fluence optimization.
    Intended for debugging and benchmarking.

    Args:
        camera: CodedMaskCamera instance containing all geometric parameters
        pos: tuple of row, col indexes indicating the source peak position. The source
        sky image is cropped around `pos`.
        vignetting: If true, shadowgram model simulates vignetting.
        psfy: If true, the model used for optimization will simulate detector position
        reconstruction effects.

    Returns:
        A Callable, which is the routine for computing the model.
    """

    def f(x: npt.NDArray, shift_x: float, shift_y: float, fluence: float) -> npt.NDArray:
        """
        A simple, slow version of the model for both direction and fluence optimization.
        The input `x` represents an independent variable, and it has only been inserted
        to match the inputs of the scipy `curve_fit` procedure.

        Args:
            x: Placeholder for independent variable as in `curve_fit` doc
            shift_x: Source position x-coordinate in sky-shift space (mm)
            shift_y: Source position y-coordinate in sky-shift space (mm)
            fluence: Source intensity/fluence value

        Returns:
            Flattened and cropped 2D source-modeled sky image
        """
        modeled = model_sky(camera, shift_x, shift_y, fluence, vignetting, psfy)
        return process_skyimg(camera, modeled, pos)
    
    return f


def optimize(
    camera: CodedMaskCamera,
    sky: npt.NDArray,
    arg_sky: tuple[int, int],
    vignetting: bool = True,
    psfy: bool = True,
    verbose: bool = True,
) -> tuple[float, float, float]:
    """
    Performs the optimization to fit a point source model to sky image data.

    This function performs the optimization by simultaneously fit the candidate
    position and fluence. The starting position is inferred from the candidate
    pixel position, while the starting fluence is represented by the counts at
    the candidate extracted pixel indexes.

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
        - Bounds are set based on initial guess and physical constraints
    """
    px_dim_x, px_dim_y = (
        camera.specs.mask_deltax / camera.upscale_f.x,
        camera.specs.mask_deltay / camera.upscale_f.y,
    )

    model_shift_flux = _ModelShiftFluence(camera, arg_sky, vignetting, psfy)
    sx_start, sy_start = pos2shift(camera, *arg_sky)
    sky_peak = sky[*arg_sky]
    fluence_start = (
        sky_peak / 0.85 if psfy else sky_peak
    )
    sky_ydata = process_skyimg(camera, sky, arg_sky)
    
    results, _ = curve_fit(
        model_shift_flux,
        xdata=np.arange(len(sky_ydata)),
        ydata=sky_ydata,
        p0=[sx_start, sy_start, fluence_start],
        bounds=[
            (
                max(sx_start - 1.5 * px_dim_x, camera.bins_sky.x[0]),
                max(sy_start - 1.5 * px_dim_y, camera.bins_sky.y[0]),
                sky_peak,
            ),
            (
                min(sx_start + 1.5 * px_dim_x, camera.bins_sky.x[-1]),
                min(sy_start + 1.5 * px_dim_y, camera.bins_sky.y[-1]),
                1.25 * sky_peak,
            ),
        ],
    )
    # store the final optimized positions and fluence
    sx, sy, fluence = map(float, results)

    if verbose:
        print(
            f'\n'
            f'## Optimisation Results:\n'
            f'  - fluence START: {fluence_start}\n'
            f'  - shifts START (x, y): {sx_start}, {sy_start}\n'

            f'  - fluence OPTIM.: {fluence}\n'
            f'  - shifts OPTIM. (x, y): {sx}, {sy}\n'

            f'  - fluence GAIN %: {(fluence - fluence_start) * 100 / fluence_start:.3f}\n'
            f'  - shift_x GAIN %: {np.sign(sx_start) * (sx - sx_start) * 100 / sx_start:.3f}\n'
            f'  - shift_y GAIN %: {np.sign(sy_start) * (sy - sy_start) * 100 / sy_start:.3f}\n'
        )

    return sx, sy, fluence


# end