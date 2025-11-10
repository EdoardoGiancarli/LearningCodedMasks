"""
Optimise procedure for IROS source parameters fitting with scipy `least_squares()`.
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray 
from scipy.optimize import least_squares

from bloodmoon.coords import pos2shift
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import cutout
from bloodmoon.optim import _ModelShiftFluence #, _Loss


def _Loss(model_f: Callable) -> Callable:
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

    def f(
        args: NDArray,
        truth: NDArray,
        pos: tuple[int, int],
        camera: CodedMaskCamera,
    ) -> float:
        """
        Compute MSE loss between model prediction and truth.

        Args:
            args: Array of [shift_x, shift_y, fluence] parameters to evaluate
            truth: Full observed sky image to compare against
            pos: the (row, col) indexes of the slice center.
            camera: CodedMaskCamera instance containing geometry information
                    No need for this, but we take the parameter for compatibility with
                    optimization model interfaces.

        Returns:
            float: Mean Squared Error between model and truth in local window
        """
        #def process_img(arr: NDArray) -> NDArray:
        #    """
        #    Centers and normalises the input array in the range [-1, 1].
        #    """
        #    centered = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
        #    norm = (arr - np.min(centered))/(np.max(centered) - np.min(centered))
        #    return 2 * norm - 1

        (min_i, max_i, min_j, max_j), _ = cutout(camera, pos, fx=2, fy=2)
        model = model_f(*args)
        residual = model - truth
        metric = np.mean(np.square(residual[min_i:max_i, min_j:max_j]))
        return float(metric)

    return f


def optimize(
    camera: CodedMaskCamera,
    sky: NDArray,
    arg_sky: tuple[int, int],
    vignetting: bool = True,
    psfy: bool = True,
) -> tuple[float, float, float]:
    """
    Performs the optimization to fit a point source model to sky image data
    with the use of scipy's `least_squares()`.

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
    model_shift_flux = _ModelShiftFluence(camera, vignetting, psfy)
    loss = _Loss(model_shift_flux)
    px_dim_x, px_dim_y = (
        camera.specs.mask_deltax / camera.upscale_f.x,
        camera.specs.mask_deltay / camera.upscale_f.y,
    )

    sx_start, sy_start = pos2shift(camera, *arg_sky)
    fluence_start = sky[*arg_sky] / 0.9                 # camera coding power (Skinner et al. 2008)

    results = least_squares(
        lambda args: loss((args[0], args[1], args[2]), sky, arg_sky, camera),
        x0=np.array((sx_start, sy_start, fluence_start)),
        bounds=[
            (
                max(sx_start - 3 * px_dim_x, camera.bins_sky.x[0]),
                max(sy_start - 3 * px_dim_y, camera.bins_sky.y[0]),
                sky[*arg_sky],
            ),
            (
                min(sx_start + 3 * px_dim_x, camera.bins_sky.x[-1]),
                min(sy_start + 3 * px_dim_y, camera.bins_sky.y[-1]),
                sky[*arg_sky] / 0.8,
            ),
        ],
        xtol=1e-7,
        ftol=1e-6,
        x_scale='jac',
    )
    # store the final optimized positions and fluence.
    sx, sy, fluence = map(float, results.x[:3])

    # optimization verbose
    print(
        f'\n'
        f'## Optimisation Results:\n'
        f'  - fluence START: {fluence_start}\n'
        f'  - shifts START (x, y): {sx_start}, {sy_start}\n'

        f'  - fluence OPTIM.: {fluence}\n'
        f'  - shifts OPTIM. (x, y): {sx}, {sy}\n'

        f'  - fluence GAIN %: {(fluence - fluence_start) * 100 / fluence_start:.3f}\n'
        f'  - shift_x GAIN %: {(sx - sx_start) * 100 / sx_start:.3f}\n'
        f'  - shift_y GAIN %: {(sy - sy_start) * 100 / sy_start:.3f}\n'
    )

    return sx, sy, fluence


# end