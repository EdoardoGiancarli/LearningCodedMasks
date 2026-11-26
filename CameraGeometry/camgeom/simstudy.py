"""
Module for WISEMAN simulation analyses focusing on LEM-X camera geometric and/or thermo-elastic misalignments.
"""

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from bloodmoon.coords import pos2shift
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.optim import model_sky
from darksun.benchmarking import crop_source_psf

from .types import OptResult


# -----------   source fit    -----------

def solver(
    func: Callable[[NDArray, *tuple[float, ...]], Any],
    xdata: NDArray,
    ydata: NDArray,
    verbose: bool = False,
    **kwargs: Any,
) -> OptResult:
    """
    Defines the fit solver for the source model parameters optimisation.
    The optimisation routine is performed with scipy's `curve_fit`.
    """
    popt, pcov, info, msg, iflag = curve_fit(
        f=func,
        xdata=xdata,
        ydata=ydata,
        full_output=True,
        **kwargs,
    )
    errs = np.sqrt(np.diag(pcov))
    if verbose:
        def comp_param_gain(start: float, optm: float) -> float:
            """Computes the optimised parameter value gain wrt start."""
            if start:
                return float(np.sign(start) * (optm - start) * 100 / start)
            return float(optm)

        routine_status = [
            'Convergence in chi-square values',
            'Convergence in parameter values',
            'Convergence in both chi-square and parameter values',
            'Convergence in orthogonality',
        ]
        print('\n## Optimisation Results:')
        start_vals = (
            kwargs['p0'] if 'p0' in kwargs.keys() else np.ones_like(popt)
        )
        for idx, (s, p, dp) in enumerate(zip(start_vals, popt, errs)):
            print(
                f'  * p[{idx}]:\n'
                f'      - START.: {float(s):.7f}\n'
                f'      - OPTIM.: {float(p):.7f} +/- {float(dp):.7f}\n'
                f'      - GAIN.: {comp_param_gain(s, p):.3f} %'
            )
        print(
            f'\n## Fit Report:\n'
            f'  - func calls (also # of iters): {info['nfev']}\n'
            f'  - procedure msg: {msg}\n'
            f'  - success status {iflag}: {routine_status[iflag - 1]}\n'
        )
    return OptResult(popt, pcov)


def config_optimiser(
    camera: CodedMaskCamera,
    vignetting: bool | Callable[[CodedMaskCamera, NDArray, float, float], NDArray],
    psfy: bool | Callable[[CodedMaskCamera, NDArray], NDArray],
    fit_weights: NDArray | Callable[[NDArray], NDArray] | None = None,
    cam_coding_power: float = 0.85,
    psf_crp_xyoff: tuple[int, int] = (0, 0),
    verbose: bool = False,
) -> Callable[[NDArray, tuple[int, int]], OptResult]:
    """
    Configures source params optimiser 
    """
    def process_skyimg(
        sky: NDArray,
        pos: tuple[int, int],
    ) -> NDArray:
        """
        Processes the sky image for optimisation.
        """
        rows, cols = crop_source_psf(camera, pos, *psf_crp_xyoff)
        cropped = sky[rows, cols]
        return cropped.flatten()

    def _ModelShiftFluence(arg_sky: tuple[int, int]) -> Callable[[NDArray, float, float, float], NDArray]:
        """
        Initialises the source model.
        """
        def f(x: NDArray, shift_x: float, shift_y: float, fluence: float) -> NDArray:
            """Models the source sky image."""
            modeled = model_sky(camera, shift_x, shift_y, fluence, vignetting, psfy)
            return process_skyimg(modeled, arg_sky)
        
        return f

    def optimiser(
        sky: NDArray,
        arg_sky: tuple[int, int],
    ) -> OptResult:
        """
        Performs the optimization to fit a point source model to sky image data.
        """
        model_func = _ModelShiftFluence(arg_sky)
        sky_peak = sky[*arg_sky]

        # setup solver params
        # * setup source data + std
        sky_ydata = process_skyimg(sky, arg_sky)
        if fit_weights is not None:
            weights = (
                fit_weights if isinstance(fit_weights, np.ndarray)
                else fit_weights(sky_ydata)
            )
        else:
            weights = np.ones_like(sky_ydata)
        # * extract source coords and counts starting values
        sx_start, sy_start = pos2shift(camera, *arg_sky)
        cts_start = (
            sky_peak / cam_coding_power if psfy is not False else sky_peak
        )
        start_params_vals = np.array([sx_start, sy_start, cts_start])
        # * setup fit params boundaries
        #    - the shifts are allowed to fluctuate in a small pixel box since
        #      the extracted position is close enough to the true source pos
        #      A small box also account for superimposed or close sources,
        #      which may introduce biases in the source fit procedure
        #    - the box is built from the digital upsampling since in the worst
        #      case (source at 45 deg wrt optical axis) the high energy detected
        #      photons median absorption distance is 225um. The projection of this
        #      distance on the camera plane is always smaller than `px_size / ups`  
        #    - the fluence cannot be smaller than the one observed at the peak,
        #      and we insert a lower value just for precaution (if simulating
        #      for example an infinite detector spatial resolution)
        px_dim_x, px_dim_y = (
            camera.specs.mask_deltax / camera.upscale_f.x,
            camera.specs.mask_deltay / camera.upscale_f.y,
        )
        bounds = [
            (
                max(sx_start - camera.upscale_f.x * px_dim_x, camera.bins_sky.x[0]),
                max(sy_start - camera.upscale_f.y * px_dim_y, camera.bins_sky.y[0]),
                0.95 * sky_peak,
            ),
            (
                min(sx_start + camera.upscale_f.x * px_dim_x, camera.bins_sky.x[-1]),
                min(sy_start + camera.upscale_f.y * px_dim_y, camera.bins_sky.y[-1]),
                1.4 * sky_peak,
            ),
        ]
        # perform optimisation
        slvr_kwargs = {
            # - solver kwargs
            'p0': start_params_vals,
            'sigma': weights,
            'bounds': bounds,
            'method': 'trf',
            # - least square kwargs
            'jac': '3-point',
            'ftol': 1e-8,
            'xtol': 1e-8,
            'x_scale': 'jac',
            'loss': 'linear',
        }
        result = solver(
            func=model_func,
            xdata=np.arange(len(sky_ydata)),
            ydata=sky_ydata,
            verbose=verbose,
            **slvr_kwargs,
        )

        return result
    
    return optimiser


def config_fitter(
    optimiser: Callable[[NDArray, tuple[int, int]], OptResult],
) -> Callable[[tuple[int, int], NDArray], tuple[NDArray, NDArray]]:
    """
    Configures source candidates parameters fitter.
    """
    def fitter(arg_sky: tuple[int, int], sky: NDArray) -> tuple[NDArray, NDArray]:
        """Performs the optimisation of the source candidate params."""
        try:
            params: OptResult = optimiser(
                sky=sky,
                arg_sky=arg_sky,
            )
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}") from e
        
        return params.params, params.covar
    
    return fitter



# end