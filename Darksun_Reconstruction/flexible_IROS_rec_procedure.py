"""
Temporary module for new IROS reconstruction procedure.
In this version, the sub-logics are made flexible by allowing for customisation.
The IROS routine is now intended as a "wrapper" for the main logics (source finding process, parameters fitting and source subtraction).
"""

# search for updated versions of:
#   - finder, fitter, subtractor methods
#   - optimiser method
#
# NOTE: inside the finder there is the sky pos masking
# NOTE: the optimiser is called inside the fitter
#
# NOTE (optimiser): custom obj for `curve_fit` output for `verbose` func input
# NOTE (optimiser): general custom obj for optimising procedure? Some scipy routine have their own output obj...
# NOTE (optimiser): make `verbose` func flexible by again giving it as input


from typing import Any, Callable, Iterable, NamedTuple
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from bloodmoon.coords import pos2shift
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import decode
from bloodmoon.mask import variance
from bloodmoon.mask import snratio
from bloodmoon.optim import process_skyimg


class OptResult(NamedTuple):
    """Optimisation results."""
    params: NDArray
    covar: NDArray

class Source(NamedTuple):
    """
    Source candidate parameters container.
    """
    shift_x: float
    shift_y: float
    cts: float
    snr: float




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
            return float(np.sign(start) * (optm - start) * 100 / start)

        routine_status = [
            'Convergence in chi-square values',
            'Convergence in parameter values',
            'Convergence in both chi-square and parameter values',
            'Convergence in orthogonality',
        ]
        print('## Optimisation Results:')
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
    return OptResult(popt, errs)


def config_optimiser(
    camera: CodedMaskCamera,
    model_initialiser: Callable[[tuple[int, int]], Callable[[NDArray, *tuple[float, ...]], NDArray]],
    fit_weights: NDArray | Callable[[NDArray], NDArray] | None = None,
    camera_coding_power: float = 0.85,
) -> Callable[[NDArray, tuple[int, int], bool], OptResult]:
    """
    Configures the IROS optimiser for source parameters fitting.
    """
    def optimise(
        sky: NDArray,
        arg_sky: tuple[int, int],
        verbose: bool = False,
    ) -> OptResult:
        """
        Performs the optimization to fit a point source model to sky image data.
        """
        model_func = model_initialiser(arg_sky)
        sky_peak = sky[*arg_sky]

        # setup solver params
        # * setup source data + std
        sky_ydata = process_skyimg(camera, sky, arg_sky)
        if fit_weights is not None:
            weights = (
                fit_weights if isinstance(fit_weights, np.ndarray)
                else fit_weights(sky_ydata)
            )
        else:
            weights = np.ones_like(sky_ydata)
        # * extract source coords and counts starting values
        sx_start, sy_start = pos2shift(camera, *arg_sky)
        cts_start = sky_peak / camera_coding_power
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
                1.25 * sky_peak,
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
    
    return optimise




def config_IROS_operations(
    camera: CodedMaskCamera,
    src_sg_model: Callable[[CodedMaskCamera, float, float], NDArray],
    optimiser: Callable[[NDArray, tuple[int, int], bool], OptResult],
    snr_threshold: float = 0.0,
) -> tuple[
    Callable[[NDArray, NDArray, int], tuple[int, int] | bool],
    Callable[[tuple[int, int], NDArray, NDArray], Source],
    Callable[[Source, NDArray], NDArray],
]:
    """
    Configures the operative funcs (source finder, fit and subtractor) for the IROS procedure.
    """
    SETUP: dict[str, Any] = {
        'slit_mask_fine': int(
            camera.specs.slit_deltax * camera.upscale_f.x / camera.specs.mask_deltax
        ),
        'slit_mask_coarse': int(
            camera.specs.slit_deltay * camera.upscale_f.y / camera.specs.mask_deltay
        ),
        'skymap_mask': np.ones(camera.shape_sky, dtype=int),
    }

    def _update_skymap_mask(arg_sky: tuple[int, int]) -> None:
        """
        Updates the skymap mask with the new candidate position by covering the candidate
        half-PSF profile (to account for the camera angular resolution of a source).
        """
        rows = slice(
            arg_sky[0] - SETUP['slit_mask_coarse'] // 2, arg_sky[0] + SETUP['slit_mask_coarse'] // 2 + 1,
        )
        cols = slice(
            arg_sky[1] - SETUP['slit_mask_fine'] // 2, arg_sky[1] + SETUP['slit_mask_fine'] // 2 + 1,
        )
        SETUP['skymap_mask'][rows, cols] = 0
        return None
    
    def finder(
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
        for arg_sky in reservoir[::-1]:
            if (snr[*arg_sky] > snr_threshold) and SETUP['skymap_mask'][*arg_sky]:
                _update_skymap_mask(arg_sky)
                return tuple(arg_sky)
        return False
    
    def fitter(
        arg_sky: tuple[int, int],
        sky: NDArray,
        snr: NDArray,
    ) -> Source:
        """Performs the optimisation of the source candidate params."""
        try:
            params: OptResult = optimiser(
                sky=sky,
                arg_sky=arg_sky,
                verbose=True,
            )
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}") from e
        
        significance = float(snr[*arg_sky])
        return Source(*params.params, significance)
    
    def subtractor(
        candidate: Source,
        detector: NDArray,
    ) -> NDArray:
        """Subtracts candidate from detector image."""
        sg_model: NDArray = src_sg_model(
            camera=camera,
            shift_x=candidate.shift_x,
            shift_y=candidate.shift_y,
        )
        residual = detector - candidate.cts * sg_model
        return residual
    
    return finder, fitter, subtractor




def iros_singleCAM(
    detector: NDArray,
    camera: CodedMaskCamera,
    max_iterations: int = 5,
    finder: Callable[[NDArray, NDArray], tuple[int, int] | bool] | None = None,
    fitter: Callable[[tuple[int, int], NDArray, NDArray], Source] | None = None,
    subtractor: Callable[[Source, NDArray], NDArray] | None = None,
    varmap: NDArray | None = None,
) -> Iterable[tuple[Source, NDArray]]:
    """
    Performs the Iterative Removal of Sources (IROS) algorithm for a single coded-mask
    camera of the Wide Field Monitor observations.
    """
    # arrs setup
    detector_ = detector.copy()
    skymap = decode(camera, detector)
    varmap = (
        varmap if varmap is not None
        else variance(camera, detector)
    )
    # looping as there's no tomorrow
    for i in range(max_iterations):
        snrmap = snratio(skymap, varmap)
        candidate_pos = finder(skymap, snrmap)

        if not candidate_pos:
            print("\nNo candidates left...")
            break
        try:
            source = fitter(candidate_pos, skymap, snrmap)
        except RuntimeError as e:
            warnings.warn(f"Optimizer failed at iteration {i}:\n\n{e}")
            continue

        detector_ = subtractor(source, detector_)
        skymap = decode(camera, detector_)
        yield (source, skymap)


# end