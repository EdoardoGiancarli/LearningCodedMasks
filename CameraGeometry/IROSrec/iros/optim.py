"""
Base IROS algorithm implementation, with default steps operations.
"""

from typing import Any, Callable, Iterable
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from bloodmoon.coords import pos2shift
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import decode
from bloodmoon.mask import variance
from bloodmoon.mask import snratio
from bloodmoon.optim import model_shadowgram
from bloodmoon.optim import model_sky

from .types import OptResult
from .types import Source


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
    return OptResult(popt, errs)


def default_optimiser(
    camera: CodedMaskCamera,
    vignetting: bool | Callable[[CodedMaskCamera, NDArray, float, float], NDArray],
    psfy: bool | Callable[[CodedMaskCamera, NDArray], NDArray],
    fit_weights: NDArray | Callable[[NDArray], NDArray] | None = None,
    camera_coding_power: float = 0.85,
    verbose: bool = False,
) -> Callable[[NDArray, tuple[int, int]], OptResult]:
    """
    Configures the IROS optimiser for source parameters fitting.
    """
    def process_skyimg(
        sky: NDArray,
        pos: tuple[int, int],
    ) -> NDArray:
        """
        Processes the sky image for optimisation.
        """
        # here we crop the source PSF slit plus an offset to account for
        # shifts and to accomodate the `curve_fit` optimisation:
        #   - along y (coarse dir) we insert an offset of `5 * upscaling`, which
        #     is ~ 1/6 of the upsampled PSF slit dimension;
        #   - along x (fine dir) we insert an offset of `2 + upscaling`, to
        #     avoid the bkg contributes from the surrounding pixels;
        # NOTE: if the offset is smaller than at least the shifts bounds in
        # `optimize()`, the optimisation procedure may fail for some sources
        psf_slit_px_y, psf_slit_px_x = (
            int(camera.specs.slit_deltay * camera.upscale_f.y / camera.specs.mask_deltay),
            int(camera.specs.slit_deltax * camera.upscale_f.x / camera.specs.mask_deltax),
        )
        offset_y, offset_x = (
            5 * camera.upscale_f.y, 2 + camera.upscale_f.x,
        )
        crop_y, crop_x = (
            psf_slit_px_y + offset_y, psf_slit_px_x + offset_x,
        )
        i, j = pos
        slice_y, slice_x = (
            slice(i - crop_y, i + crop_y + 1), slice(j - crop_x, j + crop_x + 1),
        )
        cropped = sky[slice_y, slice_x]
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
            sky_peak / camera_coding_power if psfy is not False else sky_peak
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




def default_finder(
    camera: CodedMaskCamera,
    snr_threshold: float,
    batch: int = 1000,
) -> Callable[[NDArray, NDArray], tuple[int, int] | bool]:
    """
    Defines default IROS source candidates finder.
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
    
    return finder


def default_fitter(
    optimiser: Callable[[NDArray, tuple[int, int]], OptResult],
) -> Callable[[tuple[int, int], NDArray, NDArray], Source]:
    """
    Defines default IROS source candidates parameters fitter.
    """
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
            )
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}") from e
        
        significance = float(snr[*arg_sky])
        return Source(*params.params, significance)
    
    return fitter


def default_subtractor(
    camera: CodedMaskCamera,
    vignetting: bool | Callable[[CodedMaskCamera, NDArray, float, float], NDArray],
    psfy: bool | Callable[[CodedMaskCamera, NDArray], NDArray],
) -> Callable[[Source, NDArray], NDArray]:
    """
    Defines default IROS source shadowgram subtractor.
    """
    def subtractor(
        candidate: Source,
        detector: NDArray,
    ) -> NDArray:
        """Subtracts candidate from detector image."""
        sg_model: NDArray = model_shadowgram(
            camera=camera,
            shift_x=candidate.shift_x,
            shift_y=candidate.shift_y,
            vignetting=vignetting,
            psfy=psfy,
        )
        residual = detector - candidate.fluence * sg_model
        return residual
    
    return subtractor


def set_func(
    fn: Callable | None,
    default: Callable[[], Callable],
    *args: Any,
    **kwargs: Any,
) -> Callable:
    """Factory function configuration."""
    if fn is not None:
        return fn
    return default(*args, **kwargs)


def iros_singleCAM(
    camera: CodedMaskCamera,
    detector: NDArray,
    max_iterations: int,
    snr_threshold: float = 5.0,
    vignetting: bool | Callable[[CodedMaskCamera, NDArray, float, float], NDArray] = True,
    psfy: bool | Callable[[CodedMaskCamera, NDArray], NDArray] = True,
    fit_weights: NDArray | Callable[[NDArray], NDArray] | None = None,
    finder: Callable[[NDArray, NDArray], tuple[int, int] | bool] | None = None,
    fitter: Callable[[tuple[int, int], NDArray, NDArray], Source] | None = None,
    subtractor: Callable[[Source, NDArray], NDArray] | None = None,
    varmap: NDArray | None = None,
    optimiser: Callable[[NDArray, tuple[int, int]], OptResult] | None = None,
) -> Iterable[tuple[Source, NDArray]]:
    """
    Performs the Iterative Removal of Sources (IROS) algorithm on the collected data for
    a single coded-mask camera of the LEM-X observatory.

    This function implements an iterative source detection and removal procedure.
    For each iteration, it:
    1. Ranks source candidates by peak intensity
    2. Validates candidates by significance
    3. Fits source parameters
    4. Removes fitted source from the detector image
    5. Repeats until no significant sources remain or max iterations reached

    Args:
        detector (NDArray):
            Encoded sky-fields detector image.
        camera (CodedMaskCamera):
            CodedMaskCamera instance containing mask/detector geometry and parameters.
        max_iterations (int):
            Maximum number of source removal iterations to perform (default to 40 for precaution).
        snr_threshold (float, optional (default=`5.0`):
            If provided, iteration stops when maximum residual SNR falls below this value.
        vignetting (bool, Callable, optional (default=`True`):
            If `True`, the model used for optimization will simulate vignetting.
        psfy (bool, Callable, optional (default=`True`):
            If `True`, the model used for optimization will simulate
            detector position reconstruction effects.
        varmap (NDArray | None, optional (default=`None`):
            Variance map of the encoded sky-fields for significance maps computations. If `None`,
            the sky variance will be computed automathically from the input detector image.

    Yields:
        output (tuple[Source, NDArray]):
            - candidate (Source):
                Source candidate obj with local-frame sky-shift coords, fluence and significance.
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

    Examples:
    >>> for cand, residual in iros(detector, camera, max_iterations=2):
    >>>     # do your magic here
    >>>     ...
    """
    # intern logic setup
    find_candidate = set_func(finder, default_finder, camera, snr_threshold)
    optimise = set_func(
        optimiser, default_optimiser, camera, vignetting, psfy, fit_weights, verbose=True,
    )
    fit_cand_params = set_func(fitter, default_fitter, optimise)
    subtract_cand_sg = set_func(subtractor, default_subtractor, camera, vignetting, psfy)
    # arrs setup
    detector_ = detector.copy()
    skymap = decode(camera, detector)
    varmap = (
        varmap if varmap is not None else variance(camera, detector)
    )
    # looping as there's no tomorrow
    for i in range(max_iterations):
        snrmap = snratio(skymap, varmap)
        candidate_pos = find_candidate(skymap, snrmap)

        if not candidate_pos:
            print("\nNo candidates left...")
            break
        try:
            source = fit_cand_params(candidate_pos, skymap, snrmap)
        except RuntimeError as e:
            warnings.warn(f"Optimizer failed at iteration {i}:\n\n{e}")
            continue

        detector_ = subtract_cand_sg(source, detector_)
        skymap = decode(camera, detector_)
        yield (source, skymap)


# end