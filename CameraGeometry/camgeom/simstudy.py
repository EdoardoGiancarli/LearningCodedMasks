"""
Coded-mask camera geometric configuration analyser.
This module is intended for benchmarking the performance of a LEM-X camera accounting for assembling systematics,
such as mask and/or SDDs (or even whole detector plane) rotations and/or shifts with respect to the nominal design.
"""

from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm

from bloodmoon.coords import (
    pos2shift,
    shift2angle,
    equatorial2pos,
    equatorial2shift,
)
from bloodmoon.images import argmax
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import (
    count,
    decode,
    variance,
)
from bloodmoon.optim import model_sky
from bloodmoon.types import CoordEquatorial
import darksun as ds
from darksun.benchmarking import source_catalogue_data, crop_source_psf
from darksun.data import DataLoader, CatalogueLoader
from darksun.filtering import select_source_photons

from .types import OptResult


# ------   I/O operations   ------

def check_and_pick(parent: Path, pattern: str) -> Path:
    matches = tuple(parent.glob(pattern))
    if not matches:
        raise ValueError(f"A file matching the pattern {str(parent / pattern)} is expected but missing.")
    f, *extra_matches = matches
    if extra_matches:
        raise ValueError(
            f"Found unexpected extra matches for glob pattern {str(parent / pattern)}."
            f"File with pattern {pattern} should be unique"
        )
    return f

def save_to_csv(df: pd.DataFrame, save_to: str | Path, **kwargs) -> None:
    """Saves input DataFrame to CSV file."""
    df.to_csv(save_to, **kwargs)
    return

def load_from_csv(filepath: str | Path, index_col: int = 0, **kwargs) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    df = pd.read_csv(filepath, index_col=index_col, **kwargs)
    return df


# ------   coords handling   ------

def get_source_coords(sourceID: str, catalogue: CatalogueLoader) -> CoordEquatorial:
    """Extracts the source RA/Dec coords from catalogue."""
    data = source_catalogue_data(sourceID, catalogue.DLdata)
    return CoordEquatorial(data['RA'], data['DEC'])

def equatorial2angle(
    sdl: DataLoader,
    camera: CodedMaskCamera,
    ra: float,
    dec: float,
) -> tuple[float, float]:
    """Converts RA/Dec coords to camera local-frame angular coords."""
    shifts = equatorial2shift(sdl, camera, ra, dec)
    tx, ty = map(lambda s: shift2angle(camera, s), shifts)
    return tx, ty


# ------   array operations   ------

def find_boxmax(
    arr: NDArray,
    centre: tuple[int, int],
    boxsize: int | tuple[int, int] = 5,
) -> tuple[int, int]:
    """Finds argmax of given `arr` in box centered around `centre`."""
    br, bc = (boxsize, boxsize) if isinstance(boxsize, int) else boxsize
    p, q = centre
    r0, r1 = max(0, p - br), min(arr.shape[0], p + br + 1)
    c0, c1 = max(0, q - bc), min(arr.shape[1], q + bc + 1)
    n, m = argmax(arr[r0 : r1, c0 : c1])
    return r0 + int(n), c0 + int(m)


# ------   output data logging   ------

def srcdmap_to_df(dmap: dict[str, dict[str, float]], set_srcID_col: bool = True) -> pd.DataFrame:
    """Convert given source data dmap to pandas DataFrame."""
    df = pd.DataFrame.from_dict(dmap, orient='index')
    if set_srcID_col:
        df.reset_index(names=['srcID'], inplace=True)
    return df


# -----------   source PSF fit    -----------

def solver(
    func: Callable[[NDArray, tuple[float, ...]], Any],
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
    fit_weights: Optional[NDArray | Callable[[NDArray], NDArray]] = None,
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
        return cropped.ravel()

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
        #    - the box is built from the digital upsampling to account for both
        #      vignetting and inclined penetration from off-axis photons
        #    - the fluence cannot be smaller than the one observed at the peak,
        #      and we insert a lower value just for precaution (if simulating
        #      for example an infinite detector spatial resolution)
        halfbox_x, halfbox_y = (
            2 * camera.specs.mask_deltax / camera.upscale_f.x,
            2 * camera.specs.mask_deltay / camera.upscale_f.y,
        )
        bounds = [
            (
                max(sx_start - halfbox_x, camera.bins_sky.x[0]),
                max(sy_start - halfbox_y, camera.bins_sky.y[0]),
                0.95 * sky_peak,
            ),
            (
                min(sx_start + halfbox_x, camera.bins_sky.x[-1]),
                min(sy_start + halfbox_y, camera.bins_sky.y[-1]),
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


# ------   analysis   ------

def fit_src_in_skygrid(
    camera: CodedMaskCamera,
    sdl: DataLoader,
    ctg: CatalogueLoader,
    config_optim: Callable[[NDArray], Callable[[NDArray, tuple[int, int]], OptResult]],
    psf_crp_xyoff: tuple[int, int] = (0, 0),
    log_progress: bool = True,
    **phs_corr,
) -> dict[str, dict[str, float]]:
    """
    Doc...
    """
    out_dmap: dict[str, dict[str, float]] = {}
    src_list = ctg.DLdata[(ctg.DLdata['ID'] != '-1')]  # exclude CXB

    # filter CXB (same for all sources) and create detector image to only filter for
    # target source photons once (needed in loop for shadowgram and true counts);
    # total detector image is CXB + SRC for each SRC in list
    exclude_srcs: list[CoordEquatorial] = [
        CoordEquatorial(ra, dec) for ra, dec in zip(src_list['RA'], src_list['DEC'])
    ]
    cxb = ds.filter_data(sdl.DLdata, E_min=None, E_max=None, coords=exclude_srcs)
    cxb_detector, _ = count(camera, cxb, **phs_corr)

    loop = zip(src_list['NAME'], src_list['RA'], src_list['DEC'])
    loop = tqdm(loop, total=len(src_list), desc="Sky-field Analysis") if log_progress else loop

    for srcID, ra, dec in loop:
        if log_progress: loop.set_description_str(f'ID: {srcID.upper()}')
        
        # filter for source + CAI operations
        src_coords = CoordEquatorial(ra, dec)
        src_phs = select_source_photons(src_coords, sdl.DLdata, verbose=False)
        src_detector, _ = count(camera, src_phs, **phs_corr)

        detector = src_detector + cxb_detector
        skymap = decode(camera, detector)
        varmap = variance(camera, detector)

        # fit source params from PSF
        src_pos = equatorial2pos(sdl, camera, *src_coords)
        src_peak = find_boxmax(skymap, src_pos, boxsize=2 * camera.upscale_f.x + 1)

        rows, cols = crop_source_psf(camera, src_peak, *psf_crp_xyoff)
        psfvar = varmap[rows, cols]
        fit_weights = np.pow(np.clip(psfvar, a_min=1.0, a_max=None).ravel(), -0.5)

        optimiser = config_optim(fit_weights=fit_weights)
        fit_src_pars = config_fitter(optimiser)
        pars, pcov = fit_src_pars(src_peak, skymap)
        sx_opt, sy_opt, cts_opt = pars

        # compute out vals
        tx_true, ty_true = equatorial2angle(sdl, camera, *src_coords)
        tx_opt, ty_opt = map(lambda x: shift2angle(camera, x), (sx_opt, sy_opt))
        
        cts_peak, var_peak = skymap[*src_peak], varmap[*src_peak]
        peak_sigma = pow(var_peak, 0.5)
        tot_sigma = pow(var_peak + pcov[-1, -1], 0.5)

        cts_true = src_detector.sum()

        out_dmap[srcID] = {
            'shiftX_opt [mm]': sx_opt,
            'shiftY_opt [mm]': sy_opt,

            'angleX [deg]': tx_true,
            'angleY [deg]': ty_true,
            'DangleX [arcmin]': (tx_opt - tx_true) * 60,
            'DangleY [arcmin]': (ty_opt - ty_true) * 60,

            'true cts': cts_true,
            'optim cts': cts_opt,
            'Dcts [%]': (cts_opt - cts_true) * 100 / cts_true,
            'D_Dcts [%]': tot_sigma * 100 / cts_true,
            'Dcts [sigma]': (cts_opt - cts_true) / tot_sigma,

            'cc peak cts': cts_peak,
            'cc Dcts [%]': (cts_peak - cts_true) * 100 / cts_true,
            'cc Dcts [sigma]': (cts_peak - cts_true) / peak_sigma,

            'SNR': cts_peak / peak_sigma,
        }
        for a in out_dmap[srcID]:
            out_dmap[srcID][a] = float(out_dmap[srcID][a])

    return out_dmap


def analyse_camgeom(
    in_out_fpaths: tuple[str, str],
    camera: CodedMaskCamera,
    energy_band: tuple[Optional[float], Optional[float]],
    vignetting: bool | Callable[[CodedMaskCamera, NDArray, float, float], NDArray],
    psfy: bool | Callable[[CodedMaskCamera, NDArray], NDArray],
    camID: str = 'cam1a',
    psf_crp_xyoff: tuple[int, int] = (0, 0),
    log_progress: bool = True,
    src_optim_kws: dict[str, Any] = {},
    phs_corr: dict[str, Any] = {},
) -> None:
    """
    Doc...
    """
    simpath, outpath = map(Path, in_out_fpaths)
    E_min, E_max = energy_band
    dataset = 'reconstructed' if psfy else 'detected'

    opt_factory = dict(
        camera=camera,
        psf_crp_xyoff=psf_crp_xyoff,
        vignetting=vignetting, 
        psfy=psfy,
        **src_optim_kws,
    )
    config_optim = partial(config_optimiser, **opt_factory)

    ctg = ds.get_catalogue(check_and_pick(simpath, f'{camID}/*sources*.fits'))
    sdl = ds.get_data(check_and_pick(simpath, f'{camID}/*{dataset}*.fits'), E_min=E_min, E_max=E_max)

    src_dmap = fit_src_in_skygrid(
        camera=camera,
        sdl=sdl,
        ctg=ctg,
        config_optim=config_optim,
        psf_crp_xyoff=psf_crp_xyoff,
        log_progress=log_progress,
        **phs_corr,
    )
    src_df = srcdmap_to_df(src_dmap)

    upx, upy = camera.upscale_f
    out_fname = f'{simpath.name}_{dataset}_{E_min or 2.0}-{E_max or 50.0}keV_upx{upx}upy{upy}.csv'
    outpath.mkdir(parents=True, exist_ok=True)
    save_to_csv(src_df, outpath / out_fname)

    return


# end