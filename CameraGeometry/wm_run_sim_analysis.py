"""
Script for the analysis of WISEMAN simulation concerning LEM-X camera geometry.
"""

import os

# avoid BLAS / OpenMP thread oversubscription from scipy
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

from pathlib import Path
from functools import lru_cache
import multiprocessing
from typing import Any, Callable, Optional
import warnings

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve

from bloodmoon.mask import CodedMaskCamera, codedmask
from camgeom.simstudy import analyse_camgeom

# ------   helpers ------

_WORKER_STATE_DMAP: dict[str, Any] = {}

def _init_worker_objs(
    camera: CodedMaskCamera,
    energy_band: tuple[Optional[float], Optional[float]],
    vignetting: bool | Callable[[CodedMaskCamera, NDArray, float, float], NDArray] = True,
    psfy: bool | Callable[[CodedMaskCamera, NDArray], NDArray] = True,
    log_progress: bool = True,
    **kwargs,
) -> None:
    _WORKER_STATE_DMAP['camera'] = camera
    _WORKER_STATE_DMAP['energy_band'] = energy_band
    _WORKER_STATE_DMAP['vignetting'] = vignetting
    _WORKER_STATE_DMAP['psfy'] = psfy
    _WORKER_STATE_DMAP['log_progress'] = log_progress
    _WORKER_STATE_DMAP['kwargs'] = kwargs
    return

def select_sims(
    parent: str | Path,
    pattern: str = '*',
    exclude: tuple[str | Path, ...] = (),
) -> tuple[Path, ...]:
    """
    Globs WM sims with given pattern in parent dir.
    """
    def check_fpath(p: str | Path, matches: set[Path]) -> Path:
        p = Path(p)
        p = parent / p if not p.is_absolute() else p
        if p not in matches:
            warnings.warn(f"Chosen file (to select/exclude) '{p.name}' not in directory.")
        return p

    parent = Path(parent)
    matches = set(parent.glob(pattern))
    if not exclude:
        return tuple(matches)

    exclude = tuple(check_fpath(p, matches) for p in exclude)
    return tuple(matches.difference(exclude))


# ------   analysis   ------

def single_run_analyser(in_out_fpaths: tuple[str, str]) -> None:
    """Runs single analysis for LEM-X camera geometry."""
    analyse_camgeom(
        in_out_fpaths=in_out_fpaths,
        camera=_WORKER_STATE_DMAP['camera'],
        energy_band=_WORKER_STATE_DMAP['energy_band'],
        vignetting=_WORKER_STATE_DMAP['vignetting'],
        psfy=_WORKER_STATE_DMAP['psfy'],
        log_progress=_WORKER_STATE_DMAP['log_progress'],
        kwargs=_WORKER_STATE_DMAP['kwargs'],
    )
    return

def analyse_geomsims(
    camera: CodedMaskCamera,
    sims: list[tuple[str, str]],
    energy_band: tuple[Optional[float], Optional[float]],
    vignetting: bool | Callable[[CodedMaskCamera, NDArray, float, float], NDArray] = True,
    psfy: bool | Callable[[CodedMaskCamera, NDArray], NDArray] = True,
    log_progress: bool = True,
    n_workers: int = 4,
    **kwargs,
) -> None:
    """
    Runs LEM-X coded-mask camera performance analysis given the
    list of data and target output CSV database directory paths.
    """
    n_workers = (
        max(1, min(n_workers, multiprocessing.cpu_count() - 1))
        if len(sims) > 1 else 1
    )
    print(f'Starting analysis on {n_workers} cores...')

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker_objs,
        initargs=(camera, energy_band, vignetting, psfy, log_progress, kwargs),
    ) as pool:
        list(
            tqdm(
                pool.imap_unordered(single_run_analyser, sims),
                total=len(sims),
                desc='Camera Analysis',
            )
        )
    print('Analysis complete!')
    return


# ------   SDD spatial resolution kernel   ------

@lru_cache(maxsize=1)
def _psfy_kernel_cached(camera: CodedMaskCamera) -> NDArray:
    """Cached SDD resolution kernel."""
    gauss_kws = {"mode": 'constant', "cval": 0.0}
    alpha, beta, sigma = 0.15, 0.51, 0.1

    px_ydim = camera.specs.mask_deltay / camera.upscale_f.y
    slit_dim = camera.specs.slit_deltay
    bins = np.linspace(-slit_dim, slit_dim, int(2 * slit_dim / px_ydim) + 1)

    modsec = 1.0 / np.cosh(np.abs(bins / alpha) ** beta)
    psfy = gaussian_filter(modsec, sigma, **gauss_kws)

    return psfy / np.sum(psfy)

def psfy_for_crab(camera: CodedMaskCamera, shadowgram: NDArray) -> NDArray:
    """Models SDD spatial resolution correction, calibrated for point-like Crab in [2, 50] keV."""
    kernel = _psfy_kernel_cached(camera)
    return convolve(shadowgram, kernel, mode="same")
    




if __name__ == '__main__':

    basepath: str = '/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data'
    mask_fpath: str = f'{basepath}/Simulations/mask_NTHT_20260129_CORRECTED.fits'

    # basepath: str = '/mnt/d/PhD_AASS/Coding/Images_fits'
    # mask_fpath: str = f'{basepath}/mask_NTHT_20260129_CORRECTED.fits'

    sims_parent: str = f'{basepath}/Simulations/CameraGeometry'
    sims_out: str = f'{basepath}/Outputs/OutCameraGeometry'

    # - pars ----------------------------------------------------------------------------------------
    energy_band = (2.0, 50.0)   # [keV]
    wfm: CodedMaskCamera = codedmask(mask_fpath, 2, 1, hide_bulk_els_y=1.5)

    # - sim/output filepaths ------------------------------------------------------------------------
    sims: list[tuple[str, str]] = []

    # ---------   BASELINE   ---------
    field = 'baseline'
    simpaths = select_sims(f"{sims_parent}/{field}")
    sims += [
        (p, Path(f"{sims_out}/{field}/{p.name}")) for p in simpaths
    ]

    # ---------   MASK   ---------
    # - ROTATIONS
    field = 'mask_rots/mask_y_rots'
    simpaths = select_sims(f"{sims_parent}/{field}")
    sims += [
        (p, Path(f"{sims_out}/{field}/{p.name}")) for p in simpaths
    ]
    field = 'mask_rots/mask_z_rots'
    simpaths = select_sims(f"{sims_parent}/{field}")
    sims += [
        (p, Path(f"{sims_out}/{field}/{p.name}")) for p in simpaths
    ]

    print(len(sims))
    exit()

    # - PLATE WARPING
    field = '/'
    simpaths = select_sims(f"{sims_parent}/{field}")
    sims += [
        (p, Path(f"{sims_out}/{field}/{p.name}")) for p in simpaths
    ]

    # - PLATE DEFORMATIONS
    field = '/'
    simpaths = select_sims(f"{sims_parent}/{field}")
    sims += [
        (p, Path(f"{sims_out}/{field}/{p.name}")) for p in simpaths
    ]

    # ---------   DETECTOR PLANE   ---------
    # - DEFORMATIONS
    field = '/'
    simpaths = select_sims(f"{sims_parent}/{field}")
    sims += [
        (p, Path(f"{sims_out}/{field}/{p.name}")) for p in simpaths
    ]

    # - IN-PLANE SDD ROTATIONS
    field = '/'
    simpaths = select_sims(f"{sims_parent}/{field}")
    sims += [
        (p, Path(f"{sims_out}/{field}/{p.name}")) for p in simpaths
    ]


    # - analyse camera geometry sims ----------------------------------------------------------------
    analyse_geomsims(
        camera=wfm,
        sims=sims,
        energy_band=energy_band,
        n_workers=4,
        #
        psfy=psfy_for_crab,
        psf_crp_xyoff=(2 + wfm.upscale_f.x, 5 * wfm.upscale_f.y),
        src_optim_kws=None,
        phs_corr=None,
    )


# end