"""
Coded-mask camera geometric configuration analyser.
This module is intended for benchmarking the performance of a LEM-X camera accounting for assembling systematics,
such as mask and/or SDDs (or even whole detector plane) rotations and/or shifts with respect to the nominal design.
"""

from functools import partial
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from tqdm import tqdm

from bloodmoon.types import CoordEquatorial
from bloodmoon.mask import CodedMaskCamera, codedmask, count
from bloodmoon.mask import decode
from bloodmoon.optim import model_sky
import darksun as ds
from darksun.types import LogEntry
from darksun.data import Log, DataLoader, CatalogueLoader

from IROSrec.iros.optim import iros_singleCAM
from IROSrec.iros.procedure import run_IROS, get_sources_database
import imgmaker as mgm


DIRPATH: str = '/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data'

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

def perform_IROS(
    camera: CodedMaskCamera,
    detector: NDArray,
    max_iterations: int,
    camID: str | None = None,
    **iros_kwargs: Any,
) -> Log:
    """Runs the IROS procedure and stores optimised sources params."""
    loop = iros_singleCAM(camera, detector, max_iterations, **iros_kwargs)
    log, _ = run_IROS(camera, loop, camID)
    return log

def add_skypeaks_to_log(camera: CodedMaskCamera, log: Log, skymap: NDArray, boxsize: int = 5, **src_kws: Any) -> Log:
    """Adds the sky peak counts at first sky CC reconstruction and after each IROS iteration."""
    it_skymap = skymap.copy()
    log.insert(
        (LogEntry('cc_peak_cts', '', 'ph'), LogEntry('iter_peak_cts', '', 'ph')),
    )
    for x, y, sx, sy, f in zip(
        log.log['x'], log.log['y'], log.log['shift_x'], log.log['shift_y'], log.log['fluence'],
    ):
        box = (
            slice(y - 2 * boxsize, y + 2 * boxsize + 1),
            slice(x - boxsize, x + boxsize + 1),
        )
        cc_cts, it_cts = map(np.max, (skymap[*box], it_skymap[*box]))
        log.update([('cc_peak_cts', cc_cts), ('iter_peak_cts', it_cts)])
        it_skymap -= model_sky(camera, sx, sy, f, **src_kws)

    return log

def gather_cam_data(
    log: Log,
    catalogue: CatalogueLoader,
    sdl: DataLoader,
    camera: CodedMaskCamera,
) -> pd.DataFrame:
    """
    Gathers single camera data from IROS reconstruction database.
    """
    ids = np.array([src.upper() for src in log.log['ID']])
    theta_res_x, theta_res_y = mgm.get_angularcoords_residues(
        log, catalogue, sdl, camera,
    )
    cts = np.array(log.log['fluence'])
    cc_peak_cts = np.array(log.log['cc_peak_cts'])
    it_peak_cts = np.array(log.log['iter_peak_cts'])
    true_cts = mgm.extract_catalogue_fluences(log, catalogue, sdl, camera)
    dmap = {
        'Source': ids,

        'thetaX [deg]': np.array(log.log['angle_x']),
        'thetaY [deg]': np.array(log.log['angle_y']),
        'DthetaX [arcmin]': theta_res_x,
        'DthetaY [arcmin]': theta_res_y,

        'true cts [ph]': true_cts,
        'optim cts [ph]': cts.astype(np.int32),
        'Dcts [%]': (cts - true_cts) * 100 / true_cts,
        'Dcts [sigma]': (cts - true_cts) / np.sqrt(true_cts),

        'cc peak cts [ph]': cc_peak_cts.astype(np.int32),
        'Dcts cc [%]': (cc_peak_cts - true_cts) * 100 / true_cts,
        'iter peak cts [ph]': it_peak_cts.astype(np.int32),
        'Dcts iter [%]': (it_peak_cts - true_cts) * 100 / true_cts,

        'SNR [sigma]': np.array(log.log['snr']),
    }
    return pd.DataFrame(dmap)

def df_to_csv(df: pd.DataFrame, save_to: str | Path) -> None:
    """Saves input dataframe to `.csv` file."""
    kws = {
        'index': False,
        'float_format': '%.6f',
    }
    df.to_csv(save_to, **kws)
    return

def analyse_sim(
    filepaths: tuple[str, str],
    camera_path: str,
    upscaling: tuple[int, int],
    dataset: str,
    energy_range: tuple[float | None, float | None],
    camID: str = 'cam1a',
    vignetting: bool = True,
    psfy: bool = True,
    save_data: bool = True,
    **iros_kws: Any,
) -> None:
    """Runs the analysis for specified camera layout."""
    simpath, outpath = map(Path, filepaths)
    E_min, E_max = energy_range
    wfm: CodedMaskCamera = codedmask(camera_path, *upscaling)
    out_dfs: list[pd.DataFrame] = []

    sdl = ds.get_data(
        check_and_pick(simpath, f'{camID}/*{dataset}*.fits'), E_min=E_min, E_max=E_max, coords=None,
    )
    cat = ds.get_catalogue(check_and_pick(simpath, f'{camID}/*sources*.fits'))
    # select up to second-to-last row to exclude CXB
    src_list = cat.DLdata[:-1]

    for srcID in src_list['NAME']:
        mask = (src_list['NAME'] != srcID)
        exclude_srcs: list[CoordEquatorial] = [
            CoordEquatorial(ra, dec) for ra, dec in zip(src_list[mask]['RA'], src_list[mask]['DEC'])
        ]
        phs = ds.filter_data(sdl.DLdata, E_min=None, E_max=None, coords=exclude_srcs)

        detector, _ = count(wfm, phs)
        skymap = decode(wfm, detector)

        log = perform_IROS(wfm, detector, max_iterations=1, camID=camID, vignetting=vignetting, psfy=psfy, **iros_kws)
        log = get_sources_database(wfm, sdl, cat, log, vignetting=vignetting)
        log = add_skypeaks_to_log(wfm, log, skymap, vignetting=vignetting, psfy=psfy)
        out_dfs.append(gather_cam_data(log, cat, sdl, wfm))
    
    # merge single src data and save csv
    out_src_data = pd.concat(out_dfs, ignore_index=True)
    if save_data:
        print(f'Saving {simpath.name}')
        outpath.mkdir(parents=True, exist_ok=True)
        upx, upy = wfm.upscale_f
        df_to_csv(out_src_data, outpath / f'{simpath.name}_{dataset}_{E_min or 2.0}-{E_max or 50.0}keV_upx{upx}upy{upy}.csv')

    return




def main(sims: list[tuple[str, str]], n_workers: int = 4) -> None:
    """
    Runs LEM-X single coded-mask camera performance analysis.

    Args:
        sims (list[tuple[str, str]]):
            List of tuples with data directory path and respective directory path to save output CSV database.
    """
    MASK_FITS: str = f"{DIRPATH}/Simulations/mask_NTHT_20260129_CORRECTED.fits"
    UPS_X, UPS_Y = 5, 1

    DATASET: str = 'detected'
    E_min: float | None = None
    E_max: float | None = None

    VIGNETTING: bool = True
    PSFY: bool = mgm.config_psfy_flag(DATASET)
    IROS_KWS: dict[str, Any] = {}

    worker_fn = partial(
        analyse_sim,
        camera_path=MASK_FITS,
        upscaling=(UPS_X, UPS_Y),
        dataset=DATASET,
        energy_range=(E_min, E_max),
        vignetting=VIGNETTING,
        psfy=PSFY,
        **IROS_KWS,
    )
    n_workers_ = max(1, min(n_workers, multiprocessing.cpu_count() - 1))
    print(f'Starting analysis on {n_workers_} cores...')
    with multiprocessing.Pool(processes=n_workers_) as pool:
        list(
            tqdm(
                pool.imap_unordered(worker_fn, sims),
                total=len(sims),
                desc='Camera Analysis',
            )
        )
    print('Analysis complete!')

    return


if __name__ == '__main__':

    simspath: str = f'{DIRPATH}/Simulations/CameraGeometry'
    outspath: str = f'{DIRPATH}/Outputs/OutCameraGeometry'
    CASE_STUDY: list[str] = [
        # Baseline
        # (f'{simspath}/baseline/baseline', f'{outspath}/baseline'),

        # Mask rotations
        # # - X axis
        # (f'{simspath}/mask_rots/mask_Xrot/Xrot_0.5arcmin', f'{outspath}/mask_rots/mask_Xrot'),
        # (f'{simspath}/mask_rots/mask_Xrot/Xrot_1arcmin', f'{outspath}/mask_rots/mask_Xrot'),
        # (f'{simspath}/mask_rots/mask_Xrot/Xrot_2arcmin', f'{outspath}/mask_rots/mask_Xrot'),
        # # - Y axis
        # (f'{simspath}/mask_rots/mask_Yrot/Yrot_4arcmin', f'{outspath}/mask_rots/mask_Yrot'),
        # (f'{simspath}/mask_rots/mask_Yrot/Yrot_6arcmin', f'{outspath}/mask_rots/mask_Yrot'),
        # - Z axis
        # (f'{simspath}/mask_rots/mask_Zrot/Zrot_0.5arcmin', f'{outspath}/mask_rots/mask_Zrot'),
        # (f'{simspath}/mask_rots/mask_Zrot/Zrot_1arcmin', f'{outspath}/mask_rots/mask_Zrot'),
        # (f'{simspath}/mask_rots/mask_Zrot/Zrot_2arcmin', f'{outspath}/mask_rots/mask_Zrot'),
        # (f'{simspath}/mask_rots/mask_Zrot/Zrot_4arcmin', f'{outspath}/mask_rots/mask_Zrot'),
        # (f'{simspath}/mask_rots/mask_Zrot/Zrot_6arcmin', f'{outspath}/mask_rots/mask_Zrot'),

        # # SDD_00 rotations
        # # - X axis
        # (f'{simspath}/sdd00_rots/sdd00_Xrot/Xrot_0.5arcmin', f'{outspath}/sdd00_rots/sdd00_Xrot'),
        # (f'{simspath}/sdd00_rots/sdd00_Xrot/Xrot_1arcmin', f'{outspath}/sdd00_rots/sdd00_Xrot'),
        # (f'{simspath}/sdd00_rots/sdd00_Xrot/Xrot_2arcmin', f'{outspath}/sdd00_rots/sdd00_Xrot'),
        # # - Y axis
        # (f'{simspath}/sdd00_rots/sdd00_Yrot/Yrot_0.5arcmin', f'{outspath}/sdd00_rots/sdd00_Yrot'),
        # (f'{simspath}/sdd00_rots/sdd00_Yrot/Yrot_1arcmin', f'{outspath}/sdd00_rots/sdd00_Yrot'),
        # (f'{simspath}/sdd00_rots/sdd00_Yrot/Yrot_2arcmin', f'{outspath}/sdd00_rots/sdd00_Yrot'),
        # # - Z axis
        # (f'{simspath}/sdd00_rots/sdd00_Zrot/Zrot_0.5arcmin', f'{outspath}/sdd00_rots/sdd00_Zrot'),
        # (f'{simspath}/sdd00_rots/sdd00_Zrot/Zrot_1arcmin', f'{outspath}/sdd00_rots/sdd00_Zrot'),
        # (f'{simspath}/sdd00_rots/sdd00_Zrot/Zrot_2arcmin', f'{outspath}/sdd00_rots/sdd00_Zrot'),

        # # SDD_00 shifts
        # # - X axis
        # (f'{simspath}/sdd00_shifts/sdd00_Xshift/Xshift_10um', f'{outspath}/sdd00_shifts/sdd00_Xshift'),
        # (f'{simspath}/sdd00_shifts/sdd00_Xshift/Xshift_30um', f'{outspath}/sdd00_shifts/sdd00_Xshift'),
        # (f'{simspath}/sdd00_shifts/sdd00_Xshift/Xshift_50um', f'{outspath}/sdd00_shifts/sdd00_Xshift'),
        # # - Y axis
        # (f'{simspath}/sdd00_shifts/sdd00_Yshift/Yshift_10um', f'{outspath}/sdd00_shifts/sdd00_Yshift'),
        # (f'{simspath}/sdd00_shifts/sdd00_Yshift/Yshift_30um', f'{outspath}/sdd00_shifts/sdd00_Yshift'),
        # (f'{simspath}/sdd00_shifts/sdd00_Yshift/Yshift_50um', f'{outspath}/sdd00_shifts/sdd00_Yshift'),
        # # - Z axis
        # (f'{simspath}/sdd00_shifts/sdd00_Zshift/Zshift_10um', f'{outspath}/sdd00_shifts/sdd00_Zshift'),
        # (f'{simspath}/sdd00_shifts/sdd00_Zshift/Zshift_30um', f'{outspath}/sdd00_shifts/sdd00_Zshift'),
        # (f'{simspath}/sdd00_shifts/sdd00_Zshift/Zshift_50um', f'{outspath}/sdd00_shifts/sdd00_Zshift'),

        # SDD plane positive warping
        # - Y axis
        (f'{simspath}/sdd_plane_warps/sdd_plane_Ywarp/pYwarp_2arcmin', f'{outspath}/sdd_plane_warps/sdd_plane_Ywarp'),
        (f'{simspath}/sdd_plane_warps/sdd_plane_Ywarp/pYwarp_4arcmin', f'{outspath}/sdd_plane_warps/sdd_plane_Ywarp'),
        (f'{simspath}/sdd_plane_warps/sdd_plane_Ywarp/pYwarp_6arcmin', f'{outspath}/sdd_plane_warps/sdd_plane_Ywarp'),
    ]

    main(CASE_STUDY)


# end