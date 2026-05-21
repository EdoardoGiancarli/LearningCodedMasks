"""
Module for camera geometry tests and IROS performance.
"""

from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from bloodmoon.types import CoordEquatorial
from bloodmoon.mask import CodedMaskCamera, codedmask, count
from bloodmoon.mask import decode, variance, snratio
from bloodmoon.optim import model_sky
import darksun as ds
from darksun.types import LogEntry
from darksun.data import Log, DataLoader, CatalogueLoader

from IROSrec.handle import config_dirpaths
from IROSrec.iros.optim import iros_singleCAM
from IROSrec.iros.procedure import run_IROS, get_sources_database
import imgmaker as mgm


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

def add_skypeaks_to_log(camera: CodedMaskCamera, log: Log, skymap: NDArray, **kws: Any) -> Log:
    """Adds the sky peak counts at first sky CC reconstruction and after each IROS iteration."""
    box_sz = 5
    true = skymap.copy()
    log.insert(
        (LogEntry('cc_peak_cts', '', 'ph'), LogEntry('iter_peak_cts', '', 'ph')),
    )

    for x, y, sx, sy, f in zip(
        log.log['x'], log.log['y'], log.log['shift_x'], log.log['shift_y'], log.log['fluence'],
    ):
        box = (
            slice(y - 2 * box_sz, y + 2 * box_sz + 1),
            slice(x - box_sz, x + box_sz + 1),
        )
        cc_cts = np.max(skymap[*box])
        it_cts = np.max(true[*box])
        log.update([('cc_peak_cts', cc_cts), ('iter_peak_cts', it_cts)])
        true -= model_sky(camera, sx, sy, f, **kws)

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

def save_sky(
    sky: NDArray,
    sdl: DataLoader,
    save_to: str | Path,
    snr: NDArray | None = None,
    detector: NDArray | None = None,
    wcs: WCS = None,
) -> None:
    """
    Saves the given sky array and its significance (SNR) as FITS Image files,
    including optional World Coordinate System (WCS) information if provided.
    """
    print("# Saving sky...")
    arrs = {
        'sky': np.int32(sky),
        'snr': np.float32(snr) if isinstance(snr, np.ndarray) else None,
        'detector': np.int32(detector) if isinstance(detector, np.ndarray) else None,
    }
    hdu_list = fits.HDUList([])
    primary_hdu = fits.PrimaryHDU()
    hdu_list.append(primary_hdu)
    for name, img in arrs.items():
        if img is not None:
            image_hdu = fits.ImageHDU(
                data=img,
                header=sdl.header,
                name=name.upper(),
            )
            if wcs: image_hdu.header.update(wcs.to_header())
            hdu_list.append(image_hdu)
    
    hdu_list.writeto(save_to, output_verify="fix+ignore")
    hdu_list.close()
    print("# Saving completed!")
    return

def df_to_csv(df: pd.DataFrame, save_to: str) -> None:
    """Saves input dataframe to `.csv` file."""
    kws = {
        'index': False,
        'float_format': '%.2f',
    }
    df.to_csv(save_to, **kws)
    return




def main() -> None:
    """
    Executes IROS and sources output data comparison with true values.
    """
    MASK_FITS: str = "mask_NTHT_20260129_CORRECTED.fits"
    UPS_X, UPS_Y = 2, 1

    SKYFIELD: str = "CameraGeometry"
    DATA_FITS: str = "baseline_2-50keV_1ks"
    # DATA_FITS: str = "mask_Z2arcmin_2-50keV_1ks"

    RUN_ID: str = 'baseline_wcxb'
    E_min: float | None = None
    E_max: float | None = None
    # exclude_coords: CoordEquatorial | list[CoordEquatorial] | None = [
    #     CoordEquatorial(318.950871147977, -51.1920260999562),   # s0
    #     CoordEquatorial(296.376333558386, 5.25424886863538),    # s1
    #     CoordEquatorial(236.423666441614, 5.25424886863538),    # s2
    #     CoordEquatorial(213.849128852023, -51.1920260999562),   # s3
    #     CoordEquatorial(286.170341181402, -42.2037736309785),   # s4
    #     CoordEquatorial(281.331753525598, -13.4867719837475),   # s5
    #     CoordEquatorial(251.468246474402, -13.4867719837475),   # s6
    #     CoordEquatorial(246.629658818598, -42.2037736309785),   # s7
    #     CoordEquatorial(266.4, -28.94),                         # s8
    # ]
    exclude_coords = None

    ID_CAMERA_A: str = "cam1a"
    DATASET: str = "detected"

    VIGNETTING: bool = True
    PSFY: bool = mgm.config_psfy_flag(DATASET)

    max_iters: int = (9 - len(exclude_coords)) if exclude_coords else 10

    # --- ROUTINE BODY ---
    MASK_PATH, SIMUL_DATA_PATH, SAVE_PATH = config_dirpaths(
        mask=MASK_FITS,
        skyfield=SKYFIELD,
        simul=DATA_FITS,
        runID=RUN_ID,
    )
    get_datapaths: Callable[[str], str] = lambda dataset: SIMUL_DATA_PATH + f'{ID_CAMERA_A}/{DATA_FITS}_{ID_CAMERA_A}_{dataset}.fits'

    wfm: CodedMaskCamera = codedmask(MASK_PATH, UPS_X, UPS_Y)
    sdlA = ds.get_data(get_datapaths(DATASET), E_min=E_min, E_max=E_max, coords=exclude_coords)
    catA = ds.get_catalogue(get_datapaths('sources'))

    detector, _ = count(wfm, sdlA.DLdata)
    skymap = decode(wfm, detector)
    varmap = variance(wfm, detector)
    snrmap = snratio(skymap, varmap)
    save_sky(skymap, sdlA, f'{SAVE_PATH}/SIMUL_sky.fits', snr=snrmap, detector=detector)

    KWS: dict[str, Any] = {
        'vignetting': VIGNETTING,
        'psfy': PSFY,
    }
    log = perform_IROS(wfm, detector, max_iters, camID=ID_CAMERA_A, **KWS)
    log = get_sources_database(wfm, sdlA, catA, log, vignetting=VIGNETTING)
    ds.save_database(log_camA=log, log_camB=log, sdlA=sdlA, sdlB=sdlA, save_to=f'{SAVE_PATH}/OUT_{RUN_ID}_srcDB.fits')
    log = add_skypeaks_to_log(wfm, log, skymap, **KWS)
    outdf = gather_cam_data(log, catA, sdlA, wfm)
    df_to_csv(outdf, f'{SAVE_PATH}/OUT_{RUN_ID}.csv')

    return


if __name__ == '__main__':
    main()


# end