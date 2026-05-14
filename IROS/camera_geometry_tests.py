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

from bloodmoon.mask import CodedMaskCamera, codedmask, count
from bloodmoon.mask import decode, variance, snratio
import darksun as ds
from darksun.data import Log
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
    true_cts = mgm.extract_catalogue_fluences(log, catalogue, sdl, camera)
    dmap = {
        'Source': ids,

        'thetaX [deg]': np.array(log.log['angle_x']),
        'thetaY [deg]': np.array(log.log['angle_y']),
        'cts [ph]': log.log['fluence'],

        'DthetaX [arcmin]': theta_res_x,
        'DthetaY [arcmin]': theta_res_y,
        'Dcts [%]': (cts - true_cts) * 100 / true_cts,
        'Dcts [sigma]': (cts - true_cts) / np.sqrt(true_cts),

        'SNR [sigma]': np.array(log.log['snr']),
    }
    return pd.DataFrame(dmap)

def save_sky(
    sky: NDArray,
    snr: NDArray,
    sdl: DataLoader,
    save_to: str | Path,
    detector: NDArray | None = None,
    wcs: WCS = None,
) -> None:
    """
    Saves the given sky array and its significance (SNR) as FITS Image files,
    including optional World Coordinate System (WCS) information if provided.
    """
    print("# Saving sky...")
    # HDU list and Primary Header
    hdu_list = fits.HDUList([])
    primary_hdu = fits.PrimaryHDU()
    hdu_list.append(primary_hdu)
    arrs = [(np.int32(sky), 'sky'), (np.float32(snr), 'snr')]
    if detector is not None:
        arrs.append((np.int32(detector), 'detector'))

    # Images for data
    for img, name in arrs:
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

def save_df_totxt(df: pd.DataFrame, save_to: str) -> None:
    """Saves input dataframe to `.txt` file."""
    with open(save_to, "w", encoding="utf-8") as f:
        f.write(df.to_string(col_space=[10] * len(df.columns), index=False, justify='center'))
    return




def main() -> None:
    """
    Executes IROS and sources output data comparison with true values.
    """
    MASK_FITS: str = "mask_NTHT_20260129_CORRECTED.fits"
    UPS_X, UPS_Y = 2, 1

    SKYFIELD: str = "CameraGeometry"
    # DATA_FITS: str = "baseline_sources_2-50keV_1ks"
    DATA_FITS: str = "mask_misaligned_Z1deg_2-50keV_1ks"

    RUN_ID: str = 'baseline_misalignIROS_1ks_2-50keV_detected'

    ID_CAMERA_A: str = "cam1a"
    DATASET: str = "detected"

    VIGNETTING = True
    PSFY = mgm.config_psfy_flag(DATASET)

    MASK_PATH, SIMUL_DATA_PATH, SAVE_PATH = config_dirpaths(
        mask=MASK_FITS,
        skyfield=SKYFIELD,
        simul=DATA_FITS,
        runID=RUN_ID,
    )
    get_datapaths: Callable[[str], str] = lambda dataset: SIMUL_DATA_PATH + f'{ID_CAMERA_A}/{DATA_FITS}_{ID_CAMERA_A}_{dataset}.fits'

    wfm: CodedMaskCamera = codedmask(MASK_PATH, UPS_X, UPS_Y)
    sdlA = ds.get_data(get_datapaths(DATASET), E_min=None, E_max=None)
    catA = ds.get_catalogue(get_datapaths('sources'))

    detector, _ = count(wfm, sdlA.DLdata)
    skymap = decode(wfm, detector)
    varmap = variance(wfm, detector)
    snrmap = snratio(skymap, varmap)
    save_sky(skymap, snrmap, sdlA, f'{SAVE_PATH}/SIMUL_sky.fits', detector=detector)

    max_iters = 1
    KWS: dict[str, Any] = {
        'vignetting': VIGNETTING,
        'psfy': PSFY,
    }
    log = perform_IROS(wfm, detector, max_iters, camID=ID_CAMERA_A, **KWS)
    log = get_sources_database(wfm, sdlA, catA, log, vignetting=VIGNETTING)
    outdf = gather_cam_data(log, catA, sdlA, wfm)
    save_df_totxt(outdf, f'{SAVE_PATH}/OUT_{RUN_ID}.txt')

    return


if __name__ == '__main__':
    main()


# end