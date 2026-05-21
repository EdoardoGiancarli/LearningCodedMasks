"""
Adds the CXB to given WM photon list.
"""

from pathlib import Path
import numpy as np
from astropy.io import fits
from bloodmoon.io import validate_fits


def add_cxb(
    cxb_filepath: str | Path,
    data_filepath: str | Path,
) -> None:
    """
    Adds the CXB photons to given WM simulation. User confirm required.\n
    NOTE: BE CAREFUL with the input files!
    """
    for f in (cxb_filepath, data_filepath):
        if not validate_fits(f): raise ValueError(f"Invalid input file '{f}'.")
    
    confirm_update = input(
        f"Photon list in '{data_filepath}' will be updated with data from '{cxb_filepath}'.\n"
        f'Continue? (y/n): '
    )
    if confirm_update.lower() == 'n':
        print('Aborted photons list update...')
        return
    
    print('Appending CXB photons...')
    with (
        fits.open(cxb_filepath, mode="readonly") as hdu_cxb,
        fits.open(data_filepath, mode="update") as hdu_data,
    ):
        print(f'Photons list elements: {len(hdu_data[1].data)}')
        hdu_data[1].data = np.append(hdu_data[1].data, hdu_cxb[1].data)
        print(f'Updated photons list elements: {len(hdu_data[1].data)}')
    print('CXB added succesfully!')

    return




def main() -> None:

    SIMUL_PATH = "/mnt/d/PhD_AASS/Coding/Images_fits/CameraGeometry"
    BASELINE_SIMUL = "mask_Z2arcmin_2-50keV_1ks"
    CXB_SIMUL = "mask_Z2arcmin_cxb_2-50keV_1ks"

    DATASET = 'reconstructed'
    ID_CAMERA = 'cam1a'

    trg_sources_file = f'{SIMUL_PATH}/{BASELINE_SIMUL}/{ID_CAMERA}/{BASELINE_SIMUL}_{ID_CAMERA}_{DATASET}.fits'
    trg_cxb_file = f'{SIMUL_PATH}/{CXB_SIMUL}/{ID_CAMERA}/{CXB_SIMUL}_{ID_CAMERA}_{DATASET}.fits'

    add_cxb(trg_cxb_file, trg_sources_file)


if __name__ == '__main__':
    main()


# end