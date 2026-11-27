"""
Mask warp effects support module. This script join the photon lists of two symmetric simulations with non-zero
mask pitch around the Y-axis to simulate at first order warping stress of the coded-mask plate.
"""

from functools import partial
import multiprocessing
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from astropy.io import fits
from astropy.table import Table, vstack
from tqdm import tqdm

import darksun as ds
from darksun.data import DataLoader

from camgeom.simstudy import check_and_pick

def _grep_phIDs(sdl: DataLoader, cond: Callable[[NDArray], NDArray]) -> NDArray:
    """Extracts photon IDs from given list based on mask output X coord condition."""
    mask = cond(sdl.DLdata['X'])
    return sdl.DLdata['ID'][mask]

def _intersect_IDs(all_ids: NDArray, collected_ids: NDArray, uniqueIDs: bool = True) -> NDArray:
    """Extracts valid photon ID indexes within collected photons list."""
    _, _, idx_b = np.intersect1d(all_ids, collected_ids, assume_unique=uniqueIDs, return_indices=True)
    return idx_b

def extract_phs(sdl_mask: DataLoader, sdl: DataLoader, cond: Callable[[NDArray], NDArray]) -> fits.FITS_rec:
    """Extracts valid photons from ID intersection between collected photons and selected output from mask."""
    phIDs = _grep_phIDs(sdl_mask, cond)
    IDidxs = _intersect_IDs(phIDs, sdl.DLdata['ID'])
    return sdl.DLdata[IDidxs]


def merge_photons(
    phs_left: fits.FITS_rec,
    phs_right: fits.FITS_rec,
    save_to: str | Path | None = None,
    overwrite: bool = False,
    tab_name: str | None = None,
    header: fits.Header | None = None,
) -> fits.FITS_rec:
    """Merges given data."""
    tl, tr = map(lambda x: Table(x), (phs_left, phs_right))
    merged_table = vstack([tl, tr])
    merged_hdu = fits.BinTableHDU(data=merged_table, name=tab_name)
    print(f'Merged photon lists with total elements: {len(merged_table)}')

    if save_to is not None:
        print("# Saving merged photon list...")
        primary_hdu = fits.PrimaryHDU()
        if header is not None:
            try:
                primary_hdu.header.extend(header, strip=True, update=True)
                print('Updated BinTable header!')
            except Exception as e:
                print(f'Encountered error {e}, skipping header update...')
        hdu_list = fits.HDUList([primary_hdu, merged_hdu])
        hdu_list.writeto(save_to, output_verify="fix+ignore", overwrite=overwrite)
        hdu_list.close()
        print("# Saving completed!")
    
    return merged_hdu.data


def run(
    filepaths: tuple[tuple[str, str], str],
    dataset: str,
    camID: str,
    overwrite: bool = False,
) -> None:
    """
    """
    # filepaths to FITS, refer to which half-mask to retain
    path_left, path_right, save_to = map(Path, (*filepaths[0], filepaths[-1]))
    # sdls with data, for both sims (photons from mask output and collected by SDDs)
    sdl_mask_left, sdl_mask_right = map(lambda path: ds.get_data(check_and_pick(path, f'{camID}/*mask*.fits')), (path_left, path_right))
    sdl_left, sdl_right = map(lambda path: ds.get_data(check_and_pick(path, f'{camID}/*{dataset}*.fits')), (path_left, path_right))
    # define cond and select photons
    print('Extracting photons...')
    valid_phs_left = extract_phs(sdl_mask_left, sdl_left, lambda x: x < 0.0)
    valid_phs_right = extract_phs(sdl_mask_right, sdl_right, lambda x: x >= 0.0)
    print('Photons successfully extracted!')
    # merge photon lists and save
    save_to.parent.mkdir(parents=True, exist_ok=True)
    kws = dict(overwrite=overwrite, tab_name=dataset.upper(), header=sdl_left.header)
    _ = merge_photons(valid_phs_left, valid_phs_right, save_to=save_to, **kws)
    return




def main(
    sims: list[tuple[tuple[str, str], str]],
    dataset: str,
    camID: str = 'cam1a',
    n_workers: int = 4,
    overwrite_outfits: bool = False,
) -> None:
    """Executes script."""
    # NOTE: removed merged FITS_Rec output from `run()` as with list comprehension and
    #       multiprocessing may lead to processing lag and/or out-of-memory crashes
    worker_fn = partial(
        run,
        dataset=dataset,
        camID=camID,
        overwrite=overwrite_outfits,
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
    DATASET: str = 'reconstructed'
    ID_CAM: str = 'cam1a'
    DIRPATH: str = '/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data/Simulations/CameraGeometry/mask_warps/mask_y_warps'

    pos_y_warp_vals: list[float] = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]  # [arcmin]
    neg_y_warp_vals: list[float] = [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5]  # [arcmin]

    CASE_STUDY: list[tuple[tuple[str, str], str]] = [
        # - POSITIVE Y MASK WARPING
        # NOTE: positive on the left (X < 0), negative on the right (X >= 0)
        *tuple(
            (
                (f'{DIRPATH}/SIMS_FOR_MASK_WARPING/pos_{val}arcmin', f'{DIRPATH}/SIMS_FOR_MASK_WARPING/neg_{val}arcmin'),
                f'{DIRPATH}/y_warps_{val}arcmin/{ID_CAM}/{val}arcmin_2-50keV_1ks_{DATASET}.fits'
            )
            for val in pos_y_warp_vals
        ),
        # - NEGATIVE Y MASK WARPING
        # NOTE: negative on the left (X < 0), positive on the right (X >= 0)
        *tuple(
            (
                (f'{DIRPATH}/SIMS_FOR_MASK_WARPING/neg_{val}arcmin', f'{DIRPATH}/SIMS_FOR_MASK_WARPING/pos_{val}arcmin'),
                f'{DIRPATH}/y_warps_{val}arcmin/{ID_CAM}/{val}arcmin_2-50keV_1ks_{DATASET}.fits'
            )
            for val in neg_y_warp_vals
        )
    ]

    main(CASE_STUDY, dataset=DATASET, camID=ID_CAM, n_workers=4)


# end