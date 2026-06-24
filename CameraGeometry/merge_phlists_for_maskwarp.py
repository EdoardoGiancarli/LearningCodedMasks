from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from astropy.io import fits
from astropy.table import Table, vstack

from bloodmoon.mask import CodedMaskCamera, codedmask
import darksun as ds
from darksun.data import DataLoader


def grep_phIDs(sdl: DataLoader, cond: Callable[[NDArray], NDArray]) -> NDArray:
    """Extracts photon IDs from given list based on mask output X coord condition."""
    mask = cond(sdl.DLdata['X'])
    return sdl.DLdata['ID'][mask]


def intersect_IDs(all_ids: NDArray, collected_ids: NDArray, uniqueIDs: bool = True) -> NDArray:
    """Extracts valid photon ID indexes within collected photons list."""
    _, _, idx_b = np.intersect1d(all_ids, collected_ids, assume_unique=uniqueIDs, return_indices=True)
    return idx_b


def merge_photons(
    phs_left: fits.FITS_rec,
    phs_right: fits.FITS_rec,
    save_to: str | Path | None = None,
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
        if header is not None:
            try:
                merged_hdu.header.extend(header, strip=True, update=True)
                print('Updated BinTable header!')
            except Exception as e:
                print(f'Encountered error {e}, skipping header update...')
        hdu_list = fits.HDUList([fits.PrimaryHDU(), merged_hdu])
        hdu_list.writeto(save_to, output_verify="fix+ignore")
        hdu_list.close()
        print("# Saving completed!")
    
    return merged_hdu.data







def main() -> None:
    """"""
    ...


if __name__ == '__main__':
    main()


# end