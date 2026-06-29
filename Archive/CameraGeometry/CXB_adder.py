"""
Adds the CXB to given WM photon list.
"""

from pathlib import Path
from astropy.io import fits
from bloodmoon.io import validate_fits


def update_photons_list(
    append_to: str | Path,
    take_from: str | Path,
    ext: int = 1,
    force: bool = False,
) -> None:
    """
    Appends photon records from a source FITS table to a target FITS table in-place.
    The photons must be contained in the input `ext` extension data of both files.

    Validates both input files and appends the photon array data from the source
    file into the destination file in-place.
    The operation requires confirmation prompt from the user (by default).

    Args:
        append_to (str | Path):
            The path to the destination FITS file that will be modified.
        take_from (str | Path):
            The path to the source FITS file containing the photons to copy.
        ext (int, optional (default=`1`)):
            FITS files extension in which the update is performed/data is taken.
        force (bool, optional (default=`False`)):
            If True, bypasses the terminal user-prompt interaction check.

    Raises:
        ValueError: If file validation fails or table schemas mismatch.
        TypeError: If target HDU 1 does not contain structured tabular data.

    ## Notes
        This operation modifies the `append_to` file **in-place**. Ensure to have
        simulation data backups before running this function. Both FITS tables
        must share identical column structures for the numpy append to succeed.
    """
    for f in (append_to, take_from):
        if not validate_fits(f): raise ValueError(f"Invalid input FITS file '{f}'.")
    
    if not force:
        confirm = input(
            f"Photon list in '{append_to}' will be updated with data from '{take_from}'.\n"
            f'Continue? (y/n): '
        )
        if confirm.lower() not in ('y', 'yes', 'daje'):
            print('Aborted photons list update...')
            return
    
    print('Appending new photons...')
    with (
        fits.open(append_to, mode="update") as hdu_at,
        fits.open(take_from, mode="readonly") as hdu_tf,
    ):
        if not isinstance(hdu_at[ext], fits.BinTableHDU):
            raise TypeError(f"Invalid target file HDU ext {ext} format, must be 'BinTableHDU'.")

        tab_at: fits.BinTableHDU = hdu_at[ext]
        tab_tf: fits.BinTableHDU = hdu_tf[ext]

        if tab_at.columns.names != tab_tf.columns.names:
            raise ValueError('BinTable schema mismatch between target HDU and source HDU.')

        # - to update the target table efficiently we create an empty BinTable and fill it
        #   by using vectorised slicing (faster and cleaner than brutal `np.append`)
        # - the new BinTable inherits TUNIT, TFORM, and metadata formatting from target HDU
        orig_rows, append_rows = map(len, (tab_at.data, tab_tf.data))
        total_rows = orig_rows + append_rows

        print(f'Photons list elements: {orig_rows}')
        optim_hdu = fits.BinTableHDU.from_columns(tab_at.columns, nrows=total_rows)
        for col in tab_at.columns.names:
            optim_hdu.data[col][:orig_rows] = tab_at.data[col]
            optim_hdu.data[col][orig_rows:] = tab_tf.data[col]
        
        # - preserve header kws and ignores previous index keys in original target HDU
        ignored_kws = {
            'XTENSION', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'PCOUNT', 'GCOUNT', 'TFIELDS',
        }
        for key, value in tab_at.header.items():
            if key not in ignored_kws and key not in optim_hdu.header:
                optim_hdu.header[key] = value

        # overwrite HDU extension with updated BinTable and flush out to disk safely
        hdu_at[ext] = optim_hdu
        hdu_at.flush()
        print(f'Updated photons list elements: {total_rows}')

    print('Photons added succesfully!')
    return




def main() -> None:

    SIMUL_PATH = "/mnt/d/PhD_AASS/Coding/Images_fits/CameraGeometry"
    BASELINE_SIMUL = "mask_Z2arcmin_2-50keV_1ks"
    CXB_SIMUL = "mask_Z2arcmin_cxb_2-50keV_1ks"

    DATASET = 'reconstructed'
    ID_CAMERA = 'cam1a'

    trg_sources_file = f'{SIMUL_PATH}/{BASELINE_SIMUL}/{ID_CAMERA}/{BASELINE_SIMUL}_{ID_CAMERA}_{DATASET}.fits'
    src_cxb_file = f'{SIMUL_PATH}/{CXB_SIMUL}/{ID_CAMERA}/{CXB_SIMUL}_{ID_CAMERA}_{DATASET}.fits'

    update_photons_list(trg_sources_file, src_cxb_file)


if __name__ == '__main__':
    main()


# end