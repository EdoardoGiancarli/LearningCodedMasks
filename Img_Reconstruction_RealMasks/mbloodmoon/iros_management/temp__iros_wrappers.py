"""
Collections of wrappers for IROS testing and analyses.
Once validated, these methods shall go in bloodmoon.

Contents:
    - IrosLog: IROS parameters log management.
    - gen_log(): initializes an IrosLog instance.
    - perform_iros(): perform the IROS loop and stores output.
    - computes_params(): takes IROS output and compute parameters.
    - compare_w_catalog(): compares IROS reconstruction with given catalog.


    - iros_sky(): creates sky image from reconstructed sources.
    
    - save_iros_output(): saves `perform_iros()` output.
    - load_iros_output(): loads `perform_iros()` output.
    - save_iros_data(): saves `computes_params()` output.
    - load_iros_data(): loads `computes_params()` output.
    - HELPER: save_pickle() -> saves pickle file.
    - HELPER: load_pickle() -> loads pickle file.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
from astropy.io import fits
from tqdm import tqdm
import pickle

from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points
from astropy.wcs import WCS
from mbloodmoon.coords import pos2equatorial

from mbloodmoon.io import SimulationDataLoader
from mbloodmoon.mask import CodedMaskCamera

from mbloodmoon.coords import shift2equatorial
from mbloodmoon.io import _validate_fits
from mbloodmoon.images import _shift, argmax
import mbloodmoon as bm

import matplotlib.pyplot as plt







































def save_iros_output(
    data: dict,
    mask_file: str | Path,
    save_to: str | Path,
) -> None:
    """
    Saves IROS output into a FITS file.

    Args:
        - data: dict
        IROS data output from `perform_iros()`.
        - mask_file: str | Path
        Path to the FITS file for the WFM mask.
        - save_to: str | Path
        Path to the directory for saving the FITS file.
    """
    def make_column(
        name: str,
        col_data: np.array,
    ) -> fits.Column:
        return fits.Column(name=f"{name.upper()}", array=col_data, format="D")

    def make_bintable(
        name: str,
        tab_data: list,
    ) -> fits.BinTableHDU:
        table_hdu = fits.BinTableHDU.from_columns(
            columns=tab_data,
            name=f"{name.upper()}",
        )
        return table_hdu
    
    print("# Saving data...")
    # HDU list and Primary Header
    hdu_list = fits.HDUList([])
    primary_header = fits.getheader(mask_file, ext=2)
    primary_header["EXTNAME"] = "PRIMARY"
    primary_hdu = fits.PrimaryHDU(header=primary_header)
    hdu_list.append(primary_hdu)

    # BinTables for data
    for camera in data.keys():
        cam = data[camera]
        columns = [
            make_column(key, cam[key]) for key in list(cam.keys())
        ]
        table_hdu = make_bintable(camera, columns)
        hdu_list.append(table_hdu)

    # save data
    hdu_list.writeto(save_to, output_verify="fix+ignore")
    hdu_list.close()
    print("# Saving completed!")




def load_iros_output(
    filepath: str | Path,
) -> dict:
    """
    Loads IROS output FITS file and converts it to a dict
    with the same structure described in `perform_iros()`.

    Args:
        - filepath: str | Path
        Path to the FITS file.

    Returns:
        - data: dict
        Dictionary with info for the sources observed by the WFM
        and reconstructed with IROS.

    Raises:
        - FileNotFoundError: if FITS file does not exists.
        - ValueError: if file not in valid FITS format.
    """
    def check_fits(filepath: Path) -> bool:
        """Check presence and validity of the FITS file."""
        if not filepath.is_file():
            raise FileNotFoundError("FITS file does not exists.")
        elif not _validate_fits(filepath):
            raise ValueError("File not in valid FITS format.")
        return True

    def load_data(filepath: Path) -> dict:
        """Open FITS and store info in a dictionary."""
        with fits.open(filepath) as hdul:
            hdus = [dict(hdul[1].header), dict(hdul[2].header)]
            hdus_data = [hdul[1].data, hdul[2].data]

        data = {
            hdu["EXTNAME"].lower(): {
                hdu["TTYPE" + str(idx + 1)].lower(): hdu_data.field(idx)
                for idx in range(len(hdus_data[0][0]))
            }
            for hdu, hdu_data in zip(hdus, hdus_data)
        }
        return data

    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if check_fits(filepath):
        print("# Loading data...")
        data = load_data(filepath)
        print("# Loading completed!")
        return data


def save_iros_data(
    data: dict,
    mask_file: str | Path,
    sdls: tuple[object],
    save_to: str | Path,
) -> None:
    """
    Saves the computed parameter from IROS into a FITS file.

    Args:
        - data: dict
        IROS data output from `compute_params()`.
        - mask_file: str | Path
        Path to the FITS file for the WFM mask.
        - sdls: tuple(SimulationDataLoader)
        SDL instances for the cameras of the WFM.
        - save_to: str | Path
        Path to the directory for saving the FITS file.
    """

    def make_column(
        name: str,
        col_data: np.array,
        data_format: str,
        unit: str,
    ) -> fits.Column:
        return fits.Column(name=f"{name.upper()}", array=col_data, format=data_format, unit=unit)

    def make_bintable(
        name: str,
        tab_data: list,
        sdl_header: fits.Header,
    ) -> fits.BinTableHDU:
        table_hdu = fits.BinTableHDU.from_columns(
            columns=tab_data,
            header=sdl_header,
            name=f"{name.upper()}",
        )
        return table_hdu

    print("# Saving data...")
    # HDU list and Primary Header
    hdu_list = fits.HDUList([])
    primary_header = fits.getheader(mask_file, ext=2)
    primary_header["EXTNAME"] = "PRIMARY"
    primary_hdu = fits.PrimaryHDU(header=primary_header)
    hdu_list.append(primary_hdu)

    # BinTables
    for camera, sdl in zip(data.keys(), sdls):
        cam = data[camera]
        columns = [
            make_column(key, cam[key]["data"], cam[key]["format"], cam[key]["unit"])
            for key in list(cam.keys())
        ]
        table_hdu = make_bintable(camera, columns, sdl.header)
        hdu_list.append(table_hdu)

    # save data
    hdu_list.writeto(save_to, output_verify="fix+ignore")
    hdu_list.close()
    print("# Saving completed!")


def load_iros_data(
    filepath: str | Path,
) -> dict:
    """
    Loads the IROS computed parameters FITS file and converts it to
    a dict with the same structure described in `compute_params()`.

    Args:
        - filepath: str | Path
        Path to the FITS file.

    Returns:
        - data: dict
        Dictionary with info for the sources observed by the WFM
        and reconstructed with IROS.

    Raises:
        - FileNotFoundError: if FITS file does not exists.
        - ValueError: if file not in valid FITS format.
    """

    def check_fits(filepath: Path) -> bool:
        """Check presence and validity of the FITS file."""
        if not filepath.is_file():
            raise FileNotFoundError("FITS file does not exists.")
        elif not _validate_fits(filepath):
            raise ValueError("File not in valid FITS format.")
        return True

    def load_data(filepath: Path) -> dict:
        """Open FITS and store info in a dictionary."""
        with fits.open(filepath) as hdul:
            hdus = [dict(hdul[1].header), dict(hdul[2].header)]
            hdus_data = [hdul[1].data, hdul[2].data]
        data = {
            hdu["EXTNAME"].lower(): {
                hdu["TTYPE" + str(idx)].lower(): {
                    "data": hdu_data.field(idx - 1),
                    "format": hdu["TFORM" + str(idx)],
                    "unit": hdu["TUNIT" + str(idx)] if "TUNIT" + str(idx) in hdu.keys() else "",
                }
                for idx in range(1, len(hdus_data[0][0]) + 1)
            }
            for hdu, hdu_data in zip(hdus, hdus_data)
        }
        return data

    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if check_fits(filepath):
        print("# Loading data...")
        data = load_data(filepath)
        print("# Loading completed!")
        return data


def save_sky(
    sky: np.array,
    snr: np.array,
    sdl: SimulationDataLoader,
    save_to: str | Path,
    wcs: WCS = None,
) -> None:
    """Saves sky array to FITS Image."""

    sky, snr = np.int16(sky), np.float32(snr)
    print("# Saving Sky...")
    # HDU list and Primary Header
    hdu_list = fits.HDUList([])
    primary_hdu = fits.PrimaryHDU()
    hdu_list.append(primary_hdu)

    # Images for data
    for img, name in zip([sky, snr], ["sky", "snr"]):
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


def load_sky(
    filepath: str | Path,
) -> tuple[np.array, np.array]:
    """Loads sky and its SNR from FITS."""
    def check_fits(filepath: Path) -> bool:
        """Check presence and validity of the FITS file."""
        if not filepath.is_file():
            raise FileNotFoundError("FITS file does not exists.")
        elif not _validate_fits(filepath):
            raise ValueError("File not in valid FITS format.")
        return True

    def load_data(filepath: Path) -> dict:
        """Open FITS and store Images in 2D-array."""
        with fits.open(filepath) as hdu:
            sky = hdu[1].data
            snr = hdu[2].data
        return sky, snr
    
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if check_fits(filepath):
        print("# Loading data...")
        sky, snr = load_data(filepath)
        print("# Loading completed!")
        return sky, snr





# end




#def double_cam_comparison() -> dict:
#    """Identifies common sources observed by the WFM cameras."""
#    # TODO:
#    #   - maybe for the whole comparison is better to transform the input data dict
#    #     in a Pandas/Polaris dataframe, for a better management of the data itself
#    #   - could be also helpful for this CAM comparison (if there are sources detected
#    #     only by one of the two camera pair)
#    #   - as of now, I assume that IROS reconstructs only the same source for both cameras
#    #     and so the data from single CAM is "aligned" in the input dict (still checked, though)
#
#    max_len = min(len(data[camA]["catalog_name"]), len(data[camB]["catalog_name"]))
#    double_cam = {"source": [], **{f"{key}_{cam}": [] for cam in [camA, camB] for key in ["ra", "dec", "flux"]}}
#
#    for idx in range(max_len):
#        name = data[camA]["catalog_name"][idx]
#        if name == data[camB]["catalog_name"][idx]:
#            double_cam["source"].append(name)
#            for cam in [camA, camB]:
#                for key in ["ra", "dec", "flux"]:
#                    double_cam[f"{key}_{cam}"].append(data[cam][key]["data"][idx])
#
#    return double_cam











## BinTables for sky residues
#for camera in data.keys():
#    skyres = data[camera]["sky_residues"]
#    values = skyres.ravel()
#    y, x = np.unravel_index(np.arange(skyres.size), skyres.shape)
#    columns = [
#        make_column(key, col, frmt) for key, col, frmt in zip(
#            ["value", "y", "x"], [values, y, x], ["D", "J", "J"],
#        )
#    ]
#    table_hdu = make_bintable(camera + "_skyres", columns)
#    table_hdu.header["ZEROEL"] = "Top-left (C-ordering, Row-major from Python)"
#    table_hdu.header["ROWS"], table_hdu.header["COLS"] = skyres.shape
#    hdu_list.append(table_hdu)
#
#
#def get_sky(
#    hdu_data: fits.FITS_rec,
#    sky_shape: tuple,
#) -> np.array:
#    values = hdu_data.field(0)
#    y, x = hdu_data.field(1), hdu_data.field(2)
#    sky = np.zeros(sky_shape); sky[y, x] = values
#    return sky