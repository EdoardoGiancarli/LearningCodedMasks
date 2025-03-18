"""
IROS output data handling.
"""

from pathlib import Path

import numpy as np
import pickle
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points
from astropy.wcs import WCS
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject.mosaicking import reproject_and_coadd

from mbloodmoon.io import SimulationDataLoader
from mbloodmoon.mask import CodedMaskCamera

from mbloodmoon.io import _validate_fits
from mbloodmoon.coords import pos2equatorial
#from mbloodmoon.coords import shift2equatorial


def _make_column(
    name: str,
    data: np.array,
    frmt: str = None,
    unit: str = None,
) -> fits.Column:
    """
    Creates a FITS table column with the specified parameters.

    Args:
        name (str):
            Name of the column.
        data (np.array):
            Data to be stored in the column.
        frmt (str, optional (default=None)):
            FITS format of the column data.
        unit (str, optional (default=None)):
            Physical unit of the column data.

    Returns:
        column (fits.Column): FITS column instance.
    """
    column = fits.Column(
        name=f"{name.upper()}",
        array=data,
        format=frmt,
        unit=unit,
    )
    return column


def _make_bintable(
    name: str,
    columns: list[fits.Column],
    header: fits.Header = None,
) -> fits.BinTableHDU:
    """
    Creates a FITS binary table HDU.

    Args:
        name (str):
            Name of the binary table.
        columns (list[fits.Column]):
            List of FITS Column objects.
        header (fits.Header, optional (default=None)):
            FITS Header object for the table.

    Returns:
        table (fits.BinTableHDU): FITS binary table HDU.
    """
    table = fits.BinTableHDU.from_columns(
        columns=columns,
        header=header,
        name=f"{name.upper()}",
    )
    return table


def save_iros_output(
    data: dict,
    mask_file: str | Path,
    save_to: str | Path,
) -> None:
    """
    Saves IROS output into a FITS file.

    Args:
        data (dict):
            IROS data output from `perform_iros()`.
        mask_file (str | Path):
            Path to the FITS file for the WFM mask.
        save_to (str | Path):
            Path to the directory for saving the FITS file.
    """    
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
            _make_column(key, cam[key], "D") for key in list(cam.keys())
        ]
        table_hdu = _make_bintable(camera, columns)
        hdu_list.append(table_hdu)

    # save data
    hdu_list.writeto(save_to, output_verify="fix+ignore")
    hdu_list.close()
    print("# Saving completed!")


def save_iros_data(
    data: dict,
    mask_file: str | Path,
    sdls: tuple[SimulationDataLoader],
    save_to: str | Path,
) -> None:
    """
    Saves the computed parameter from IROS into a FITS file.

    Args:
        data (dict):
            IROS data output from `compute_params()`.
        mask_file (str | Path):
            Path to the FITS file for the WFM mask.
        sdls (tuple(SimulationDataLoader)):
            SDL instances for the cameras of the WFM (camA and camB).
        save_to (str | Path):
            Path to the directory for saving the FITS file._
    """
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
            _make_column(
                key, cam[key]["data"], cam[key]["format"], cam[key]["unit"],
            )
            for key in list(cam.keys())
        ]
        table_hdu = _make_bintable(camera, columns, sdl.header)
        hdu_list.append(table_hdu)

    # save data
    hdu_list.writeto(save_to, output_verify="fix+ignore")
    hdu_list.close()
    print("# Saving completed!")


def save_sky(
    sky: np.array,
    snr: np.array,
    sdl: SimulationDataLoader,
    save_to: str | Path,
    wcs: WCS = None,
) -> None:
    """
    Saves the given sky array and its significance (SNR) as FITS Image files,
    including optional World Coordinate System (WCS) information if provided.

    Args:
        sky (np.array):
            Sky data array to be saved.
        snr (np.array):
            Sky significance array.
        sdl (SimulationDataLoader):
            SimulationDataLoader instance providing additional metadata.
        save_to (str | Path):
            File path or directory where the FITS image will be saved.
        wcs (WCS, optional (default=None)):
            World Coordinate System instance, which can be used to
            include coordinate information in the FITS header.
    """

    sky, snr = np.int32(sky), np.float32(snr)
    print("# Saving sky...")
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


def save_pickle(data: object, save_to: str | Path) -> None:
    """
    Saves data in pickle format.

    Args:
        data (object):
            Data to save.
        save_to (str | Path):
            Path to the directory for saving the pickle file.
    """
    print("# Saving data...")
    with open(save_to, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("# Saving completed!")

"""
                                                           @       @  
                                                          @@@@    @@@ 
                                                        @@@@@@@@@@@@@ 
                                                       @@@@@@@@@@@@@@@
                                                       @@@@@@@@@@@@@@@
                                                       @@@@@@@@@@@@@@@
                                            @@@@@@@@@@@@@@@@@@@@@@@@@ 
                                          @@@@@@@@@@@@@@@@@@@@@@@@@@  
                                        @@@@@@@@@@@@@@@@@@@@@@@@@@@@  
                                       @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
                                      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
                                      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
                                     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
                                   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
       @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   
       @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@      
"""

def check_fits(filepath: Path) -> bool:
    """
    Check presence and validity of the FITS file.

    Args:
        filepath (str | Path): Path to the FITS file.
    
    Returns:
        output (bool): True if FITS exists and in valid format.

    Raises:
        FileNotFoundError: If FITS file does not exists.
        ValueError: If file not in valid FITS format.
    """
    if not filepath.is_file():
        raise FileNotFoundError("FITS file does not exists.")
    elif not _validate_fits(filepath):
        raise ValueError("File not in valid FITS format.")
    return True


def load_iros_output(filepath: str | Path) -> dict:
    """
    Loads IROS output FITS file and converts it to a dict
    with the same structure in `perform_iros()`.

    Args:
        filepath (str | Path):
            Path to the FITS file.

    Returns:
        data (dict):
            Database with info for the sources observed
            by the WFM and reconstructed with IROS.
    """
    def load_data(filepath: Path) -> dict:
        """Open FITS and store info in a dictionary."""
        with fits.open(filepath) as hdul:
            hdus = [dict(hdul[1].header), dict(hdul[2].header)]
            hdus_data = [hdul[1].data, hdul[2].data]

        data = {
            hdu["EXTNAME"].lower(): {
                hdu[f"TTYPE{idx + 1}"].lower(): hdu_data.field(idx)
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


def load_iros_data(filepath: str | Path) -> dict:
    """
    Loads the IROS computed parameters FITS file and converts it to
    a dict with the same structure described in `compute_params()`.

    Args:
        filepath (str | Path): Path to the FITS file.

    Returns:
        data (dict):
            Database with info for the sources observed
            by the WFM and reconstructed with IROS.
    """
    def load_data(filepath: Path) -> dict:
        """Open FITS and store info in a dictionary."""
        with fits.open(filepath) as hdul:
            hdus = [dict(hdul[1].header), dict(hdul[2].header)]
            hdus_data = [hdul[1].data, hdul[2].data]
        data = {
            hdu["EXTNAME"].lower(): {
                hdu[f"TTYPE{idx}"].lower(): {
                    "data": hdu_data.field(idx - 1),
                    "format": hdu[f"TFORM{idx}"],
                    "unit": hdu[f"TUNIT{idx}"] if f"TUNIT{idx}" in hdu.keys() else "",
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


def load_sky(filepath: str | Path) -> tuple[np.array]:
    """
    Loads sky and its SNR from FITS.

    Args:
        filepath (str | Path): Path to the FITS file.

    Returns:
        output (tuple):
            - sky (np.array): 2D array for the sky.
            - snr (np.array): sky significance.
    """
    def load_data(filepath: Path) -> dict:
        """Open FITS and store Images in 2D-array."""
        with fits.open(filepath) as hdu:
            sky, snr = hdu[1].data, hdu[2].data
        return sky, snr
    
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if check_fits(filepath):
        print("# Loading data...")
        sky, snr = load_data(filepath)
        print("# Loading completed!")
        return sky, snr


def load_pickle(filepath: str | Path) -> object:
    """
    Loads data from pickle file.

    Args:
        filepath (str | Path):
            Path to the pickle file.
    
    Returns:
        output (object): Loaded object.
    """
    print("# Loading data...")
    with open(filepath, "rb") as handle:
        data = pickle.load(handle)
    print("# Loading completed!")
    return data
































def fit_WCS(
    camera: CodedMaskCamera,
    sdl: SimulationDataLoader,
    pixels: list[tuple[int]] = None,
    grid_step: int = 200,
) -> WCS:
    """
    Fit the WCS for a camera of the WCS fitting given RA/DEC
    and sky pixels.

    Args:
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        sdl (SimulationDataLoader):
            SimulationDataLoader instance for the given camera.
        pixels (list[tuple[int]], optional (default=None)):
            List of pxs position for the WCS fit.
        grid_step (int, optional (default=200)):
            Step for the points in a sky grid for computing the fit.
    
    Returns:
        output (WCS):
            WCS instance with info on the coords fit.
    """
    n, m = camera.sky_shape
    pxs = pixels if pixels else [
        (grid_step*y, grid_step*x) for y in range(1, n//grid_step) for x in range(1, m//grid_step)
    ]

    coords = [pos2equatorial(sdl, camera, *pos) for pos in pxs]
    # WARNING: the next is not a typo, WCS wants the px indexes as (x, y)
    coord_pxs = tuple(np.array([px[idx] for px in pxs]) for idx in (1, 0))
    coord_radec = SkyCoord(
        ra=np.array([c.ra for c in coords]),
        dec=np.array([c.dec for c in coords]),
        frame="icrs", unit="deg",
    )
    wcs = fit_wcs_from_points(
        xy=coord_pxs,
        world_coords=coord_radec,
        projection="TAN",
        sip_degree=1,
        proj_point=SkyCoord(*sdl.pointings["z"], frame="icrs", unit="deg"),
    )
    return wcs


def camera_composition(
    skyA_path: str | Path,
    skyB_path: str | Path,
    save_to: str | Path,
) -> None:
    """
    Performs the composition of the WFM cameras skies and significances,
    including the reprojection of the World Coordinates System for RA/Dec.

    Specifically, it:
        - Opens the skies FITS file
        - Finds the optimal WCS fit and sky shape for the composition
        - Reprojects and sums the two skies making the composition
        - Reprojects and averages the two SNRs making the composition
        - Saves the composition FITS file

    Args:
        skyA_path (str, Path):
            File path for the camera A sky.
        skyB_path (str, Path):
            File path for the camera B sky.
        save_to (str, Path):
            File path or directory where the FITS image will be saved.

    Notes:
        - If the WCS fit keys are not present in the camera skies headers,
          a TypeError will be raised from `find_optimal_celestial_wcs()`:
        >>> TypeError: "WCS does not have celestial components."
    """
    with fits.open(skyA_path) as hduA, fits.open(skyB_path) as hduB:
        print("# Composing WFM skies...")
        skies, snrs = (hduA[1], hduB[1]), (hduA[2], hduB[2])
        wcs_out, shape_out = find_optimal_celestial_wcs(input_data=skies)

        sky_comp, _ = reproject_and_coadd(
            input_data=skies,
            output_projection=wcs_out,
            shape_out=shape_out,
            reproject_function=reproject_interp,
            combine_function="sum",
        )
        snr_comp, _ = reproject_and_coadd(
            input_data=snrs,
            output_projection=wcs_out,
            shape_out=shape_out,
            reproject_function=reproject_interp,
            combine_function="mean",
        )
        #snr_compA, _ = reproject_and_coadd(
        #    input_data=snrs,
        #    output_projection=wcs_out,
        #    shape_out=shape_out,
        #    reproject_function=reproject_interp,
        #    combine_function="first",
        #)
        #snr_compB, _ = reproject_and_coadd(
        #    input_data=snrs,
        #    output_projection=wcs_out,
        #    shape_out=shape_out,
        #    reproject_function=reproject_interp,
        #    combine_function="last",
        #)
        #snr_comp = np.sqrt(np.square(snr_compA) + np.square(snr_compB))

        #hduA[1].header.update(wcs_out.to_header())  # updating the header of CAMERA A
        hdu_list = fits.HDUList([fits.PrimaryHDU()])

        for img, name in zip([np.int32(sky_comp), np.float32(snr_comp)], ["sky", "snr"]):
            image_hdu = fits.ImageHDU(
                data=img,
                header=wcs_out.to_header(),
                name=name.upper(),
            )
            hdu_list.append(image_hdu)
        
        hdu_list.writeto(save_to, output_verify="fix+ignore")
        hdu_list.close()
        print("# WFM composition completed!")


# end