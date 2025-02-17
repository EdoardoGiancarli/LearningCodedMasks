from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
import pickle

from mbloodmoon.images import argmax
from mbloodmoon.coords import shift2equatorial
import mbloodmoon as bm


def perform_IROS(simul_data: str,
                 mask_file: str,
                 cam_a: str,
                 cam_b: str,
                 dataset: str = "reconstructed",
                 max_iterations: int = 10,
                 snr_threshold: int | float = 5,
                 upscale_y: int = 1,
                 upscale_x: int = 5,
                 ) -> dict:
    
    def init_log() -> dict:
        print("## Initializing Log...")
        def template(fmt: str, unit: str) -> dict:
            return {"data": [], "format": fmt, "unit": unit}     #TODO: insert errors?

        keys = {
            "y": template("J", "px"),              # coords in [px, px]
            "x": template("J", "px"),
            "theta_y": template("F", "rad"),       # coords in angles
            "theta_x": template("F", "rad"),
            "ra": template("F", "rad"),            # RA and DEC
            "dec": template("F", "rad"),
            "fluence": template("F", "ph"),        # fluence at the peak for source choice
            "flux": template("F", "ph/cm^2/s"),
            "rate": template("F", "ph/s"),
            "sub_fluence": template("F", "ph"),    # effective counts subtracted from the detector
            "est_fluence": template("F", "ph"),    # estimated counts from the source (before the mask)
            "SNR": template("F", ""),
            "chisquare": template("F", ""),
        }

        return {camera: deepcopy(keys) for camera in [cam_a, cam_b]}
    
    def update_log(rec_source: tuple[tuple, tuple],
                   source_snr: tuple[float, float],
                   chisquare: tuple[float, float],
                   sdls: list,
                   exposure: float,
                   area: float,
                   ) -> None:
        keys = iros_log[cam_a].keys()
        for idx, camera in enumerate(iros_log.keys()):
            shiftx, shifty, counts = rec_source[idx]
            y, x = bm.shift2pos(wfm, shiftx, shifty)                    # pos in px
            ra, dec = shift2equatorial(sdls[idx], wfm, shiftx, shifty)  # RA, DEC
            thetay = ra - sdls[idx].pointings["z"].ra                   # TODO: pos in angles wrt axis
            thetax = dec - sdls[idx].pointings["z"].dec
            rate = counts/exposure[idx]                                 # TODO: rate
            flux = rate/area                                            # TODO: flux
            sub_counts = -100                                           # TODO: subtracted fluence
            est_counts = -100                                           # TODO: estimated source counts
            snr = source_snr[idx]                                       # snr source
            chi = chisquare[idx]                                        # chi-square fit source

            q = [y, x, thetay, thetax, ra, dec, counts, flux,
                rate, sub_counts, est_counts, snr, chi]
            for key, i in zip(keys, q):
                iros_log[camera][key]["data"].append(i)
    
    def f(log):
        keys = log[cam_a].keys()
        for camera in [cam_a, cam_b]:
            for key in keys:
                log[camera][key]["data"] = np.asarray(log[camera][key]["data"])
        return log
    

    # get obs data and init log
    filepaths = bm.simulation_files(simul_data)
    wfm = bm.codedmask(mask_file, upscale_x=upscale_x, upscale_y=upscale_y)
    sdlA = bm.simulation(filepaths[cam_a][dataset])
    sdlB = bm.simulation(filepaths[cam_b][dataset])
    sdls = [sdlA, sdlB]
    exposure = [data.header["EXPOSURE"] for data in sdls]                                # camera exposure [s] 
    area = 1e-2*wfm.specs["mask_deltax"]*wfm.specs["mask_deltay"]/(upscale_x*upscale_y)  # px area [cm^2]
    iros_log = init_log()

    # IROS
    print("## Looping around the FOV...")
    loop = bm.iros(
        camera=wfm,
        sdl_cam1a=sdlA,
        sdl_cam1b=sdlB,
        max_iterations=max_iterations,
        snr_threshold=snr_threshold,
        dataset=dataset,
        )

    #TODO: fix IROS loop
    # - SNR in bm.optim.iros() -> subtract() or out
    # - chi^2 in bm.optim.iros() -> subtract() -> optimize()
    for sources, residuals in tqdm(loop):
        snrs, chis = (-100, -100), (-100, -100)
        update_log(sources, snrs, chis,
                   sdls, exposure, area)
    
    iros_log = f(iros_log)
    for camera, sdl in zip([cam_a, cam_b], sdls):
        iros_log[camera]["info"] = sdl.header

    return iros_log



def save_iros_output(data: dict,
                     mask_file: str | Path,
                     save_to: str | Path,
                     ) -> None:
    """
    Saves the IROS output data.
    """
    print("# Saving data...")

    def make_column(name: str,
                    col_data: Sequence,
                    data_format: str,
                    unit: str,
                    ) -> fits.Column:
        return fits.Column(name=f"{name.upper()}", array=col_data,
                           format=data_format, unit=unit)

    def make_bintable(name: str,
                      tab_data: list,
                      sdl_header: fits.Header,
                      ) -> fits.BinTableHDU:
        table_hdu = fits.BinTableHDU.from_columns(
            columns=tab_data,
            header=sdl_header,
            name=f"{name.upper()}",
        )
        return table_hdu

    # HDU list and Primary Header
    hdu_list = fits.HDUList([])
    primary_header = fits.getheader(mask_file, ext=2)
    primary_header['EXTNAME'] = 'PRIMARY'
    primary_hdu = fits.PrimaryHDU(header=primary_header)
    hdu_list.append(primary_hdu)

    # BinTables
    for camera in data.keys():
        cam = data[camera]
        columns = [
            make_column(key, cam[key]["data"], cam[key]["format"], cam[key]["unit"])
            for key in list(cam.keys())[:-1]
        ]
        table_hdu = make_bintable(camera, columns, cam["info"])
        hdu_list.append(table_hdu)
    
    # save data
    hdu_list.writeto(save_to)
    print("# Saving completed!")



def save_pickle(data: object, save_to: str | Path) -> None:
    """
    Saves data in .pickle format.

    Args:
        - data: object
        Data to save.
        - save_to: str | Path
        Path to save the data
    """
    print("# Saving data...")
    with open(save_to + ".pickle", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("# Saving completed!")



def load_iros_output(filepath: str | Path) -> dict:
    print("# Loading data...")

    print("# Loading completed!")


def load_pickle(filepath: str | Path) -> object:
    """
    Loads data from .pickle file.
    
    Args:
        - filepath: str | Path
    """
    print("# Loading data...")
    with open(filepath + ".pickle", "rb") as handle:
        data = pickle.load(handle)
    print("# Loading completed!")
    return data



def compare_w_catalog() -> dict:
    pass


# end