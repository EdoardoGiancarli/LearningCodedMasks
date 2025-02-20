from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
from astropy.io import fits
from tqdm import tqdm
import pickle

from mbloodmoon.coords import shift2equatorial
from mbloodmoon.io import _validate_fits
from mbloodmoon.images import _shift
import mbloodmoon as bm



def perform_IROS(simul_data: str | Path,
                 mask_file: str | Path,
                 camA: str,
                 camB: str,
                 dataset: str = "reconstructed",
                 max_iterations: int = 10,
                 snr_threshold: int | float = 5,
                 upscale_y: int = 1,
                 upscale_x: int = 5,
                 ) -> dict:
    """
    Stand-alone method that initializes the IROS algorithm and returns a log
    with all the useful info.

    Args:
        - simul_data: str | Path
        Path to the directory with simulations from WFM cameras.
        - mask_file: str | Path
        Path to the FITS file for the WFM mask.
        - camA, camB: str
        Camera A and B of the WFM (e.g. cam1a and cam1b).
        - dataset: str, default = `reconstructed`
        Which dataset to analyze. Either `detected` (simulated data prior to
        reconstruction) or `reconstructed` (position-reconstructed data).
        - max_iterations: int, default = 10
        Maximum number of source removal iterations to perform.
        - snr_threshold: float, default = 5
        SNR threshold for stopping IROS.
        - upscale_x: int, default = 5 
        Upscaling factor over the x direction.
        - upscale_y: int, default = 1
        Upscaling factor over the y direction.
    
    Returns:
        - iros_log: dict
        Dictionary containing useful info about the IROS reconstruction.
        The dict contains two dictionaries (for camera A and B of the WFM) in
        which are stored:
            1. source pos in [px]
            2. source angles shifts wrt optical axis in [rad]
            3. source RA and DEC in [deg]
            4. source detected fluence [ph]
            5. source detected flux [ph/cm^{2}/s]
            6. source detected rate [ph/s]
            7. effective counts subtracted from the detector [ph] TODO
            8. estimated counts before the mask [ph] TODO
            9. reconstructed source SNR TODO
            10. source pos fit parameter TODO
            11. Header of the FITS for the simulated data
    
    Notes:
        - This is just an auxiliary method to run IROS.
        - We can also extract the `init_log()` and `update_log()` inner
          methods, if it's more convenient, and write them better.
    """

    def init_log() -> dict:
        """Initializes the log dict structure."""

        def template(fmt: str, unit: str) -> dict:
            return {"data": [], "format": fmt, "unit": unit}     #TODO: insert errors?

        print("## Initializing Log...")
        init_keys = {
            "y": template("J", "px"),
            "x": template("J", "px"),
            "theta_y": template("D", "rad"),
            "theta_x": template("D", "rad"),
            "ra": template("D", "rad"),
            "dec": template("D", "rad"),
            "fluence": template("D", "ph"),
            "flux": template("D", "ph/cm2/s"),
            "rate": template("D", "ph/s"),
            "sub_fluence": template("D", "ph"),
            "est_fluence": template("D", "ph"),
            "SNR": template("D", ""),
            "chisquare": template("D", ""),
        }

        return {camera: deepcopy(init_keys) for camera in [camA, camB]}
    
    def update_log(rec_source: tuple[tuple, tuple],
                   source_snr: tuple[float, float],
                   chisquare: tuple[float, float],
                   ) -> None:
        """Updates log with the IROS output."""
        
        def _get_eff_area(shift_x, shift_y):
            """
            Computes the effective area seen by the source on the
            detector to compute the flux.

            Notes: TODO
                - This should be done inside IROS while finding the source
                  counts, to overcome the inner pos-depending reconstruction.
            """
            def shift(x, y):
                I_bulk = np.zeros(wfm.detector_shape)
                I_bulk[wfm.bulk > 0] = 1
                return _shift(wfm.bulk, (-x, -y))*I_bulk

            shift_x_px = int(shift_x/wfm.specs["mask_deltax"])
            shift_y_px = int(shift_y/wfm.specs["mask_deltay"])
            shifted_bulk = shift(shift_x_px, shift_y_px)
            eff_area = px_area*shifted_bulk.sum()                       # effective detector area seen by the source [cm^2]
            return eff_area

        keys = iros_log[camA].keys()
        for idx, camera in enumerate(iros_log.keys()):
            shiftx, shifty, counts = rec_source[idx]                    # shifts [mm], counts
            y, x = bm.shift2pos(wfm, shiftx, shifty)                    # pos in px
            thetay, thetax = np.arctan(shifty/l), np.arctan(shiftx/l)   # pos in angles wrt axis [rad] (TODO: from IROS)
            ra, dec = shift2equatorial(sdls[idx], wfm, shiftx, shifty)  # RA, DEC [rad]
            rate = counts/exposure[idx]                                 # rate [ph/s]
            flux = rate/_get_eff_area(shiftx, shifty)                   # flux [ph/cm^2/s] (TODO: correct counts for eff area inside IROS)
            sub_counts = -100                                           # effective counts subtracted from the detector (TODO: from IROS)
            est_counts = -100                                           # estimated counts from the source (before the mask)  (TODO: from IROS)
            snr = source_snr[idx]                                       # snr source (TODO: from IROS)
            chi = chisquare[idx]                                        # chi-square fit source (TODO: from IROS)

            q = [y, x, thetay, thetax, ra, dec, counts, flux,
                 rate, sub_counts, est_counts, snr, chi]
            for key, i in zip(keys, q):
                iros_log[camera][key]["data"].append(i)
    
    def data_to_array(log) -> dict:
        """Converts the lists in the log in arrays."""
        keys = log[camA].keys()
        for camera in [camA, camB]:
            for key in keys:
                log[camera][key]["data"] = np.asarray(log[camera][key]["data"])
        return log
    

    # get obs data and init log
    filepaths = bm.simulation_files(simul_data)
    wfm = bm.codedmask(mask_file, upscale_x=upscale_x, upscale_y=upscale_y)
    sdlA = bm.simulation(filepaths[camA][dataset])
    sdlB = bm.simulation(filepaths[camB][dataset])
    sdls = [sdlA, sdlB]

    px_area = 1e-2*wfm.specs["mask_deltax"]*wfm.specs["mask_deltay"]/(upscale_x*upscale_y)  # px area [cm^2] TODO: units with astropy
    exposure = [data.header["EXPOSURE"] for data in sdls]                                   # camera exposure [s]
    l = wfm.specs["mask_detector_distance"]                                                 # mask-detector distance [mm]
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

    #TODO: extract other info from IROS loop
    # - SNR in bm.optim.iros() -> subtract() or out
    # - chi^2 in bm.optim.iros() -> subtract() -> optimize() (IDEA: computed by hand from errors in the minimized function)
    # - could be also inserted into sources tuples
    for sources, residuals in tqdm(loop):
        snrs, chis = (-100, -100), (-100, -100)
        update_log(sources, snrs, chis)
    
    iros_log = data_to_array(iros_log)
    for camera, sdl in zip([camA, camB], sdls):
        iros_log[camera]["info"] = sdl.header

    return iros_log


def save_iros_output(data: dict,
                     mask_file: str | Path,
                     save_to: str | Path,
                     ) -> None:
    """
    Saves the IROS output data into a FITS file.

    Args:
        - data: dict
        IROS data output from `perform_IROS()`.
        - mask_file: str | Path
        Path to the FITS file for the WFM mask.
        - save_to: str | Path
        Path to the directory for saving the FITS file.
    """
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

    print("# Saving data...")
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
    Saves data in pickle format.

    Args:
        - data: object
        Data to save.
        - save_to: str | Path
        Path to the directory for saving the pickle file.
    """
    print("# Saving data...")
    with open(save_to + ".pickle", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("# Saving completed!")



def load_iros_output(filepath: str | Path) -> dict:
    """
    Loads the IROS output FITS file and converts it to a dict
    with the same structure described in `perform_IROS()`.

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
            hdu['EXTNAME'].lower(): {
                hdu["TTYPE" + str(idx)].lower(): {
                    "data": hdu_data.field(idx - 1),
                    "format": hdu["TFORM" + str(idx)],
                    "unit": hdu["TUNIT" + str(idx)] if "TUNIT" + str(idx) in hdu.keys() else "",
                }
                for idx in range(1, len(hdus_data[0][0]) + 1)
            }
            for hdu, hdu_data in zip(hdus, hdus_data)
        }

        for camera, hdu in zip(data.keys(), hdus):
            data[camera]["info"] = hdu

        return data
    
    print("# Loading data...")
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if check_fits(filepath):
        print("# Loading completed!")
        return load_data(filepath)


def load_pickle(filepath: str | Path) -> object:
    """
    Loads data from pickle file.
    
    Args:
        - filepath: str | Path
        Path to the pickle file.
    """
    print("# Loading data...")
    with open(filepath + ".pickle", "rb") as handle:
        data = pickle.load(handle)
    print("# Loading completed!")
    return data



def compare_w_catalog(simul_data: str | Path,
                      data: dict,
                      camA: str = "cam1a",
                      camB: str = "cam1b",
                      min_flux: float = 0.0,
                      ) -> dict:
    """
    Compares the reconstructed IROS sources data with the catalog
    containing the simulated sources.

    Args:
        - simul_data: str | Path
        Path to the directory with simulations from WFM cameras.
        - data: dict
        IROS data output from `perform_IROS()`.
        - camA: str, default = `cam1a`
        Camera A of the WFM.
        - camB: str, default = `cam1b`
        Camera B of the WFM.
        - min_flux: float, default = 0
        Threshold for the minimum flux to be extracted from the
        given catalogs (to associate the IROS sources).

    Returns:
        - data: dict
        Updated input data with source name matched from the catalog
        and source flux recovered from the catalog.
        - double_cam: dict
        Dictionary with common sources observed by the two cameras,
        with RA, DEC and flux from each camera.

    Raises:
        - FileNotFoundError: if FITS files do not exist.
        - ValueError: if files not in valid FITS format.
    """
    
    def get_catalogs(filepath: Path) -> tuple:
        """Returns the catalogs data."""

        def check_fits(pattern: Path) -> bool:
            """Check presence and validity of the FITS file."""
            if not pattern.is_file():
                raise FileNotFoundError("FITS file does not exists.")
            elif not _validate_fits(pattern):
                raise ValueError("File not in valid FITS format.")
            return True
        
        paths = bm.simulation_files(filepath)
        pA = Path(paths[camA]["sources"])
        pB = Path(paths[camB]["sources"])
        if check_fits(pA) and check_fits(pB):
            catA = fits.getdata(pA)
            catB = fits.getdata(pB)
            return catA, catB
    
    def single_cam_comparison(catalogs: list) -> None:
        """
        Compares respective catalogs for the two cameras and
        updates the input data dictionary.
        """

        def optimized_pos(catalog: dict,
                          pos: tuple[float, float],
                          ) -> int:
            """Source association from catalog."""
            arg = np.argmin(
                np.square(catalog["RA"] - pos[0]) + np.square(catalog["DEC"] - pos[1])
            )
            return arg
        
        for catalog, camera in zip(catalogs, [camA, camB]):
            catalog = catalog[catalog["FLUX"] > min_flux]
            data[camera]["catalog_name"] = []
            data[camera]["catalog_flux"] = []

            for ra, dec in zip(data[camera]["ra"]["data"], data[camera]["dec"]["data"]):
                argsource = optimized_pos(catalog, (ra, dec))
                data[camera]["catalog_name"].append(catalog["NAME"][argsource])
                data[camera]["catalog_flux"].append(catalog["FLUX"][argsource])

    def double_cam_comparison() -> dict:
        """Identifies common sources observed by the WFM cameras."""
        # TODO:
        #   - maybe for the whole comparison is better to transform the input data dict
        #     in a Pandas/Polaris dataframe, for a better management of the data itself
        #   - could be also helpful for this CAM comparison (if there are sources detected
        #     only by one of the two camera pair)
        #   - as of now, I assume that IROS reconstructs only the same source for both cameras
        #     and so the data from single CAM is "aligned" in the input dict (still checked, though)

        max_len = min(len(data[camA]["catalog_name"]), len(data[camB]["catalog_name"]))
        double_cam = {
            "source": [],
            **{f"{key}_{cam}": []
               for cam in [camA, camB]
               for key in ["ra", "dec", "flux"]}
            }

        for idx in range(max_len):
            name = data[camA]["catalog_name"][idx]
            if name == data[camB]["catalog_name"][idx]:
                double_cam["source"].append(name)
                for cam in [camA, camB]:
                    for key in ["ra", "dec", "flux"]:
                        double_cam[f"{key}_{cam}"].append(data[cam][key]["data"][idx])
        
        return double_cam

    print("## Comparing with Catalog...")
    if not isinstance(simul_data, Path):
        simul_data = Path(simul_data)
    
    catA, catB = get_catalogs(simul_data)
    single_cam_comparison([catA, catB])
    double_cam = double_cam_comparison()
    print("## Successful comparison!")
    
    return data, double_cam


# end