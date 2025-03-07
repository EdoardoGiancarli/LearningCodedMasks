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

from mbloodmoon.coords import shift2equatorial
from mbloodmoon.io import _validate_fits
from mbloodmoon.images import _shift, argmax
import mbloodmoon as bm

import matplotlib.pyplot as plt


class IrosLog:
    """IROS parameters log management."""

    def __init__(self, cams: tuple[str]):
        self._cams = cams
        self._log = self.make_log()

    def make_log(self) -> dict:
        """Makes the log dict structure."""

        def template(fmt: str, unit: str) -> dict:
            return {"data": [], "format": fmt, "unit": unit}

        init_keys = {
            "y": template("J", "px"),
            "x": template("J", "px"),
            "shift_x": template("D", "mm"),
            "dshift_x": template("D", "mm"),
            "shift_y": template("D", "mm"),
            "dshift_y": template("D", "mm"),
            "theta_x": template("D", "rad"),
            "dtheta_x": template("D", "rad"),
            "theta_y": template("D", "rad"),
            "dtheta_y": template("D", "rad"),
            "ra": template("D", "rad"),
            "dra": template("D", "rad"),
            "dec": template("D", "rad"),
            "ddec": template("D", "rad"),
            "fluence": template("D", "ph"),
            "dfluence": template("D", "ph"),
            "rate": template("D", "ph/s"),
            "drate": template("D", "ph/s"),
            "flux": template("D", "ph/cm2/s"),
            "dflux": template("D", "ph/cm2/s"),
            "obs_fluence": template("D", "ph"),
            "dobs_fluence": template("D", "ph"),
            "sub_fluence": template("D", "ph"),
            "simulphotons": template("D", "ph"),
            "snr": template("D", ""),
            "chisquare": template("D", ""),
        }

        return {camera: deepcopy(init_keys) for camera in self._cams}

    def initialize(self) -> dict:
        """Initializes the log."""
        return self._log

    def update(self, camera: str, params: list) -> None:
        """Updates the log."""
        keys = self._log[self._cams[0]].keys()
        for key, p in zip(keys, params):
            self._log[camera][key]["data"].append(p)


def perform_iros(
    wfm: object,
    sdlA: object,
    sdlB: object,
    cameras: tuple[str],
    max_iterations: int = 10,
    snr_threshold: int | float = 5,
    dataset: str = "reconstructed",
) -> tuple[dict, tuple[np.array, np.array]]:
    """Runs IROS loop and stores output."""

    def init_log() -> dict:
        """Initializes the log dict structure."""
        init_keys = {
            "shiftx": [], "shifty": [], "fluence": [],
            "snr": [], "obs_counts": [], "sub_counts": [],
        }
        return {camera: deepcopy(init_keys) for camera in cameras}
    
    def store_output(
        rec_source: tuple[tuple, tuple],
        obs_counts: tuple[float, float],
        sub_counts: tuple[float, float],
    ) -> None:
        """Stores sources info into log."""
        keys = log_output[cameras[0]].keys()
        for idx, camera in enumerate(log_output.keys()):
            params = [*rec_source[idx], obs_counts[idx], sub_counts[idx]]
            for key, p in zip(keys, params):
                log_output[camera][key].append(p)
    
    def data_to_array(log) -> dict:
        """Converts the log lists in arrays."""
        keys = log[cameras[0]].keys()
        for camera in list(log.keys()):
            for key in keys:
                if not isinstance(log[camera][key], np.ndarray):
                    log[camera][key] = np.asarray(log[camera][key])
        return log
    
    log_output = init_log()
    detectors = tuple(bm.count(wfm, sdl.data)[0] for sdl in [sdlA, sdlB])
    skies = tuple(bm.decode(wfm, d) for d in detectors)
    skies_max = [tuple(np.max(sky) for sky in skies)]
    skies = [skies]

    print("## Looping around the FOV...")
    loop = bm.iros(
        camera=wfm,
        sdl_cam1a=sdlA,
        sdl_cam1b=sdlB,
        max_iterations=max_iterations,
        snr_threshold=snr_threshold,
        dataset=dataset,
    )

    for sources, residuals in tqdm(loop):
        skies.append(residuals)
        skies_max.append(tuple(np.max(r) for r in residuals))
        obs_counts = skies_max[0]
        sub_counts = tuple(s.max() - r[*argmax(s)] for s, r in zip(skies[0], skies[1]))
        skies.pop(0); skies_max.pop(0)
        store_output(sources, obs_counts, sub_counts)

    return data_to_array(log_output), residuals


def gen_log(cams: tuple[str]) -> IrosLog:
    """Initializes an IrosLog instance."""
    return IrosLog(cams)


def compute_params(
    iros_output: dict,
    wfm: object,
    sdlA: object,
    sdlB: object,
    log: IrosLog,
) -> dict:
    """Computes useful parameters for IROS reconstructed sources."""

    def update_log(camera: str) -> None:

        def _get_params(
            shiftx: float,
            shifty: float,
            counts: float,
            snr: float,
            obs_counts: float,
            sub_counts: float,
        ) -> list:
            
            def _get_eff_area():
                """
                Computes the effective area seen by the source on the
                detector to compute the flux.
                """
                def shift(x, y):
                    I_bulk = np.zeros(wfm.detector_shape)
                    I_bulk[wfm.bulk > 0] = 1
                    return _shift(wfm.bulk, (x, y)) * I_bulk

                scalingy, scalingx = tuple(
                    d / s for d, s in zip(wfm.detector_shape, wfm.sky_shape)
                )
                shiftx_px = int(shiftx * scalingx / wfm.specs["mask_deltax"])
                shifty_px = int(shifty * scalingy / wfm.specs["mask_deltay"])
                shifted_bulk = shift(-shiftx_px, -shifty_px)        # shift is opposed wrt source pos
                eff_area = px_area * shifted_bulk.sum()             # effective detector area seen by the source [cm^2]
                return eff_area
        
            def _get_theta_errs():
                def propagation(s, ds):
                    return 1 / (1 + np.square(s/l)) * np.sqrt(np.square(ds / l) + np.square(s * dl / np.square(l)))
                return propagation(shifty, dshifty), propagation(shiftx, dshiftx)
            
            def _get_coord_errs(sdl):
                up_ra, up_dec = shift2equatorial(sdl, wfm, shiftx + dshiftx, shifty + dshifty)
                down_ra, down_dec = shift2equatorial(sdl, wfm, shiftx - dshiftx, shifty - dshifty)
                return np.abs(up_ra - down_ra)/2, np.abs(up_dec - down_dec)/2
            
            def _get_simulphs(sdl: object, sigma: float = 5.0) -> None:
                """
                Sums simulated photons contained in the SDLs taking into
                account the IROS reconstructed position and the error box.
                """
                up_ra, up_dec = ra + sigma*dra, dec + sigma*ddec
                down_ra, down_dec = ra - sigma*dra, dec - sigma*ddec
                phs = sdl.data[(sdl.data["RA"] > down_ra) & (sdl.data["RA"] < up_ra)]
                phs =  phs[(phs["DEC"] > down_dec) & (phs["DEC"] < up_dec)]
                return len(phs)
        
            y, x = bm.shift2pos(wfm, shiftx, shifty)                        # pos in px (from optimized shifts)
            thetay, thetax = np.arctan(shifty / l), np.arctan(shiftx / l)   # pos in angles wrt axis [rad]
            dthetay, dthetax = _get_theta_errs()                            # theta errs [rad]
            ra, dec = shift2equatorial(sdls[idx], wfm, shiftx, shifty)      # RA, DEC [rad]
            dra, ddec = _get_coord_errs(sdls[idx])                          # RA, DEC errs [rad]
            dcounts = np.sqrt(counts)                                       # fluence err (Poissonian) [ph]
            rate = counts / exposure[idx]                                   # rate [ph/s]
            drate = dcounts / exposure[idx]                                 # rate err [ph/s]
            flux = rate / _get_eff_area()                                   # flux [ph/cm^2/s]
            dflux = dcounts / exposure[idx]                                 # flux err [ph/cm^2/s]
            chi = -100                                                      # fit goodness
            simulphotons = _get_simulphs(sdls[idx])                         # simulated photons [ph]


            params = [
                y, x, shiftx, dshiftx, shifty, dshifty, thetax, dthetax, thetay, dthetay,
                ra, dra, dec, ddec, counts, dcounts, rate, drate, flux, dflux, obs_counts,
                np.sqrt(obs_counts), sub_counts, simulphotons, snr, chi,
            ]

            return params

        for sx, sy, f, snr, oc, sc, _ in zip(*iros_output[camera].values()):
            params = _get_params(sx, sy, f, snr, oc, sc)
            log.update(camera, params)

    def data_to_array(camera: str) -> dict:
        """Converts the lists in the log to arrays."""
        for key in iros_data[camera].keys():
            iros_data[camera][key]["data"] = np.asarray(iros_data[camera][key]["data"])

    # mask physical params
    sdls = [sdlA, sdlB]
    iros_data = log.initialize()

    ups = np.prod((wfm.upscale_f.x, wfm.upscale_f.y))          # px area [cm^2] TODO: units with astropy
    px_area = (
        1e-2 * wfm.specs["mask_deltax"] * wfm.specs["mask_deltay"] / ups
    )
    exposure = [data.header["EXPOSURE"] for data in sdls]      # camera exposure [s]
    l, dl = wfm.specs["mask_detector_distance"], 0.1           # mask-detector distance, error [mm]
    dshiftx = np.abs(wfm.bins_sky.x[0] - wfm.bins_sky.x[1])/2  # shift error along x [mm]
    dshifty = np.abs(wfm.bins_sky.y[0] - wfm.bins_sky.y[1])/2  # shift error along y [mm]

    for idx, cam in enumerate(iros_output.keys()):
        update_log(cam)
        data_to_array(cam)

    return iros_data


def compare_w_catalog(
    data: dict,
    catalogA: str | Path,
    catalogB: str | Path,
    cameras: tuple[str, str],
    min_flux: float = 0.0,
) -> dict:
    """
    Compares the reconstructed IROS sources data with the catalog
    containing the simulated sources.

    Args:
        - data: dict
        IROS data output from `compute_params()`.
        - catalogA: str | Path
        Path to the catalog for camera A.
        - catalogB: str | Path
        Path to the catalog for camera B.
        - cameras: tuple(str)
        Camera A and B of the WFM.
        - min_flux: float, default = 0.0
        Threshold for the minimum flux to be extracted from the
        given catalogs (to associate the IROS sources).

    Returns:
        - data: dict
        Updated input data with source name matched from the
        catalog and source flux recovered from the catalog.

    Raises:
        - FileNotFoundError: if FITS files do not exist.
        - ValueError: if files not in valid FITS format.
    """

    def get_catalogs() -> tuple[np.recarray]:
        """Returns the catalogs data."""

        def check_fits(pattern: Path) -> bool:
            """Check presence and validity of the FITS file."""
            if not pattern.is_file():
                raise FileNotFoundError("FITS file does not exists.")
            elif not _validate_fits(pattern):
                raise ValueError("File not in valid FITS format.")
            return True

        if check_fits(catalogA) and check_fits(catalogB):
            catA = fits.getdata(catalogA)
            catB = fits.getdata(catalogB)
            return catA, catB
    
    def camera_comparison(catalogs: list) -> None:
        """
        Compares respective catalogs for the two cameras and
        updates the input data dictionary.
        """

        def optimized_pos(
            catalog: np.recarray,
            pos: tuple[float, float],
        ) -> int:
            """Source association from catalog."""
            arg = np.argmin(np.square(catalog["RA"] - pos[0]) + np.square(catalog["DEC"] - pos[1]))
            return arg
        
        def source_in_db(camera: str, name: str) -> bool:
            """Checks database to avoid sources repetition."""
            if name in data[camera]["catalog_name"]["data"]:
                return True
            return False

        def remove_source(arg: int) -> None:
            """Removes repeting source in database."""
            # TODO: implement source removal by SNR comparison
            cam = data[camera]
            for key in list(cam.keys())[:-2]:
                cam[key]["data"] = np.delete(cam[key]["data"], arg)

        fake_sources = ["gctr_diffuse"]
        for catalog, camera in zip(catalogs, cameras):
            catalog = catalog[catalog["FLUX"] > min_flux]
            data[camera]["catalog_name"] = {"data": [], "format": "20A", "unit": ""}
            data[camera]["catalog_flux"] = {"data": [], "format": "D", "unit": "ph/cm2/s"}

            for ra, dec in zip(data[camera]["ra"]["data"], data[camera]["dec"]["data"]):
                argsource = optimized_pos(catalog, (ra, dec))
                source_id = catalog["NAME"][argsource] 
                if not source_in_db(camera, source_id) and (source_id not in fake_sources):
                    data[camera]["catalog_name"]["data"].append(source_id)
                    data[camera]["catalog_flux"]["data"].append(catalog["FLUX"][argsource])
                else:
                    arg = len(data[camera]["catalog_name"]["data"])
                    remove_source(arg)
    
    for c in [catalogA, catalogB]:
        if not isinstance(c, Path):
            c = Path(c)

    print("## Comparing with Catalogs...")
    catA, catB = get_catalogs()
    camera_comparison([catA, catB])
    print("## Successful comparison!")

    return data






























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
    sdl: object,
    wfm: object,
    save_to: str | Path,
) -> None:
    """Saves sky array to FITS Image."""

    sky = np.int16(sky)
    snr = np.float32(snr)

    def fit_WCS() -> WCS:
        """Fit the WCS."""
        n, m = wfm.sky_shape
        pxs = [
            (n - 1, 0), (n - 1, m - 1), (0, m - 1),
            (0, 0), (-n//4, m//4), (-n//4, -m//4),
            (n//4, -m//4), (n//4, m//4), (n//2, m//2),
        ]
        coords = [pos2equatorial(sdl, wfm, *pos) for pos in pxs]
        
        coord_pxs = tuple(np.array([px[idx] for px in pxs]) for idx in (1, 0))
        coord_radec = SkyCoord(
            ra=np.array([c.ra for c in coords]),
            dec=np.array([c.dec for c in coords]),
            frame="icrs", unit="deg",
        )
        wcs = fit_wcs_from_points(
            xy=coord_pxs, world_coords=coord_radec,
            projection="TAN", sip_degree=0,
        )
        return wcs

    wcs = fit_WCS()
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
        image_hdu.header.update(wcs.to_header())
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