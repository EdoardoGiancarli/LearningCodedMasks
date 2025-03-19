"""
IROS output data management and computation.
"""

from typing import Literal
from pathlib import Path
from copy import deepcopy

import numpy as np
from tqdm import tqdm
from astropy.io import fits
from pandas import DataFrame

from mbloodmoon.io import SimulationDataLoader
from mbloodmoon.mask import CodedMaskCamera

from mbloodmoon.io import _validate_fits
from mbloodmoon.coords import shift2equatorial
from mbloodmoon.mask import count
from mbloodmoon.mask import decode
from mbloodmoon.mask import shift2pos
from mbloodmoon.images import _shift
from mbloodmoon.images import argmax
from mbloodmoon.optim import iros


def perform_iros(
    camerasID: tuple[str],
    camera: CodedMaskCamera,
    sdl_camA: SimulationDataLoader,
    sdl_camB: SimulationDataLoader,
    max_iterations: int = 25,
    snr_threshold: int | float = 5,
    vignetting: bool = True,
    psfy: bool = True,
) -> tuple[dict, tuple[np.array, np.array]]:
    """
    Runs the IROS (Iterative Removal of Sources) loop and stores the output.
    This function iteratively removes detected sources from the sky and updates 
    a log until either the maximum number of iterations is reached or the 
    SNR threshold is met.

    Args:
        camerasID (tuple[str]):
            Cameras of the WFM being processed.
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        sdl_camA (SimulationDataLoader):
            SimulationDataLoader instance for camera A.
        sdl_camB (SimulationDataLoader):
            SimulationDataLoader instance for camera B.
        max_iterations (int, optional (default=25)):
            Maximum number of iterations for the IROS loop.
        snr_threshold (int | float, optional (default=5)):
            Minimum SNR value required to continue the iterative source removal process.
        vignetting (bool, optional (default=`True`)):
            If `True`, the model used for optimization will simulate vignetting.
        psfy (bool, optional (default=`True`)):
            If `True`, the model used for optimization will simulate detector
            position reconstruction effects.

    Returns:
        output (tuple):
            - log_output (dict): Log with metadata and results from IROS.
            - residuals (tuple[np.array, np.array]): Sky residuals after IROS.
    """
    def init_log() -> dict:
        """Initializes the log dict structure."""
        init_keys = {
            "shiftx": [], "shifty": [], "fluence": [],
            "snr": [], "obs_counts": [], "sub_counts": [],
        }
        return {camera: deepcopy(init_keys) for camera in camerasID}
    
    def store_output(
        rec_source: tuple[tuple],
        obs_counts: tuple[float],
        sub_counts: tuple[float],
    ) -> None:
        """Stores sources info into log."""
        for idx, camera in enumerate(camerasID):
            params = [*rec_source[idx], obs_counts[idx], sub_counts[idx]]
            for key, p in zip(log_keys, params):
                log_output[camera][key].append(p)
    
    def data2array() -> dict:
        """Converts the log lists in arrays."""
        for camera in camerasID:
            for key in log_keys:
                if not isinstance(log_output[camera][key], np.ndarray):
                    log_output[camera][key] = np.asarray(log_output[camera][key])
    
    print("## Computing stuff...")
    log_output = init_log()
    log_keys = list(log_output[camerasID[0]].keys())
    detectors = tuple(count(camera, sdl.data)[0] for sdl in [sdl_camA, sdl_camB])
    skies = tuple(decode(camera, d) for d in detectors)
    skies_max = [tuple(np.max(sky) for sky in skies)]
    skies = [skies]

    loop = iros(
        camera=camera,
        sdl_cam1a=sdl_camA,
        sdl_cam1b=sdl_camB,
        max_iterations=max_iterations,
        snr_threshold=snr_threshold,
        vignetting=vignetting,
        psfy=psfy,
    )

    print("## Looping around the FOV...")
    for sources, residuals in tqdm(loop):
        skies.append(residuals)
        skies_max.append(tuple(np.max(r) for r in residuals))
        obs_counts = skies_max[0]
        sub_counts = tuple(s.max() - r[*argmax(s)] for s, r in zip(skies[0], skies[1]))
        skies.pop(0); skies_max.pop(0)
        store_output(sources, obs_counts, sub_counts)

    data2array()
    return log_output, residuals


class IrosParams:
    """IROS parameters log management."""
    def __init__(self, camerasID: tuple[str]):
        self.cams = camerasID
        self.log = self.make_log()
        self.keys = self.log[self.cams[0]].keys()

    def make_log(self) -> dict:
        """Creates the IROS parameters log structure."""
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
        return {camera: deepcopy(init_keys) for camera in self.cams}

    def initialize(self) -> dict:
        """Initializes the log."""
        return self.log

    def update(self, cameraID: str, params: list) -> None:
        """Updates the log."""
        for key, p in zip(self.keys, params):
            self.log[cameraID][key]["data"].append(p)


def gen_params_log(camerasID: tuple[str]) -> IrosParams:
    """
    Initializes an IrosParams instance to manage the
    logging of IROS parameters for the specified cameras.

    Args:
        camerasID (tuple[str]):
            Cameras of the WFM being processed.

    Returns:
        log (IrosParams):
            IrosParams instance containing the initialized log structure.
    """
    return IrosParams(camerasID)


def compute_params(
    iros_output: dict,
    camera: CodedMaskCamera,
    sdl_camA: SimulationDataLoader,
    sdl_camB: SimulationDataLoader,
    log: IrosParams,
) -> dict:
    """
    Computes useful parameters for IROS reconstructed sources.

    Args:
        iros_output (dict):
            IROS data output from `perform_iros()`.
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        sdl_camA (SimulationDataLoader):
            SimulationDataLoader instance for camera A.
        sdl_camB (SimulationDataLoader):
            SimulationDataLoader instance for camera B.
        log (IrosParams):
            IrosParams instance used to store computed parameters.

    Returns:
        iros_data (dict):
            Database with computed parameters for each source
            (deepcopy of the log in IrosParams).
    """

    def update_log(cameraID: str) -> None:
        """Update the parameters log."""

        def _get_params(
            shiftx: float,
            shifty: float,
            counts: float,
            snr: float,
            obs_counts: float,
            sub_counts: float,
        ) -> list:
            """Computes needed parameters for analyses and storing."""
            
            def _get_eff_area() -> float:
                """
                Computes the effective area seen by the source on the
                detector to compute the flux.
                """
                def shift(x: int, y: int) -> np.array:
                    """Shifts bulk according to source projection."""
                    I_bulk = np.zeros(camera.detector_shape)
                    I_bulk[camera.bulk > 0] = 1
                    return _shift(camera.bulk, (x, y)) * I_bulk

                scalingy, scalingx = tuple(
                    d / s for d, s in zip(camera.detector_shape, camera.sky_shape)
                )
                shiftx_px = int(shiftx * scalingx / camera.specs["mask_deltax"])
                shifty_px = int(shifty * scalingy / camera.specs["mask_deltay"])
                shifted_bulk = shift(-shiftx_px, -shifty_px)                      # shift is opposed wrt source pos
                eff_area = px_area * shifted_bulk.sum()                           # effective detector area seen by the source [cm^2]
                return eff_area
        
            def _get_theta_errs():
                """Computes angular sky coords errors."""
                def propagation(s, ds) -> tuple[float]:
                    return 1 / (1 + np.square(s/l)) * np.sqrt(np.square(ds / l) + np.square(s * dl / np.square(l)))
                return propagation(shifty, dshifty), propagation(shiftx, dshiftx)
            
            def _get_coord_errs(sdl: SimulationDataLoader) -> tuple[float]:
                """Computes RA/DEC source errors."""
                up_ra, up_dec = shift2equatorial(sdl, camera, shiftx + dshiftx, shifty + dshifty)
                down_ra, down_dec = shift2equatorial(sdl, camera, shiftx - dshiftx, shifty - dshifty)
                return np.abs(up_ra - down_ra)/2, np.abs(up_dec - down_dec)/2
            
            def _get_simulphs(
                    sdl: SimulationDataLoader,
                    sigma: float = 5.0,
                ) -> None:
                """
                Sums simulated photons contained in the SDLs, taking into
                account the IROS reconstructed RA/DEC and their error box.
                """
                up_ra, up_dec = ra + sigma*dra, dec + sigma*ddec
                down_ra, down_dec = ra - sigma*dra, dec - sigma*ddec
                phs = sdl.data[(sdl.data["RA"] > down_ra) & (sdl.data["RA"] < up_ra)]
                phs =  phs[(phs["DEC"] > down_dec) & (phs["DEC"] < up_dec)]
                return len(phs)
        
            y, x = shift2pos(camera, shiftx, shifty)                        # pos in px (from optimized shifts)
            thetay, thetax = np.arctan(shifty / l), np.arctan(shiftx / l)   # pos in angles wrt axis [rad]
            dthetay, dthetax = _get_theta_errs()                            # theta errs [rad]
            ra, dec = shift2equatorial(sdls[idx], camera, shiftx, shifty)   # RA, DEC [rad]
            dra, ddec = _get_coord_errs(sdls[idx])                          # RA, DEC errs [rad]
            dcounts = np.sqrt(counts)                                       # fluence err (Poissonian) [ph]
            rate = counts / exposure[idx]                                   # rate [ph/s]
            drate = dcounts / exposure[idx]                                 # rate err [ph/s]
            flux = rate / _get_eff_area()                                   # flux [ph/cm^2/s]
            dflux = dcounts / exposure[idx]                                 # flux err [ph/cm^2/s]
            chi = -100                                                      # fit goodness
            simulphotons = _get_simulphs(sdls[idx])                         # simulated photons [ph]

            params = [
                y, x, shiftx, dshiftx, shifty, dshifty, thetax, dthetax,
                thetay, dthetay, ra, dra, dec, ddec, counts, dcounts,
                rate, drate, flux, dflux, obs_counts, np.sqrt(obs_counts),
                sub_counts, simulphotons, snr, chi,
            ]
            return params

        for sx, sy, f, snr, oc, sc in zip(*iros_output[cameraID].values()):
            params = _get_params(sx, sy, f, snr, oc, sc)
            log.update(cameraID, params)

    def data_to_array(cameraID: str) -> None:
        """Converts the lists in the log to arrays."""
        for key in iros_data[cameraID].keys():
            iros_data[cameraID][key]["data"] = np.asarray(iros_data[cameraID][key]["data"])

    
    sdls = [sdl_camA, sdl_camB]
    iros_data = log.initialize()

    # mask and data params
    ups = np.prod((camera.upscale_f.x, camera.upscale_f.y))          # px area [cm^2] TODO: units with astropy
    px_area = (
        1e-2 * camera.specs["mask_deltax"] * camera.specs["mask_deltay"] / ups
    )
    exposure = [data.header["EXPOSURE"] for data in sdls]            # camera exposure [s]
    l, dl = camera.specs["mask_detector_distance"], 0.1              # mask-detector distance, error [mm]
    dshiftx = np.abs(camera.bins_sky.x[0] - camera.bins_sky.x[1])/2  # shift error along x [mm]
    dshifty = np.abs(camera.bins_sky.y[0] - camera.bins_sky.y[1])/2  # shift error along y [mm]

    for idx, cam in enumerate(iros_output.keys()):
        update_log(cam)
        data_to_array(cam)

    return deepcopy(iros_data)


def compare_w_catalog(
    data: dict,
    catalogA: str | Path,
    catalogB: str | Path,
    camerasID: tuple[str],
    min_flux: float = 0.0,
) -> dict:
    """
    Compares the reconstructed IROS sources data with the catalog
    containing the simulated sources.

    Args:
        data (dict):
            IROS data output from `compute_params()`.
        catalogA (str | Path):
            Path to the catalog for camera A.
        catalogB (str | Path):
            Path to the catalog for camera B.
        camerasID (tuple[str]):
            Cameras of the WFM being processed.
        min_flux (float, optional (default=0.0)):
            Threshold for the minimum flux to be extracted from the
            given catalogs (to associate the IROS sources).

    Returns:
        data (dict):
            Deepcopy of the updated input data with source name matched
            from the catalog and source flux recovered from the catalog.

    Raises:
        FileNotFoundError: if FITS files do not exist.
        ValueError: if files not in valid FITS format.
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
            pos: tuple[float],
        ) -> int:
            """Source association from catalog."""
            arg = np.argmin(np.square(catalog["RA"] - pos[0]) + np.square(catalog["DEC"] - pos[1]))
            return arg
        
        def check_source(catalog: np.recarray, arg: int) -> str:
            """Checks if the source is in the catalog."""
            # TODO: this nested method is not really of help here
            # This should become important in later simulations
            pass
        
        def source_in_db(cameraID: str, name: str) -> bool:
            """Checks database to avoid sources repetition."""
            if name in data[cameraID]["catalog_name"]["data"]:
                return True
            return False

        def remove_source(cameraID: str, arg: int) -> None:
            """Removes repeting source in database."""
            # TODO: implement source removal by SNR comparison
            cam = data[cameraID]
            for key in list(cam.keys())[:-2]:
                cam[key]["data"] = np.delete(cam[key]["data"], arg)

        fake_sources = ["gctr_diffuse"]
        for catalog, cameraID in zip(catalogs, camerasID):
            catalog = catalog[catalog["FLUX"] > min_flux]
            data[cameraID]["catalog_name"] = {"data": [], "format": "20A", "unit": ""}
            data[cameraID]["catalog_flux"] = {"data": [], "format": "D", "unit": "ph/cm2/s"}

            for ra, dec in zip(data[cameraID]["ra"]["data"], data[cameraID]["dec"]["data"]):
                argsource = optimized_pos(catalog, (ra, dec))
                sourceID = catalog["NAME"][argsource]
                if not source_in_db(cameraID, sourceID) and (sourceID not in fake_sources):
                    data[cameraID]["catalog_name"]["data"].append(sourceID)
                    data[cameraID]["catalog_flux"]["data"].append(catalog["FLUX"][argsource])
                else:
                    arg = len(data[cameraID]["catalog_name"]["data"])
                    remove_source(cameraID, arg)

    for c in [catalogA, catalogB]:
        if not isinstance(c, Path):
            c = Path(c)

    print("## Comparing with Catalogs...")
    catA, catB = get_catalogs()
    camera_comparison([catA, catB])
    print("## Successful comparison!")
    return deepcopy(data)


def dict2df(data: dict) -> DataFrame:
    """
    Converts input data in a Pandas dataframe.

    Args:
        data (dict): input data
    
    Returns:
        df (DataFrame): output dataframe
    """
    df = DataFrame({
        (cam, param): values
        for cam, cam_data in data.items() 
        for param, values in cam_data.items()
    })
    return df


# end