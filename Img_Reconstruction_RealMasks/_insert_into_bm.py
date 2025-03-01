from pathlib import Path
import numpy as np
from astropy.io import fits
from mbloodmoon.io import _validate_fits
import mbloodmoon as bm


def compare_w_catalog(
    simul_data: str | Path,
    data: dict,
    camA: str = "cam1a",
    camB: str = "cam1b",
    min_flux: float = 0.0,
) -> tuple[dict, dict]:
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

        def optimized_pos(
            catalog: dict,
            pos: tuple[float, float],
        ) -> int:
            """Source association from catalog."""
            arg = np.argmin(np.square(catalog["RA"] - pos[0]) + np.square(catalog["DEC"] - pos[1]))
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
        double_cam = {"source": [], **{f"{key}_{cam}": [] for cam in [camA, camB] for key in ["ra", "dec", "flux"]}}

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
