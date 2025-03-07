from pathlib import Path
from mbloodmoon.io import _validate_fits
from astropy.io import fits
import numpy as np
from mbloodmoon.images import upscale, compose
from mbloodmoon.iros_management.iros_wrappers import save_sky, load_sky
import mbloodmoon as bm

def load_output(
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
        def get_sky(
            hdu_data: fits.FITS_rec,
            sky_shape: tuple,
        ) -> np.array:
            values = hdu_data.field(0)
            y, x = hdu_data.field(1), hdu_data.field(2)
            sky = np.zeros(sky_shape); sky[y, x] = values
            return sky
        
        with fits.open(filepath) as hdul:
            hdus = [dict(hdul[1].header), dict(hdul[2].header)]
            hdus_data = [hdul[1].data, hdul[2].data]
            hdus_sky = [dict(hdul[3].header), dict(hdul[4].header)]
            hdus_data_sky = [hdul[3].data, hdul[4].data]

        data = {
            hdu["EXTNAME"].lower(): {
                hdu["TTYPE" + str(idx + 1)].lower(): hdu_data.field(idx)
                for idx in range(len(hdus_data[0][0]))
            }
            for hdu, hdu_data in zip(hdus, hdus_data)
        }

        for camera, skyhdu, skyhdu_data in zip(data.keys(), hdus_sky, hdus_data_sky):
            data[camera]["sky_reidues"] = get_sky(skyhdu_data, (skyhdu["ROWS"], skyhdu["COLS"]))
        
        return data

    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if check_fits(filepath):
        print("# Loading data...")
        data = load_data(filepath)
        print("# Loading completed!")
        return data




if __name__ == "__main__":

    root_path = "/mnt/d/PhD_AASS/Coding/Images_fits/"
    mask_file = root_path + "wfm_mask.fits"
    simul_data = root_path + "iros_simulation_GC_LMC/20241011_galctr_rxte_sax_2-30keV_1ks_2cams_sources_cxb/"

    cam_a = "cam1a"
    cam_b = "cam1b"
    dataset = "reconstructed"

    filepaths = bm.simulation_files(simul_data)
    wfm = bm.codedmask(mask_file, upscale_x=5, upscale_y=1)
    sdlA = bm.simulation(filepaths[cam_a][dataset])
    sdlB = bm.simulation(filepaths[cam_b][dataset])

    max_iterations = 25
    snr_threshold = 5

    n_test = 0


    iros_output_temp = load_output(root_path + f"iros_output{n_test}.fits")

    skyA = iros_output_temp[cam_a]["sky_reidues"]
    skyB = iros_output_temp[cam_b]["sky_reidues"]


    detectors = tuple(bm.count(wfm, sdl.data)[0] for sdl in [sdlA, sdlB])
    variances = tuple(bm.variance(wfm, d) for d in detectors)


    save_sky(
        sky=upscale(skyA, upscale_y=8),
        snr=upscale(bm.snratio(skyA, variances[0]), upscale_y=8),
        sdl=sdlA,
        wfm=wfm,
        save_to=root_path + f"skyfits_test_{cam_a.upper()}_{0}.fits",
    )

    save_sky(
        sky=upscale(skyB, upscale_y=8),
        snr=upscale(bm.snratio(skyB, variances[1]), upscale_y=8),
        sdl=sdlB,
        wfm=wfm,
        save_to=root_path + f"skyfits_test_{cam_b.upper()}_{0}.fits",
    )


# end