"""
Module for basic IROS sky reconstruction run.
"""

from typing import Sequence

from bloodmoon.io import simulation_files
from bloodmoon.mask import codedmask
from bloodmoon.types import CoordEquatorial
import darksun as ds

from iros.optim import iros_singleCAM
from iros.procedure import run_IROS
from iros.procedure import config_instrument_effects
from iros.procedure import get_sources_database
from iros.handle import config_dirpaths
from iros.handle import config_filenames


def main(
    mask: str,
    skyfield: str,
    dataID: str,
    analysisID: str,
    dataset: str,
    ups: tuple[int, int],
    iros_iters: int,
    snr_threshold: float,
    thin_mask: bool,
    eband: tuple[int | float | None, int | float | None],
    coords: CoordEquatorial | Sequence[CoordEquatorial] | None,
) -> None:
    """
    Runs the IROS reconstruction.
    """
    # config dirpaths
    mask_file, simul_data, save_path = config_dirpaths(
        mask=mask,
        skyfield=skyfield,
        simul=dataID,
        run_name=analysisID,
    )
    # config instrument
    ID_CAM_A, ID_CAM_B = "cam1a", "cam1b"
    VIGNETTING, PSFY = config_instrument_effects(thin_mask, dataset)
    # config filenames
    filenames = config_filenames(save_path, (ID_CAM_A, ID_CAM_B))
    # setup data
    wfm = codedmask(mask_file, *ups)
    filepaths = simulation_files(simul_data)
    emin, emax = eband
    # run IROS

    # - RUN IROS FOR CAMERA A
    sdlA = ds.get_data(
        filepaths[ID_CAM_A][dataset], E_min=emin, E_max=emax, coords=coords,
    )

    # - RUN IROS FOR CAMERA B
    sdlB = ds.get_data(
        filepaths[ID_CAM_B][dataset], E_min=emin, E_max=emax, coords=coords,
    )

    # save database
    ...
    
    return




if __name__ == '__main__':
    
    #### --- LEM-X CAMERAS MASK PATTERN
    MASK_FITS: str = "mask_NTHT_20260129_CORRECTED.fits"
    THIN_MASK: bool = False                                            # removes vignetting effects

    #### --- OBSERVATION DATA
    SKYFIELD: str = "Crab"                                             # skyfield selection
    DATA_FITS: str = "crab_30deg_1s"
    DATASET: str = "reconstructed"

    #### --- IMAGES UPSCALING
    UPSX_0: int = 2                     # initial upscaling (with which IROS is performed)
    UPSY_0: int = 1

    #### --- ANALYSIS ID
    ANALYSIS_ID: str = f"test_crabSens_{DATA_FITS}_{DATASET}_noAnodesMask"

    #### --- IROS SETUP
    MAX_ITERATIONS: int = 3
    SNR_THRESHOLD: int | float = 3

    #### --- DATA FILTERS SETUP
    # - photons energy filter - [keV]
    PHOTONS_ENERGY_RANGE: tuple[int | float | None, int | float | None] = (None, None)
    # - RA/Dec filter (sources to filter out) - [deg]
    PHOTONS_COORDS: CoordEquatorial | Sequence[CoordEquatorial] | None = None #CoordEquatorial(244.9797, -15.6401)
    
    #### --- RUN RECONSTRUCTION
    main(
        mask=MASK_FITS,
        skyfield=SKYFIELD,
        dataID=DATA_FITS,
        analysisID=ANALYSIS_ID,
        dataset=DATASET,
        ups=(UPSX_0, UPSY_0),
        iros_iters=MAX_ITERATIONS,
        snr_threshold=SNR_THRESHOLD,
        thin_mask=THIN_MASK,
        eband=PHOTONS_ENERGY_RANGE,
        coords=PHOTONS_COORDS,
    )


# end