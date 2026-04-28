"""
Module for basic IROS sky reconstruction run.
"""

from functools import partial
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from bloodmoon.io import simulation_files
from bloodmoon.mask import codedmask
from bloodmoon.mask import count
from bloodmoon.types import CoordEquatorial
import darksun as ds

from iros.optim import iros_singleCAM
from iros.procedure import run_IROS
from iros.procedure import config_instrument_effects
from iros.procedure import get_sources_database
from iros.handle import config_dirpaths
from iros.handle import config_filenames
from iros.handle import save_region_file


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
    fband: tuple[float | None, float | None],
) -> None:
    """
    Runs the IROS reconstruction.
    """
    def comp_fit_weights(obs: NDArray) -> NDArray:
        """Computes the weights for the loss metric in the optimisation procedure."""
        return 1.0 / np.sqrt(np.clip(obs, a_min=1.0, a_max=None))

    # config dirpaths
    mask_file, simul_data, save_path = config_dirpaths(
        mask=mask,
        skyfield=skyfield,
        simul=dataID,
        runID=analysisID,
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
    fmin, fmax = fband
    get_loop: Callable = partial(
        iros_singleCAM,
        camera=wfm,
        max_iterations=iros_iters,
        snr_threshold=snr_threshold,
        vignetting=VIGNETTING,
        psfy=PSFY,
        fit_weights=comp_fit_weights,
    )
    
    # - RUN IROS FOR CAMERA A
    print(f'\n#### - Analysing CAMERA {ID_CAM_A.upper()}')
    sdlA = ds.get_data(
        filepaths[ID_CAM_A][dataset], E_min=emin, E_max=emax, coords=coords,
    )
    catA = ds.get_catalogue(
        filepaths[ID_CAM_A]['sources'], F_min=fmin, F_max=fmax,
    )
    detector, _ = count(wfm, sdlA.DLdata)
    loop = get_loop(detector)
    outlog_camA, _ = run_IROS(wfm, loop, ID_CAM_A)

    # - RUN IROS FOR CAMERA B
    print(f'\n#### - Analysing CAMERA {ID_CAM_B.upper()}')
    sdlB = ds.get_data(
        filepaths[ID_CAM_B][dataset], E_min=emin, E_max=emax, coords=coords,
    )
    catB = ds.get_catalogue(
        filepaths[ID_CAM_B]['sources'], F_min=fmin, F_max=fmax,
    )
    detector, _ = count(wfm, sdlB.DLdata)
    loop = get_loop(detector)
    outlog_camB, _ = run_IROS(wfm, loop, ID_CAM_B)

    # compute sources databases
    log_camA = get_sources_database(wfm, sdlA, catA, outlog_camA, VIGNETTING)
    log_camB = get_sources_database(wfm, sdlB, catB, outlog_camB, VIGNETTING)
    # save database and region files
    save_region_file(log_camA, catA, filenames['OUT_REG'][0])
    save_region_file(log_camB, catB, filenames['OUT_REG'][1])
    ds.save_database(
        log_camA=log_camA,
        log_camB=log_camB,
        sdlA=sdlA,
        sdlB=sdlB,
        save_to=filenames['SRCS_DB'],
    )
    
    return




if __name__ == '__main__':
    
    #### --- LEM-X CAMERAS MASK PATTERN
    MASK_FITS: str = "wfm_mask_NTHT_20250725.fits"
    THIN_MASK: bool = False                                            # removes vignetting effects

    #### --- OBSERVATION DATA
    SKYFIELD: str = "IROSDummy"                                             # skyfield selection
    DATA_FITS: str = "iros_benchmark_2-50keV_mask_050_1040x17_1ks"
    DATASET: str = "reconstructed"

    #### --- IMAGES UPSCALING
    UPSX_0: int = 2                     # initial upscaling (with which IROS is performed)
    UPSY_0: int = 1

    #### --- ANALYSIS ID
    ANALYSIS_ID: str = f"test_routine_isWorking"

    #### --- IROS SETUP
    MAX_ITERATIONS: int = 3
    SNR_THRESHOLD: int | float = 5

    #### --- DATA FILTERS SETUP
    # - photons energy filter - [keV]
    PHOTONS_ENERGY_RANGE: tuple[int | float | None, int | float | None] = (None, None)
    # - RA/Dec filter (sources to filter out) - [deg]
    PHOTONS_COORDS: CoordEquatorial | Sequence[CoordEquatorial] | None = None #CoordEquatorial(244.9797, -15.6401)
    # - sources flux filter for the catalog comparison - [Crab]
    CATALOGUE_FLUX_RANGE: tuple[float | None, float | None] = (None, None)
    
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
        fband=CATALOGUE_FLUX_RANGE,
    )


# end