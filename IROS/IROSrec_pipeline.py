r"""
                      
                       ____    _                   _           ____      _      __  __ 
                      / ___|  (_)  _ __     __ _  | |   ___   / ___|    / \    |  \/  |
                      \___ \  | | | '_ \   / _` | | |  / _ \ | |       / _ \   | |\/| |
                       ___) | | | | | | | | (_| | | | |  __/ | |___   / ___ \  | |  | |
                      |____/  |_| |_| |_|  \__, | |_|  \___|  \____| /_/   \_\ |_|  |_|
                       ___   ____     ___  |___/_    ____  _            _ _            
                      |_ _| |  _ \   / _ \  / ___|  |  _ \(_)_ __   ___| (_)_ __   ___ 
                       | |  | |_) | | | | | \___ \  | |_) | | '_ \ / _ \ | | '_ \ / _ \
                       | |  |  _ <  | |_| |  ___) | |  __/| | |_) |  __/ | | | | |  __/
                      |___| |_| \_\  \___/  |____/  |_|   |_| .__/ \___|_|_|_| |_|\___|
                                                            |_|                        


Pipeline for running IROS, with checkpoints.

Layout:
    1. Pipeline params set up (in this module, through the variables below)
    2. IROS params set up
    3. Saving simulated skies and composition (with significance)
    4. Run IROS, saving output and residues (with significance + composition)
    5. Computing and saving candidates parameters for post-IROS analysis
    6. Candidates comparison with given catalogs, saving updated database
    7. Generating and saving output skies (IROS sources + IROS residues, with significance + composition)

Notes:
    - Variables are re-assigned to reduce memory cost
    - For each step there is a checkpoint, so that if saved FITS are already in the base directory, those
      steps are ignored (to handle compiler crashes).
      To repeat the step, just delete the output files of that step.

Dependencies for running the pipeline:
    - Change paths for data in `_handle_dirpaths()` in `_pipeline_support.py`

TODO:
    - fix skies upscaling for output visualisation
    - insert possibility to load residuals of proper shapes to act as BKG for output IROS skies (not oversampled)
    - generalize directory paths for all users
    - setup .yaml file to give it as input to this module
"""

import os
from typing import Sequence
from pathlib import Path

from bloodmoon.types import CoordEquatorial

from IROSrec.run import run_pipeline


"""
PIPELINE SET-UP.
"""
#### --- LEM-X CAMERAS MASK PATTERN
MASK_FITS: str = "mask_NTHT_20250725.fits"
THIN_MASK: bool = False                                                     # removes vignetting effects

#### --- OBSERVATION DATA
SKYFIELD: str = "IROSDummy"                                                      # skyfield selection
DATA_FITS: str = "iros_benchmark_2-50keV_mask_050_1040x17_1ks"             # directory with FITS files from WISEMAN

ID_CAMERA_A: str = "cam1a"
ID_CAMERA_B: str = "cam1b"
DATASET: str = "reconstructed"

#### --- IMAGES UPSCALING
UPSX_0: int = 2                     # initial upscaling (with which IROS is performed)
UPSY_0: int = 1

UPSX_FINAL: int = UPSX_0            # final upscaling for skies and visualisation
UPSY_FINAL: int = UPSY_0

#### --- ANALYSIS ID
ANALYSIS_ID: str = f"test_routine_2-6keV_smoothing"

#### --- IROS SETUP
MAX_ITERATIONS: int = 2
SNR_THRESHOLD: int | float = 3

MODULE_SKY_COMPOSITION: bool = True   # selects if the LEM-X module cameras are to be joined to get the composed sky

#### --- DETECTOR SMOOTHING SETUP
# - selects if detector smoothing is to be applied
SMOOTHING: bool = True
# - significance threshold for brightest sources in sky-field (min = 5.0)
SMOOTHING_SNR_THRESHOLD: int | float | None = 10
# - path to non-smoothed IROS reconstruction directory, if present (written as '../baseline/' <-- NOTE: the ending `/`)
BASELINE_IROSREC: str | Path | None = os.path.join(
    '/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7',
    f'Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data/Outputs/Out{SKYFIELD}',
    f'{DATA_FITS}/test_routine_2-6keV/',
    #'/mnt/d/PhD_AASS/Coding/Images_fits/singleCAM_iros_benchmark_detected/'
)

#### --- DATA FILTERS SETUP
# - photons energy filter - [keV]
PHOTONS_ENERGY_RANGE: tuple[int | float | None, int | float | None] = (2.0, 6.0)
# - RA/Dec filter (sources to filter out) - [deg]
PHOTONS_COORDS: CoordEquatorial | Sequence[CoordEquatorial] | None = None
# - sources flux filter for the catalog comparison - [Crab]
CATALOGUE_FLUX_RANGE: tuple[float | None, float | None] | None = None



if __name__ == "__main__":
    """
    RUNNING IROS RECONSTRUCTION PIPELINE.
    """
    print("\n#### RUNNING PIPELINE..")
    run_pipeline(
        mask=MASK_FITS,
        thin_mask=THIN_MASK,
        skyfield=SKYFIELD,
        skydata=DATA_FITS,
        unit_camsID=(ID_CAMERA_A, ID_CAMERA_B),
        dataset=DATASET,
        start_ups=(UPSX_0, UPSY_0),
        final_ups=(UPSX_FINAL, UPSY_FINAL),
        analysisID=ANALYSIS_ID,
        iros_max_iterations=MAX_ITERATIONS,
        iros_snr_threshold=SNR_THRESHOLD,
        sky_compositions=MODULE_SKY_COMPOSITION,
        smoothing=SMOOTHING,
        smoothing_thresh=SMOOTHING_SNR_THRESHOLD,
        smoothing_baseline_recnstr=BASELINE_IROSREC,
        energy_range=PHOTONS_ENERGY_RANGE,
        coords=PHOTONS_COORDS,
        flux_range=CATALOGUE_FLUX_RANGE,
    )


# end