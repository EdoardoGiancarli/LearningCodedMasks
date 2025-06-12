r"""
             ___   ____     ___    ____         ____    _                  _   _         
            |_ _| |  _ \   / _ \  / ___|       |  _ \  (_)  _ __     ___  | | (_)  _ __  
             | |  | |_) | | | | | \___ \       | |_) | | | | '_ \   / _ \ | | | | | '_ \   / _ \
             | |  |  _ <  | |_| |  ___) |      |  __/  | | | |_) | |  __/ | | | | | | | | |  __/
            |___| |_| \_\  \___/  |____/       |_|     |_| | .__/   \___| |_| |_| |_| |_|  \___|
                                                           |_|


Pipeline for running IROS, with checkpoints.

Layout:
    1. Pipeline params set up
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
    - Change paths for data in `_handle_dirpaths()` in `_IROS_support.py`

TODO:
    - fix skies upscaling for output visualisation
    - insert possibility to load residuals of proper shapes to act as BKG for output IROS skies (not oversampled)
    - generalize directory paths for all users
    - WARNING: source assignment relies only on catalog sources
"""
from collections.abc import Sequence

from mbloodmoon.types import CoordEquatorial

from _IROS_support import initialize_pipeline
from _IROS_config import run_pipeline


"""
PIPELINE SET-UP.
"""
print("\n#### INITIALIZING IROS PIPELINE")

## mask
mask_FITS: str = "wfm_mask.fits"
THIN_MASK: bool = False           # infinitely opaque and/or thin mask

## data
#skyfield: str = "GalacticCenter"
#data_FITS = "20250430_galctr_rxte_sax_2-50keV_1ks_realmask_infdet_sources_cxb"
#skyfield: str = "Crab"
#data_FITS = "20250227_crab_cxb_2-50keV_1ks"
skyfield: str = "IROSDummy"
data_FITS = "obliquedistr_infdet_2-50keV_1ks"

cam_a: str = "cam1a"
cam_b: str = "cam1b"
dataset: str = "reconstructed"

## upscaling
UPSX_0: int = 1                    # initial upscaling (with which IROS is performed) 
UPSY_0: int = 5

UPSX_FINAL: int = 1                # final upscaling for skies and visualisation
UPSY_FINAL: int = 5

## test ID
TEST_ID: str = "test_model_upx1y5_central_source"

## IROS set-up
max_iterations: int = 1
snr_threshold: int | float = 5

sky_compositions: bool = False           # if True, the WFM cameras will be joined to get the composed sky

## setup filters
# - photons energy filter - [keV]
photons_energy_range: int | tuple[int, int] | None = (2, 30)
# - RA/Dec filter (sources to filter out) - [deg]
photons_coords: CoordEquatorial | Sequence[CoordEquatorial] | None = (
    CoordEquatorial(ra=249.459, dec=-67.52572),
    CoordEquatorial(ra=255.6928, dec=-60.21748),
    CoordEquatorial(ra=260.2596, dec=-51.18224),
    CoordEquatorial(ra=263.7089, dec=-40.5835),
    CoordEquatorial(ra=268.5208, dec=-17.36688),
    CoordEquatorial(ra=270.254, dec=-6.622541),
    CoordEquatorial(ra=271.6881, dec=2.672805),
    CoordEquatorial(ra=272.8929, dec=10.39376),
)

# - number of sources in the catalog for comparison
n_sources: int | tuple[int, int] | None = None
# - sources flux filter for the catalog comparison - [ph/cm2/s]
sources_flux_range: int | float | tuple[int | float, int | float] | None = None



if __name__ == "__main__":

    """
    RUNNING PIPELINE.
    """
    print("\n#### RUNNING PIPELINE..")
    # initialize pipeline parameters
    params = initialize_pipeline(
        mask=mask_FITS,
        ideal_mask=THIN_MASK,
        skyfield=skyfield,
        skydata=data_FITS,
        wfm_cameras=(cam_a, cam_b),
        dataset_type=dataset,
        start_ups=(UPSX_0, UPSY_0),
        end_ups=(UPSX_FINAL, UPSY_FINAL),
        testID=TEST_ID,
        iros_max_iterations=max_iterations,
        iros_snr_threshold=snr_threshold,
        sky_compositions=sky_compositions,
        energy_range=photons_energy_range,
        coords=photons_coords,
        n=n_sources,
        flux_range=sources_flux_range,
    )

    # run pipeline
    run_pipeline(params)


# end