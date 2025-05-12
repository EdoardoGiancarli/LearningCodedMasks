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

from _IROS_support import initialize_pipeline
from _IROS_config import run_pipeline


if __name__ == "__main__":

    """
    PIPELINE SET-UP.
    """
    print("\n#### INITIALIZING IROS PIPELINE")

    # mask
    mask_FITS = "wfm_mask.fits"
    IDEAL_MASK = False                 # infinitely opaque and thin mask

    # data
    skyfield = "Crab"
    # data_FITS = "20241011_galctr_rxte_sax_2-30keV_1ks_2cams_sources_cxb"
    data_FITS = "20250227_crab_cxb_2-50keV_1ks"

    cam_a, cam_b = "cam1a", "cam1b"
    dataset = "reconstructed"

    # upscaling
    UPSX_0, UPSY_0 = 5, 1              # initial upscaling (with which IROS is performed)
    UPSX_FINAL, UPSY_FINAL = 5, 1      # final upscaling for skies and visualisation

    # test ID and IROS set-up
    TEST_ID = "model_psfy_params"

    max_iterations = 1
    snr_threshold = 5

    sky_compositions = False           # if True, the WFM cameras will be joined to get the composed sky



    """
    RUNNING PIPELINE.
    """
    print("\n#### RUNNING PIPELINE..")
    # initialize pipeline parameters
    params = initialize_pipeline(
        mask=mask_FITS,
        ideal_mask=IDEAL_MASK,
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
    )

    # run pipeline
    run_pipeline(params)


# end