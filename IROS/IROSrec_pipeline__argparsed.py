"""
Main module for the IROS pipeline.
"""

import argparse
from bloodmoon.types import CoordEquatorial
from IROSrec.run import run_pipeline


def main() -> None:

    descr = r"""
                        
                         ____    _                   _           ____      _      __  __ 
                        / ___|  (_)  _ __     __ _  | |   ___   / ___|    / \    |  \/  |
                        \___ \  | | | '_ \   / _` | | |  / _ \ | |       / _ \   | |\/| |
                        ___) |  | | | | | | | (_| | | | |  __/ | |___   / ___ \  | |  | |
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

    TODO:
        - fix skies upscaling for output visualisation
        - insert possibility to load residuals of proper shapes to act as BKG for output IROS skies (not oversampled)
        - generalize directory paths for all users
    """
    parser = argparse.ArgumentParser(
        description=descr,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # pipeline args set-up
    # - observation data
    parser.add_argument(
        "--run",
        type=str,
        help="Run ID and name of the directory with IROS output files.",
    )
    parser.add_argument(
        "--skyfield",
        type=str,
        help="Sky-field selection.",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        help="Directory name with FITS output files from WISEMAN.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="reconstructed",
        help="Determines which WISEMAN photon list to use (default: %(default)s).",
    )
    # - cameras mask and arrays binning structure sampling
    parser.add_argument(
        "--up_fine",
        type=int,
        default=2,
        help="Upsampling factor along the fine camera axis (default: %(default)s).",
    )
    parser.add_argument(
        "--up_coarse",
        type=int,
        default=1,
        help="Upsampling factor along the coarse camera axis (default: %(default)s).",
    )
    # - IROS setup args
    parser.add_argument(
        "--max_iters",
        type=int,
        default=25,
        help="Max IROS iterations (default: %(default)s).",
    )
    parser.add_argument(
        "--snr_thresh",
        type=float,
        default=5.0,
        help="SNR threshold for validating source candidates during IROS (default: %(default)s).",
    )
    # - data filters setup args
    parser.add_argument(
        "--energy_range",
        nargs=2,
        type=float,
        default=None,
        metavar=('MIN', 'MAX'),
        help="Photons energy filter range, in [keV] (e.g., --energy_range 2.0 50.0, default: %(default)s).",
    )
    parser.add_argument(
        "--photons_coords",
        nargs='*',
        type=float,
        default=None,
        help="RA/Dec coords filter (sources to filter out), in [deg]. Pass as pairs: RA1 DEC1 RA2 DEC2 ... (default: %(default)s)",
    )
    parser.add_argument(
        "--flux_range",
        nargs=2,
        type=float,
        default=None,
        metavar=('MIN', 'MAX'),
        help="Sources flux filter for the catalogue comparison, in [Crab] (e.g., --flux_range 0.1 10.0, default: %(default)s).",
    )
    # - LEM-X cameras coded-mask pattern
    parser.add_argument(
        "--mask_pattern",
        type=str,
        default="mask_NTHT_20260129_CORRECTED.fits",
        help="FITS file with mask pattern, reconstruction array and detector-plane sensitivity array (default: %(default)s).",
    )
    parser.add_argument(
        "--thin_mask",
        action="store_true",
        help="Flag for ideal thin coded-mask plate. If 'True', removes vignetting effects in sources shadowgram model (default: %(default)s).",
    )
    # - LEM-X Unit single-camera IDs
    parser.add_argument(
        "--IDcam_A",
        type=str,
        default="cam1a",
        help="LEM-X Unit coded-mask camera 'A' (default: %(default)s).",
    )
    parser.add_argument(
        "--IDcam_B",
        type=str,
        default="cam1b",
        help="LEM-X Unit coded-mask camera 'B' (default: %(default)s).",
    )
    parser.add_argument(
        "--compose_unit",
        action="store_true",
        help="Flag for LEM-X Unit sky-fields composition (default: %(default)s).",
    )
    # - SDDs smoothing setup args
    parser.add_argument(
        "--smoothing",
        action="store_true",
        help="Selects if background smoothing is to be applied to the detector image.",
    )
    parser.add_argument(
        "--smoothing_snr_thresh",
        type=float,
        default=10.0,
        help=(
            "SNR threshold for brightest sources in sky-field after first IROS reconstruction."
            "They are removed to apply the smoothing (min=5.0, default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--baseline_irosrec",
        type=str,
        default=None,
        help="Path to non-smoothed IROS reconstruction directory, if present (written as '../baseline/', default: %(default)s).",
    )
    
    # args config
    args = parser.parse_args()

    # - convert input photon coords to CoordEquatorial objects, if any
    parsed_coords = None
    if args.photons_coords:
        if len(args.photons_coords) % 2 != 0:
            parser.error("Invalid input coords. Must provide RA/DEC pairs for --photons_coords arg.")
        parsed_coords = [
            CoordEquatorial(args.photons_coords[i], args.photons_coords[i+1]) 
            for i in range(0, len(args.photons_coords), 2)
        ]
    # - convert ranges to tuples
    energy_range = tuple(args.energy_range) if args.energy_range else None
    flux_range = tuple(args.flux_range) if args.flux_range else None
    
    print("\n#### RUNNING PIPELINE..")
    run_pipeline(
        mask=args.mask_pattern,
        thin_mask=args.thin_mask,
        skyfield=args.skyfield,
        skydata=args.datadir,
        unit_camsID=(args.IDcam_A, args.IDcam_B),
        dataset=args.dataset_type,
        start_ups=(args.up_fine, args.up_coarse),
        final_ups=(args.up_fine, args.up_coarse),
        analysisID=args.run,
        iros_max_iterations=args.max_iters,
        iros_snr_threshold=args.snr_thresh,
        sky_compositions=args.compose_unit,
        smoothing=args.smoothing,
        smoothing_thresh=args.smoothing_snr_thresh,
        smoothing_baseline_recnstr=args.baseline_irosrec,
        energy_range=energy_range,
        coords=parsed_coords,
        flux_range=flux_range,
    )
    return


if __name__ == "__main__":
    main()


# end