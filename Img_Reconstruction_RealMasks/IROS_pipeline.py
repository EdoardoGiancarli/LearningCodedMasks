"""
Pipeline for running IROS, with checkpoints.

Layout:
    1. IROS params set up
    2. Saving simulated skies and composition (with significance)
    3. Run IROS, saving output and residues (with significance + composition)
    4. Computing and saving candidates parameters for post-IROS analysis
    5. Candidates comparison with given catalogs, saving updated database
    6. Generating and saving output skies (IROS sources + IROS residues, with significance + composition)

Notes:
    - Variables are re-assigned to reduce memory cost
    - For each step there is a checkpoint, so that if saved FITS are already in the base directory, those
      step are ignored (to handle compiler crashes).
      To repeat the step, just delete the output files of that step.

Dependencies for running the pipeline:
    - Change paths for data in `_handle_dirpaths()`

TODO:
    - insert possibility to load residuals of proper shapes to act as BKG for output IROS skies (not oversampled)
"""

from pathlib import Path
import numpy as np

import mbloodmoon.iros_management as iros

from mbloodmoon.io import simulation_files, simulation
from mbloodmoon.mask import decode, count, variance, snratio #, codedmask
# from mbloodmoon.images import upscale #, downscale
from mbloodmoon.utils import timer

from temp_camera import codedmask


def _handle_dirpaths(
    mask: str,
    skyfield: str,
    simul: str,
) -> tuple[str]:
    """Handles paths depending on the OS."""

    if Path(base_path := "/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data/").is_dir():
        if skyfield is None or simul is None:
            raise ValueError("When using Debian, 'skyfield' and 'simul' must exist.")
        mask_path = base_path + "Simulations/" + mask                                  # dirpath to WFM mask file 
        data_path = base_path + "Simulations/" + skyfield + "/" + simul + "/"          # dirpath with simul files
        save_path = base_path + "Outputs/" + "Out" + skyfield + "/" + simul + "/"      # dirpath to save output data
        
    elif Path(base_path := "/mnt/d/PhD_AASS/Coding/Images_fits/").is_dir():
        mask_path = base_path + mask
        data_path = base_path + skyfield + "/" + simul + "/"
        save_path = base_path

    else:
        raise ValueError("A0, ma ndo sei finit*?")
    
    if not Path(mask_path).is_file():
        raise ValueError(f"WFM mask '{mask}' does not exist.")
    for name, dirpath in zip(
            ("data_path", "save_path"),
            (data_path, save_path),
        ):
            if not Path(dirpath).is_dir():
                raise ValueError(f"{name} '{dirpath}' does not exist.")

    return mask_path, data_path, save_path


def _handle_simul_correction(
    ideal_mask: bool,
    dataset: str,
) -> tuple[bool, bool]:
    """Handles vignetting and psf correction along y for IROS."""

    if dataset not in ["detected", "reconstructed"]:
        raise ValueError("dataset must be either 'detected' or 'reconstructed'.")
    
    vignetting = False if ideal_mask else True
    psfy = False if dataset == "detected" else True
    return vignetting, psfy


def iros_pipeline_report(
    skyfield: tuple[str],
    dataset: str,
    mask_type: bool,
    start_upscaling: tuple[int],
    final_upscaling: tuple[int],
    iros_iterations: int,
    sky_composition: bool,
) -> None:
    """Prints out some IROS pipeline info."""
    print(
        f"Testing skyfield: '{skyfield[0]}', simulation of: {skyfield[1][:4]}/{skyfield[1][4:6]}/{skyfield[1][6:8]}\n"
        f"Dataset type: '{dataset}'\n"
        f"Mask type: '{"ideal" if mask_type else "realistic"}'\n"
        f"Starting upscaling: {start_upscaling}\n"
        f"Final upscaling: {final_upscaling}\n"
        f"Max IROS iteration: {iros_iterations}\n"
        f"Sky compositions: {sky_composition}\n"
    )




if __name__ == "__main__":

    """
    #### INITIALIZE PIPELINE.
    """
    print("\n#### Initializing...")
    mask_FITS = "wfm_mask.fits"
    IDEAL_MASK = False                 # infinitely opaque and thin mask

    N_TEST = "_" + "no_upscaling_and_shifts_offset4"
    UPSX_0, UPSY_0 = 5, 2              # initial upscaling (with which IROS is performed)
    UPSX_FINAL, UPSY_FINAL = 5, 2      # final upscaling for skies and visualisation

    skyfield = "GalacticCenter"
    data_FITS = "20241011_galctr_rxte_sax_2-30keV_1ks_2cams_sources_cxb"
    
    mask_file, simul_data, save_path = _handle_dirpaths(
        mask=mask_FITS,
        skyfield=skyfield,
        simul=data_FITS,
    )

    cam_a = "cam1a"
    cam_b = "cam1b"
    dataset = "reconstructed"

    max_iterations = 15
    snr_threshold = 5

    sky_compositions = False           # if True, the WFM cameras will be joint to get the composed sky

    iros_pipeline_report(
        skyfield=(skyfield, data_FITS),
        dataset=dataset,
        mask_type=IDEAL_MASK,
        start_upscaling=(UPSX_0, UPSY_0),
        final_upscaling=(UPSX_FINAL, UPSY_FINAL),
        iros_iterations=max_iterations,
        sky_composition=sky_compositions,
    )


    with timer("##### IROS PIPELINE #####"):
        """
        #### IROS SETUP.
        """
        print("\n#### IROS Setup...")

        with timer("IROS Setup"):
            vignetting, psfy = _handle_simul_correction(IDEAL_MASK, dataset)
            wfm = codedmask(mask_file, upscale_x=UPSX_0, upscale_y=UPSY_0)

            filepaths = simulation_files(simul_data)
            sdlA = simulation(filepaths[cam_a][dataset])
            sdlB = simulation(filepaths[cam_b][dataset])

            sdls = (sdlA, sdlB)
            with timer("Compute dets/vars"):
                detectors = tuple(count(wfm, sdl.data)[0] for sdl in sdls)
                variances = tuple(variance(wfm, d) for d in detectors)

            # WCS fit (here the camera is upscaled with the final upscaling)
            with timer("WCS fit"):
                wfm_WCS = codedmask(mask_file, upscale_x=UPSX_FINAL, upscale_y=UPSY_FINAL)
                wcs_fit = tuple(iros.fit_WCS(wfm_WCS, sdl) for sdl in sdls)



        """
        #### SAVING SIMULATED SKIES.
        """
        print("\n#### Saving simulated skies...")
        names = tuple(save_path + f"sky_SIMUL_{cam.upper()}_TEST{N_TEST}.fits" for cam in (cam_a, cam_b))
        comp_name = save_path + f"COMPOSED_sky_SIMUL_{cam_a.upper()}_{cam_b.upper()}_TEST{N_TEST}.fits"

        if not Path(names[0]).is_file() and not Path(names[1]).is_file():
            skies = tuple(decode(wfm, d) for d in detectors)
            snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skies, variances))

            # ups_skies = tuple(upscale(sky, upscale_y=UPSY_FINAL - UPSY_0 + 1) for sky in skies)
            # ups_snrs = tuple(upscale(snr, upscale_y=UPSY_FINAL - UPSY_0 + 1) for snr in snrs)

            for res, snr, sdl, name, wcs in zip(skies, snrs, sdls, names, wcs_fit):
                iros.save_sky(res, snr, sdl, name, wcs)
        else:
            print("# Simulated skies already saved!")

        if not Path(comp_name).is_file() and sky_compositions:
            with timer("Camera composition"):
                iros.camera_composition(
                    skyA_path=names[0],
                    skyB_path=names[1],
                    save_to=comp_name,
                )



        """
        #### RUN IROS AND SAVE OUTPUT + RESIDUES.
        """
        print("\n#### Running IROS...")
        iros_output_name = save_path + f"IROS_output_TEST{N_TEST}.fits"
        names = tuple(save_path + f"skyRES_IROS_{cam.upper()}_TEST{N_TEST}.fits" for cam in (cam_a, cam_b))
        comp_name = save_path + f"COMPOSED_skyRES_IROS_{cam_a.upper()}_{cam_b.upper()}_TEST{N_TEST}.fits"

        if not Path(iros_output_name).is_file() or not (Path(names[0]).is_file() and Path(names[1]).is_file()):
            iros_output, skies = iros.perform_iros(
                camerasID=(cam_a, cam_b),
                camera=wfm,
                sdl_camA=sdlA,
                sdl_camB=sdlB,
                max_iterations=max_iterations,
                snr_threshold=snr_threshold,
                vignetting=vignetting,
                psfy=psfy,
            )

            iros.save_iros_output(iros_output, mask_file, iros_output_name)

            snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skies, variances))

            # ups_skies = tuple(upscale(sky, upscale_y=UPSY_FINAL - UPSY_0 + 1) for sky in skies)
            # ups_snrs = tuple(upscale(snr, upscale_y=UPSY_FINAL - UPSY_0 + 1) for snr in snrs)

            for res, snr, sdl, name, wcs in zip(skies, snrs, sdls, names, wcs_fit):
                iros.save_sky(res, snr, sdl, name, wcs)

        else:
            print("# IROS data already saved!")
            iros_output = iros.load_iros_output(iros_output_name)


        if not Path(comp_name).is_file() and sky_compositions:
            with timer("Camera composition"):
                iros.camera_composition(
                    skyA_path=names[0],
                    skyB_path=names[1],
                    save_to=comp_name,
                )



        """
        #### COMPUTE SOURCES PARAMS WITH IROS OUTPUT.
        """
        print("\n#### Computing sources parameters...")
        iros_data_name = save_path + f"IROS_data_TEST{N_TEST}.fits"

        if not Path(iros_data_name).is_file():
            log = iros.gen_params_log((cam_a, cam_b))

            iros_data = iros.compute_params(
                iros_output=iros_output,
                camera=wfm,
                sdl_camA=sdlA,
                sdl_camB=sdlB,
                log=log,
            )

            # WARNING: the px position in this DB might not match with upscaled skies
            iros.save_iros_data(
                data=iros_data,
                mask_file=mask_file,
                sdls=(sdlA, sdlB),
                save_to=iros_data_name,
            )

        else:
            iros_data = iros.load_iros_data(iros_data_name)



        """
        #### CATALOG COMPARISON AND DATABASE UPDATE.
        """
        print("\n#### Performing catalog comparison...")
        DB_name = save_path + f"IROS_sources_database_TEST{N_TEST}.fits"
        # WARNING: source assignment relies only on catalog sources
        if not Path(DB_name).is_file():
            database = iros.compare_w_catalog(
                data=iros_data,
                catalogA=filepaths[cam_a]["sources"],
                catalogB=filepaths[cam_b]["sources"],
                camerasID=(cam_a, cam_b),
                min_flux=1e-1,
            )

            iros.save_iros_data(
                data=database,
                mask_file=mask_file,
                sdls=(sdlA, sdlB),
                save_to=DB_name,
            )

        else:
            database = iros.load_iros_data(DB_name)



        """
        #### GENERATING SKIES FROM IROS OUTPUT + RESIDUES.
        """
        print("\n#### Generating and saving IROS output skies...")
        names = tuple(save_path + f"OUTsky_IROS_{cam.upper()}_TEST{N_TEST}.fits" for cam in (cam_a, cam_b))
        comp_name = save_path + f"COMPOSED_OUTsky_IROS_{cam_a.upper()}_{cam_b.upper()}_TEST{N_TEST}.fits"

        if not Path(names[0]).is_file() and not Path(names[1]).is_file():
            with timer("IROS output skies"):
                if "skies" not in globals():
                    _names = tuple(f"skyRES_IROS_{cam.upper()}_TEST{N_TEST}.fits" for cam in (cam_a, cam_b))
                    skies = tuple(iros.load_sky(save_path + n)[0] for n in _names)
                skies = tuple(iros.make_sky(database, camID, wfm, res) for camID, res in zip((cam_a, cam_b), skies))
                snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skies, variances))

            # ups_skies = tuple(upscale(sky, upscale_y=UPSY_FINAL - UPSY_0 + 1) for sky in skies)
            # ups_snrs = tuple(upscale(snr, upscale_y=UPSY_FINAL - UPSY_0 + 1) for snr in snrs)

            for res, snr, sdl, name, wcs in zip(skies, snrs, sdls, names, wcs_fit):
                iros.save_sky(res, snr, sdl, name, wcs)
        else:
            print("# IROS output skies already saved!")

        if not Path(comp_name).is_file() and sky_compositions:
            with timer("Camera composition"):
                iros.camera_composition(
                    skyA_path=names[0],
                    skyB_path=names[1],
                    save_to=comp_name,
                )


# end