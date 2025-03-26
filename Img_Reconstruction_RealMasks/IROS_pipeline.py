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
    - Change paths for data in `_choose_OS()`

TODO:
    - insert possibility to load residuals of proper shapes to act as BKG for output IROS skies (not oversampled)
"""

from pathlib import Path
import numpy as np

import mbloodmoon.iros_management as iros

from mbloodmoon.io import simulation_files, simulation
from mbloodmoon.mask import codedmask, decode, count, variance, snratio
from mbloodmoon.images import upscale, downscale


def _choose_OS() -> tuple[str]:
    """Handles paths depending on the OS."""
    if Path(base_path := "/media/egiancarli/Data/Edos_Magnificent_Manor/PhD_AASS/Coding/Data/").is_dir():  # base dirpath
        data_path = base_path + "Simulations/"                                                             # dirpath with simul files
        save_path = base_path + "Outputs/"                                                                 # dirpath to save output data
    elif Path(base_path := "/mnt/d/PhD_AASS/Coding/Images_fits/").is_dir():
        data_path = base_path
        save_path = base_path
    else:
        raise ValueError("A0, ma ndo sei finit*?")
    return data_path, save_path


## TODO [1:-1, :] because of problems with `upscale()` and wfm upscaling
#def _upscale(arr, upsy):
#    return upscale(arr, upscale_y=upsy)#[1:-1, :]


def handle_simulation(
    ideal_mask: bool,
    dataset: str,
) -> tuple[bool, bool]:
    """Handles vignetting and psf correction along y for IROS."""

    if dataset not in ["detected", "reconstructed"]:
        raise ValueError("dataset must be either 'detected' or  'reconstructed'.")
    
    psfy = False if dataset == "detected" else True
    vignetting = False if ideal_mask else True
    return vignetting, psfy




if __name__ == "__main__":

    """
    #### INITIALIZE PIPELINE.
    """
    data_path, save_path = _choose_OS()
    IDEAL_MASK = False                     # infinitely opaque and thin mask
    N_TEST = "6_NORMALMASK"
    UPSX_0, UPSY_FINAL = 3, 5



    """
    #### IROS SETUP.
    """
    print("#### IROS Setup...\n")
    mask_file = data_path + "wfm_mask.fits"                                                                     # WFM mask
    simul_data = data_path + "iros_simulation_GC_LMC/20241011_galctr_rxte_sax_2-30keV_1ks_2cams_sources_cxb/"   # Simulated photons

    cam_a = "cam1a"
    cam_b = "cam1b"
    dataset = "reconstructed"
    vignetting, psfy = handle_simulation(IDEAL_MASK, dataset)

    wfm = codedmask(mask_file, upscale_x=UPSX_0, upscale_y=1)     # for IROS the skies are upscaled only along the x-dim

    filepaths = simulation_files(simul_data)
    sdlA = simulation(filepaths[cam_a][dataset])
    sdlB = simulation(filepaths[cam_b][dataset])

    max_iterations = 20
    snr_threshold = 5

    sdls = (sdlA, sdlB)
    detectors = tuple(count(wfm, sdl.data)[0] for sdl in sdls)
    variances = tuple(variance(wfm, d) for d in detectors)

    wfm_WCS = codedmask(mask_file, upscale_x=UPSX_0, upscale_y=UPSY_FINAL)     # WCS fit (here the camera is upscaled with the final upscaling)
    wcs_fit = tuple(iros.fit_WCS(wfm_WCS, sdl) for sdl in sdls)



    """
    #### SAVING SIMULATED SKIES.
    """
    print("#### Saving simulated skies...\n")
    names = tuple(save_path + f"sky_SIMUL_{cam.upper()}_TEST{N_TEST}.fits" for cam in (cam_a, cam_b))
    comp_name = save_path + f"COMPOSED_sky_SIMUL_{cam_a.upper()}_{cam_b.upper()}_TEST{N_TEST}.fits"

    if not Path(names[0]).is_file() and not Path(names[1]).is_file():
        skies = tuple(decode(wfm, d) for d in detectors)
        snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skies, variances))

        ups_skies = tuple(_upscale(sky, upsy=UPSY_FINAL) for sky in skies)
        ups_snrs = tuple(_upscale(snr, upsy=UPSY_FINAL) for snr in snrs)

        for res, snr, sdl, name, wcs in zip(ups_skies, ups_snrs, sdls, names, wcs_fit):
            iros.save_sky(res, snr, sdl, name, wcs)

    if not Path(comp_name).is_file():
        iros.camera_composition(
            skyA_path=names[0],
            skyB_path=names[1],
            save_to=comp_name,
        )



    """
    #### RUN IROS AND SAVE OUTPUT + RESIDUES.
    """
    print("#### Running IROS...\n")
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

        ups_skies = tuple(_upscale(sky, upsy=UPSY_FINAL) for sky in skies)
        ups_snrs = tuple(_upscale(snr, upsy=UPSY_FINAL) for snr in snrs)

        for res, snr, sdl, name, wcs in zip(ups_skies, ups_snrs, sdls, names, wcs_fit):
            iros.save_sky(res, snr, sdl, name, wcs)

    else:
        iros_output = iros.load_iros_output(iros_output_name)


    if not Path(comp_name).is_file():
        iros.camera_composition(
            skyA_path=names[0],
            skyB_path=names[1],
            save_to=comp_name,
        )



    """
    #### COMPUTE SOURCES PARAMS WITH IROS OUTPUT.
    """
    print("#### Computing sources parameters...\n")
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
    print("#### Performing catalog comparison...\n")
    DB_name = save_path + "Images_tests/" + f"IROS_sources_database_TEST{N_TEST}.fits"
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
    print("#### Generating and saving IROS output skies...\n")
    names = tuple(save_path + f"OUTsky_IROS_{cam.upper()}_TEST{N_TEST}.fits" for cam in (cam_a, cam_b))
    comp_name = save_path + f"COMPOSED_OUTsky_IROS_{cam_a.upper()}_{cam_b.upper()}_TEST{N_TEST}.fits"

    if not Path(names[0]).is_file() and not Path(names[1]).is_file():
        skies = tuple(iros.make_sky(database, camID, wfm, res) for camID, res in zip((cam_a, cam_b), skies))
        snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skies, variances))

        ups_skies = tuple(_upscale(sky, upsy=UPSY_FINAL) for sky in skies)
        ups_snrs = tuple(_upscale(snr, upsy=UPSY_FINAL) for snr in snrs)

        for res, snr, sdl, name, wcs in zip(ups_skies, ups_snrs, sdls, names, wcs_fit):
            iros.save_sky(res, snr, sdl, name, wcs)

    if not Path(comp_name).is_file():
        iros.camera_composition(
            skyA_path=names[0],
            skyB_path=names[1],
            save_to=comp_name,
        )


# end