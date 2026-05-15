"""
Configuration script for the IROS pipeline.
"""

from pathlib import Path

import numpy as np

from _IROS_support import PipelineParams
from _IROS_support import output_pipeline_files

from mbloodmoon.io import simulation_files, simulation
from mbloodmoon.mask import decode, count, variance, snratio, codedmask
from mbloodmoon.filtering import filter_catalog
# from mbloodmoon.images import upscale #, downscale
import mbloodmoon.iros_management as iros

from timing import timer

#from temp_camera import codedmask
#print("\n###___using temp_camera binning___###\n")


def run_pipeline(params: PipelineParams) -> None:
    """
    Runs the IROS pipeline.

    Args:
        params (PipelineParams):
            PipelineParams instance with the initialized parameters for the pipeline.
    """
    # upscaling setup
    UPSX_0, UPSY_0 = params.start_ups[0], params.start_ups[1]
    UPSX_FINAL, UPSY_FINAL = params.end_ups[0], params.end_ups[1]
    UPX_TO, UPY_TO = UPSX_FINAL - UPSX_0 + 1, UPSY_FINAL - UPSY_0 + 1

    # check on pipeline files
    output_pipeline_files(params)


    # start pipeline
    with timer("##### IROS PIPELINE #####"):

        # IROS SETUP
        print("\n#### IROS Setup...")
        with timer("IROS Setup"):
            cam_a, cam_b = params.wfm_cameras

            wfm = codedmask(
                mask_filepath=params.mask_file,
                upscale_x=UPSX_0,
                upscale_y=UPSY_0,
            )
            filepaths = simulation_files(params.simul_data)
            sdlA = simulation(
                filepath=filepaths[cam_a][params.dataset_type],
                energy_range=params.energy_range,
                coords=params.coords,
            )
            sdlB = simulation(
                filepath=filepaths[cam_b][params.dataset_type],
                energy_range=params.energy_range,
                coords=params.coords,
            )
            sdls = (sdlA, sdlB)

            with timer("Compute dets/vars"):
                detectors = tuple(count(wfm, sdl.data)[0] for sdl in sdls)
                variances = tuple(variance(wfm, d) for d in detectors)

            # WCS fit (here the camera is upscaled with the final upscaling)
            with timer("WCS fit"):
                wfm_WCS = codedmask(
                    mask_filepath=params.mask_file,
                    upscale_x=UPSX_FINAL,
                    upscale_y=UPSY_FINAL,
                )
                wcs_fit = tuple(iros.fit_WCS(wfm_WCS, sdl) for sdl in sdls)

        # SAVING SIMULATED SKIES
        print("\n#### Saving simulated skies...")
        if (
            not Path(params.simul_names[0]).is_file() or
            not Path(params.simul_names[1]).is_file()
        ):
            skies = tuple(decode(wfm, d) for d in detectors)
            snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skies, variances))

            # ups_skies = tuple(upscale(sky, upscale_y=UPY_TO) for sky in skies)
            # ups_snrs = tuple(upscale(snr, upscale_y=UPY_TO) for snr in snrs)

            for res, snr, sdl, name, wcs in zip(skies, snrs, sdls, params.simul_names, wcs_fit):
                if not Path(name).is_file():
                    iros.save_sky(res, snr, sdl, name, wcs)
        else:
            print("# Simulated skies already saved!")

        if not Path(params.simul_comp_name).is_file() and params.sky_compositions:
            with timer("Camera composition"):
                iros.camera_composition(
                    skyA_path=params.simul_names[0],
                    skyB_path=params.simul_names[1],
                    save_to=params.simul_comp_name,
                )

        # RUN IROS AND SAVE OUTPUT + RESIDUES
        print("\n#### Running IROS...")
        check_db = Path(params.iros_output_name).is_file()
        check_skyA = Path(params.res_names[0]).is_file()
        check_skyB = Path(params.res_names[1]).is_file()
        if not (check_db and check_skyA and check_skyB):
            iros_output, skies = iros.perform_iros(
                camerasID=params.wfm_cameras,
                camera=wfm,
                sdl_camA=sdlA,
                sdl_camB=sdlB,
                max_iterations=params.iros_max_iterations,
                snr_threshold=params.iros_snr_threshold,
                vignetting=params.vignetting,
                psfy=params.psfy,
            )

            if not check_db:
                iros.save_iros_output(iros_output, params.mask_file, params.iros_output_name)

            snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skies, variances))

            # ups_skies = tuple(upscale(sky, upscale_y=UPSY_FINAL - UPSY_0 + 1) for sky in skies)
            # ups_snrs = tuple(upscale(snr, upscale_y=UPSY_FINAL - UPSY_0 + 1) for snr in snrs)

            for check, res, snr, sdl, name, wcs in zip(
                (check_skyA, check_skyB), skies, snrs,
                sdls, params.res_names, wcs_fit,
                ):
                if not check:
                    iros.save_sky(res, snr, sdl, name, wcs)
        else:
            print("# IROS output data already saved!")
            iros_output = iros.load_iros_output(params.iros_output_name)
            skies = tuple(iros.load_sky(res)[0] for res in params.res_names)

        if not Path(params.res_comp_name).is_file() and params.sky_compositions:
            with timer("Camera composition"):
                iros.camera_composition(
                    skyA_path=params.res_names[0],
                    skyB_path=params.res_names[1],
                    save_to=params.res_comp_name,
                )

        # COMPUTE SOURCES PARAMS WITH IROS OUTPUT
        print("\n#### Computing sources parameters...")
        if not Path(params.iros_data_name).is_file():
            log = iros.gen_params_log(params.wfm_cameras)
            iros_data = iros.compute_params(
                iros_output=iros_output,
                camera=wfm,
                sdl_camA=sdlA,
                sdl_camB=sdlB,
                log=log,
            )

            iros.save_iros_data(
                data=iros_data,
                mask_file=params.mask_file,
                sdls=sdls,
                save_to=params.iros_data_name,
            )
        else:
            iros_data = iros.load_iros_data(params.iros_data_name)

        # CATALOG COMPARISON AND DATABASE UPDATE
        print("\n#### Performing catalog comparison...")
        if not Path(params.DB_name).is_file():
            catA, catB = iros.load_catalogs(
                catalogA=filepaths[cam_a]["sources"],
                catalogB=filepaths[cam_b]["sources"],
            )
            catA = filter_catalog(catA, n=params.n, flux_range=params.flux_range)
            catB = filter_catalog(catB, n=params.n, flux_range=params.flux_range)

            database = iros.catalog_comparison(
                data=iros_data,
                catalogA=catA,
                catalogB=catB,
                camerasID=params.wfm_cameras,
            )

            iros.save_iros_data(
                data=database,
                mask_file=params.mask_file,
                sdls=sdls,
                save_to=params.DB_name,
            )
        else:
            database = iros.load_iros_data(params.DB_name)

        # GENERATING SKIES FROM IROS OUTPUT + RESIDUES
        print("\n#### Generating and saving IROS output skies...")
        if (
            not Path(params.out_names[0]).is_file() or
            not Path(params.out_names[1]).is_file()
        ):
            with timer("IROS reconstructed skies"):
                skies = tuple(
                    iros.make_sky(database, camID, wfm, res, params.vignetting, params.psfy)
                    for camID, res in zip(params.wfm_cameras, skies)
                )
                snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skies, variances))

            # ups_skies = tuple(upscale(sky, upscale_y=UPSY_FINAL - UPSY_0 + 1) for sky in skies)
            # ups_snrs = tuple(upscale(snr, upscale_y=UPSY_FINAL - UPSY_0 + 1) for snr in snrs)

            for res, snr, sdl, name, wcs in zip(skies, snrs, sdls, params.out_names, wcs_fit):
                if not Path(name).is_file():
                    iros.save_sky(res, snr, sdl, name, wcs)
        else:
            print("# IROS reconstructed skies already saved!")

        if not Path(params.out_comp_name).is_file() and params.sky_compositions:
            with timer("Camera composition"):
                iros.camera_composition(
                    skyA_path=params.out_names[0],
                    skyB_path=params.out_names[1],
                    save_to=params.out_comp_name,
                )


    # final check on pipeline files
    output_pipeline_files(params, check_out=False)


# end