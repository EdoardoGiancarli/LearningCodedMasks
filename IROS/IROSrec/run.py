"""
Module for running the IROS sky-reconstruction pipeline for the LEM-X observatory.
"""

from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from bloodmoon.types import CoordEquatorial
from bloodmoon.io import simulation_files
from bloodmoon.mask import decode
from bloodmoon.mask import count
from bloodmoon.mask import variance
from bloodmoon.mask import snratio
from bloodmoon.mask import codedmask
import darksun as ds
from darksun.handle import save_region_file

from .iros.optim import iros_singleCAM
from .iros.procedure import run_IROS
from .iros.procedure import get_sources_database
from .dtypes import PipelineParams
from .handle import config_parameters
from .handle import check_outfiles_exist
from .handle import save_pipeline_params
from .handle import perform_BKG_smoothing


def run(params: PipelineParams) -> None:
    """
    Single coded-mask camera IROS pipeline body.

    Args:
        params (PipelineParams):
            PipelineParams instance with the initialized parameters for the pipeline.
    """
    def is_file(filename: str | Path) -> bool:
        """Checks if input `filename` file exists."""
        if not isinstance(filename, Path):
            filename = Path(filename)
        return filename.is_file()
    
    CAM_A, CAM_B = params.analysis_params.unit_camsID

    # upscaling setup
    UPSX_0, UPSY_0 = params.analysis_params.start_ups
    UPSX_FINAL, UPSY_FINAL = params.analysis_params.final_ups
    UPX_TO, UPY_TO = UPSX_FINAL - UPSX_0 + 1, UPSY_FINAL - UPSY_0 + 1

    # check on pipeline files
    check_outfiles_exist(params.filenames)

    # start pipeline
    with ds.timer("##### IROS PIPELINE #####"):

        # --- IROS SETUP
        print("\n#### IROS Setup...")
        with ds.timer("IROS Setup"):
            wfm = codedmask(
                mask_filepath=params.analysis_params.mask_file,
                upscale_x=UPSX_0,
                upscale_y=UPSY_0,
                # hide_bulk_els_y=1.5,
            )
            filepaths = simulation_files(params.analysis_params.simul_data)
            sdlA = ds.get_data(
                filepath=filepaths[CAM_A][params.analysis_params.dataset],
                E_min=params.filters.E_min,
                E_max=params.filters.E_max,
                coords=params.filters.coords,
            )
            sdlB = ds.get_data(
                filepath=filepaths[CAM_B][params.analysis_params.dataset],
                E_min=params.filters.E_min,
                E_max=params.filters.E_max,
                coords=params.filters.coords,
            )
            sdls = (sdlA, sdlB)

            with ds.timer("Compute dets/vars"):
                detectors = tuple(count(wfm, sdl.DLdata)[0] for sdl in sdls)
                variances = tuple(variance(wfm, d) for d in detectors)

            # WCS fit (here the camera is upscaled with the final upscaling)
            with ds.timer("WCS fit"):
                wfm_WCS = codedmask(
                    mask_filepath=params.analysis_params.mask_file,
                    upscale_x=UPSX_FINAL,
                    upscale_y=UPSY_FINAL,
                    # hide_bulk_els_y=1.5,
                )
                wcs_fit = tuple(ds.fit_WCS(wfm_WCS, sdl) for sdl in sdls)
        

        # --- SAVING SIMULATED SKIES
        print("\n#### Saving Simulated Skies...")
        sim_camA, sim_camB = params.filenames.sim_sky
        if not is_file(sim_camA) or not is_file(sim_camB):
            skies = tuple(decode(wfm, d) for d in detectors)
            snrs = tuple(snratio(sky, var_) for sky, var_ in zip(skies, variances))

            # ups_skies = tuple(upscale(sky, upscale_y=UPY_TO) for sky in skies)
            # ups_snrs = tuple(upscale(snr, upscale_y=UPY_TO) for snr in snrs)

            for sky, snr, sdl, name, wcs in zip(skies, snrs, sdls, params.filenames.sim_sky, wcs_fit):
                if not is_file(name):
                    ds.save_sky(sky, snr, sdl, name, wcs)
        else:
            print("# Simulated skies already saved!")
        
        # sky composition
        if not is_file(params.filenames.comp_sim_sky) and params.analysis_params.sky_compositions:
            with ds.timer("Camera composition"):
                comp_sky, comp_snr, comp_WCS = ds.WFM_composition(
                    skyA_path=sim_camA,
                    skyB_path=sim_camB,
                )
                ds.save_sky(comp_sky, comp_snr, sdlA, params.filenames.comp_sim_sky, comp_WCS)
        

        # --- RUN IROS AND SAVE OUTPUT + RESIDUES
        print("\n#### Running IROS...")
        res_camA, res_camB = params.filenames.iros_res
        if not (
            is_file(params.filenames.out_db) and
            is_file(res_camA) and
            is_file(res_camB)
        ):
            def comp_fit_weights(obs: NDArray) -> NDArray:
                """Computes the weights for the loss metric in the optimisation procedure."""
                return 1.0 / np.sqrt(np.clip(obs, a_min=1.0, a_max=None))
    
            get_loop: Callable = partial(
                iros_singleCAM,
                camera=wfm,
                max_iterations=params.iros_params.iros_max_iterations,
                snr_threshold=params.iros_params.iros_snr_threshold,
                vignetting=params.analysis_params.vignetting,
                psfy=params.analysis_params.psfy,
                fit_weights=comp_fit_weights,
            )
            # BKG smoothing
            if params.iros_params.smoothing:
                detectors_ = tuple(
                    perform_BKG_smoothing(
                        camera=wfm,
                        detector=det,
                        varmap=var,
                        cameraID=camID,
                        smoothing_baseline_recnstr=params.iros_params.smoothing_baseline_recnstr,
                        smoothing_thresh=params.iros_params.smoothing_thresh,
                        vignetting=params.analysis_params.vignetting,
                        psfy=params.analysis_params.psfy,
                        fit_weights=comp_fit_weights,
                    )
                    for det, var, camID in zip(detectors, variances, params.analysis_params.unit_camsID)
                )
            else:
                detectors_ = detectors
            # IROS
            print(f'\nApplying IROS to {CAM_A.upper()}')
            loop = get_loop(detector=detectors_[0], varmap=variances[0])
            log_camA, skyA = run_IROS(wfm, loop, CAM_A)

            print(f'\nApplying IROS to {CAM_B.upper()}')
            loop = get_loop(detector=detectors_[1], varmap=variances[1])
            log_camB, skyB = run_IROS(wfm, loop, CAM_B)

            # save output databases
            if not is_file(params.filenames.out_db):
                ds.save_database(
                    log_camA=log_camA,
                    log_camB=log_camB,
                    sdlA=sdlA,
                    sdlB=sdlB,
                    save_to=params.filenames.out_db,
                )
            # save IROS sky residues
            skies = (skyA, skyB)
            snrs = tuple(snratio(sky, var_) for sky, var_ in zip(skies, variances))

            # ups_skies = tuple(upscale(sky, upscale_y=UPSY_FINAL - UPSY_0 + 1) for sky in skies)
            # ups_snrs = tuple(upscale(snr, upscale_y=UPSY_FINAL - UPSY_0 + 1) for snr in snrs)

            for sky, snr, sdl, name, wcs in zip(skies, snrs, sdls, params.filenames.iros_res, wcs_fit):
                if not is_file(name):
                    ds.save_sky(sky, snr, sdl, name, wcs)
        else:
            print("# IROS output data already saved!")
            log_camA, log_camB = ds.load_database(params.filenames.out_db)
            skies = tuple(ds.load_sky(res)[0] for res in params.filenames.iros_res)

        # sky composition
        if not is_file(params.filenames.comp_iros_res) and params.analysis_params.sky_compositions:
            with ds.timer("Camera composition"):
                comp_sky, comp_snr, comp_WCS = ds.WFM_composition(
                    skyA_path=res_camA,
                    skyB_path=res_camB,
                )
                ds.save_sky(comp_sky, comp_snr, sdlA, params.filenames.comp_iros_res, comp_WCS)
        

        # --- COMPUTE SOURCES ADDITIONAL PARAMS AND CATALOGUE COMPARISON
        print("\n#### Computing Sources Parameters and Performing Catalogue Comparison...")
        if not is_file(params.filenames.srcs_db):
            # - CAMERA A  |  DB + save region file
            catA = ds.get_catalogue(
                filepath=filepaths[CAM_A]["sources"],
                F_max=params.filters.F_max,
                F_min=params.filters.F_min,
            )
            log_camA = get_sources_database(wfm, sdlA, catA, log_camA, params.analysis_params.vignetting)
            save_region_file(log_camA, catA, params.filenames.out_reg[0])

            # - CAMERA B  |  DB + save region file
            catB = ds.get_catalogue(
                filepath=filepaths[CAM_B]["sources"],
                F_max=params.filters.F_max,
                F_min=params.filters.F_min,
            )
            log_camB = get_sources_database(wfm, sdlB, catB, log_camB, params.analysis_params.vignetting)
            save_region_file(log_camB, catB, params.filenames.out_reg[1])

            # - save DB
            ds.save_database(
                log_camA=log_camA,
                log_camB=log_camB,
                sdlA=sdlA,
                sdlB=sdlB,
                save_to=params.filenames.srcs_db,
            )
        else:
            print("# Output Sources DataBase already stored!")
            log_camA, log_camB = ds.load_database(params.filenames.srcs_db)


        # --- RECONSTRUCTING SKIES FROM IROS OUTPUT + RESIDUES
        print("\n#### Reconstructing and Saving IROS Output Skies...")
        out_camA, out_camB = params.filenames.out_sky
        logs = (log_camA, log_camB)
        if (
            not is_file(out_camA) or
            not is_file(out_camB)
        ):
            with ds.timer("IROS reconstructed skies"):
                skies = tuple(
                    ds.make_sky(
                        logID.log, wfm, vignetting=params.analysis_params.vignetting, psfy=params.analysis_params.psfy, background=res,
                    )
                    for logID, res in zip(logs, skies)
                )
                snrs = tuple(snratio(sky, var_) for sky, var_ in zip(skies, variances))

                # ups_skies = tuple(upscale(sky, upscale_y=UPSY_FINAL - UPSY_0 + 1) for sky in skies)
                # ups_snrs = tuple(upscale(snr, upscale_y=UPSY_FINAL - UPSY_0 + 1) for snr in snrs)

            for sky, snr, sdl, name, wcs in zip(skies, snrs, sdls, params.filenames.out_sky, wcs_fit):
                if not is_file(name):
                    ds.save_sky(sky, snr, sdl, name, wcs)
        else:
            print("# IROS reconstructed skies already saved!")
        
        # sky composition
        if not is_file(params.filenames.comp_out_sky) and params.analysis_params.sky_compositions:
            with ds.timer("Camera composition"):
                comp_sky, comp_snr, comp_WCS = ds.WFM_composition(
                    skyA_path=out_camA,
                    skyB_path=out_camB,
                )
                ds.save_sky(comp_sky, comp_snr, sdlA, params.filenames.comp_out_sky, comp_WCS)
    
    # save .yaml file with pipeline's parameters
    if not is_file(params.filenames.pipeline_params):
        save_pipeline_params(params, params.filenames.pipeline_params)

    # final check on pipeline files
    check_outfiles_exist(params.filenames, check_out=False)

    return


def run_pipeline(
    *,
    mask: str,
    thin_mask: bool,
    skyfield: str,
    skydata: str,
    unit_camsID: tuple[str, str],
    dataset: str,
    start_ups: tuple[int, int],
    final_ups: tuple[int, int],
    analysisID: str | None,
    iros_max_iterations: int = 20,
    iros_snr_threshold: int | float = 5,
    sky_compositions: bool = False,
    smoothing: bool,
    smoothing_thresh: int | float | None,
    smoothing_baseline_recnstr: str | Path | None,
    energy_range: tuple[int | float | None, int | float | None] | None,
    coords: CoordEquatorial | Sequence[CoordEquatorial] | None,
    flux_range: tuple[int | float | None, int | float | None] | None,
) -> None:
    """
    Runs the IROS pipeline.
    This method initializes the parameters for the pipeline and then runs
    the analysis procedure, divided in blocks and with respective checkpoints.
    """
    params: PipelineParams = config_parameters(
        mask=mask,
        thin_mask=thin_mask,
        skyfield=skyfield,
        skydata=skydata,
        unit_camsID=unit_camsID,
        dataset=dataset,
        start_ups=start_ups,
        final_ups=final_ups,
        analysisID=analysisID,
        iros_max_iterations=iros_max_iterations,
        iros_snr_threshold=iros_snr_threshold,
        sky_compositions=sky_compositions,
        smoothing=smoothing,
        smoothing_thresh=smoothing_thresh,
        smoothing_baseline_recnstr=smoothing_baseline_recnstr,
        energy_range=energy_range,
        coords=coords,
        flux_range=flux_range,
    )
    run(params)
    return


# end