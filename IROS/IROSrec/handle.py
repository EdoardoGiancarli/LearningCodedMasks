"""
Module for the IROS sky-reconstruction pipeline file handling and data setup.
"""

import os
from pathlib import Path
import yaml
from typing import Any, Sequence

from numpy.typing import NDArray

from bloodmoon.types import CoordEquatorial
from bloodmoon.mask import CodedMaskCamera 
import darksun as ds

from .iros.procedure import run_IROS_loop
from .dtypes import AnalysisParams
from .dtypes import IROSParams
from .dtypes import OutFileNames
from .dtypes import WMFilters
from .dtypes import PipelineParams


def config_dirpaths(
    mask: str,
    skyfield: str,
    simul: str,
    runID: str | None = None,
) -> tuple[str, str, str]:
    """Handles paths depending on the OS."""
    # TODO: use os.path.join to be more general (still, the root folder has to be written as `/mnt`...)
    OS_SELECT = {
        'DEBIAN': '/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data',
        'WSL': '/mnt/d/PhD_AASS/Coding/Images_fits',
    }
    # define paths for data and output files
    if Path(base_path := OS_SELECT['DEBIAN']).is_dir():
        mask_path = f"{base_path}/Simulations/{mask}"                 # dirpath to cameras mask file
        data_path = f"{base_path}/Simulations/{skyfield}/{simul}/"    # dirpath with simul files

        if runID:
            save_path = f"{base_path}/Outputs/Out{skyfield}/{simul}"  # dirpath to save output data
            if not Path(f"{save_path}/{runID}").is_dir():
                os.mkdir(f"{save_path}/{runID}")
            save_path += f"/{runID}/"
        else:
            save_path = None
        
    elif Path(base_path := OS_SELECT['WSL']).is_dir():
        mask_path = f"{base_path}/{mask}"
        data_path = f"{base_path}/{skyfield}/{simul}/"

        if runID:
            save_path = base_path
            if not Path(f"{save_path}/{runID}").is_dir():
                os.mkdir(f"{save_path}/{runID}")
            save_path += f"/{runID}/"
        else:
            save_path = None

    else:
        raise ValueError("A0, ma ndo sei finit*?")
    
    # check mask FITS file and paths
    if not Path(mask_path).is_file():
        raise ValueError(f"Camera coded-mask '{mask}' does not exist.")
    for name, dirpath in zip(
            ("data_path", "save_path"),
            (data_path, save_path),
        ):
            if (dirpath and not Path(dirpath).is_dir()):
                raise ValueError(f"{name} '{dirpath}' does not exist.")

    return mask_path, data_path, save_path


def config_out_filenames(
    unit_camsID: tuple[str, str],
    save_to: str | Path,
) -> OutFileNames:
    """
    Configures the names for the pipeline output files.
    
    Contains:
        * simulated TRUE skies
        * iros output DB and sky residuals after IROS procedure
        * sources and catalog-compared DB
        * IROS reconstructed skies
        * pipeline output .reg files for camera module and .yaml file with pipeline's params
    """
    cam_a, cam_b = unit_camsID
    filenames: dict[str, str] = {}

    filenames['unit_camsID'] = unit_camsID

    filenames['sim_sky'] = tuple(save_to + f"sky_SIMUL_{cam.upper()}.fits" for cam in unit_camsID)
    filenames['comp_sim_sky'] = save_to + f"COMPOSED_sky_SIMUL_{cam_a.upper()}_{cam_b.upper()}.fits"

    filenames['out_db'] = save_to + f"IROS_output_db.fits"
    filenames['iros_res'] = tuple(save_to + f"skyRES_IROS_{cam.upper()}.fits" for cam in unit_camsID)
    filenames['comp_iros_res'] = save_to + f"COMPOSED_skyRES_IROS_{cam_a.upper()}_{cam_b.upper()}.fits"

    filenames['srcs_db'] = save_to + f"IROS_sources_db.fits"

    filenames['out_sky'] = tuple(save_to + f"OUTsky_IROS_{cam.upper()}.fits" for cam in unit_camsID)
    filenames['comp_out_sky'] = save_to + f"COMPOSED_OUTsky_IROS_{cam_a.upper()}_{cam_b.upper()}.fits"

    filenames['out_reg'] = tuple(save_to + f"OUTregionfile_IROS_{cam.upper()}.reg" for cam in unit_camsID)
    filenames['pipeline_params'] = save_to + f"OUTpipelinefile.yaml"

    return OutFileNames(**filenames)


def check_outfiles_exist(
    filenames: OutFileNames,
    check_out: bool = True,
) -> None:
    """
    Prints out which pipeline files have been saved.
    """
    def check(filename: str) -> str:
        return "SAVED" if Path(filename).is_file() else "MISSING"
    
    cam_a, cam_b = filenames.unit_camsID
    pipeline_files = {
        "Simulation Files": [
            (f"Simulated Sky {cam_a.upper()}", filenames.sim_sky[0]),
            (f"Simulated Sky {cam_b.upper()}", filenames.sim_sky[1]),
            ("Simulated Sky (composed)", filenames.comp_sim_sky),
        ],
        "Residuals Files": [
            (f"Residual Sky {cam_a.upper()}", filenames.iros_res[0]),
            (f"Residual Sky {cam_b.upper()}", filenames.iros_res[1]),
            ("Residual Sky (composed)", filenames.comp_iros_res),
        ],
        "IROS Output Files": [
            (f"Output Sky {cam_a.upper()}", filenames.out_sky[0]),
            (f"Output Sky {cam_b.upper()}", filenames.out_sky[1]),
            ("Output Sky (composed)", filenames.comp_out_sky),
        ],
        "Databases Files": [
            ("IROS Output", filenames.out_db),
            ("Sources Database", filenames.srcs_db),
            ("Pipeline Parameters", filenames.pipeline_params),
            (f"File Region {cam_a.upper()}", filenames.out_reg[0]),
            (f"File Region {cam_b.upper()}", filenames.out_reg[1]),
        ],
    }
    check_list = []
    print("\n#### GENERATED FILES")
    for category, files_list in pipeline_files.items():
        print(f"# {category}")
        for name, path in files_list:
            check_list.append(status := check(path))
            print(f"  - {name}: {status}")
    print('\n')
    
    if check_out and "MISSING" not in check_list:
        print("\nAll files present!")
        exit()

    return


def save_pipeline_params(
    params: PipelineParams,
    save_to: str | Path,
) -> None:
    """
    Saves a `.yaml` file with info about the IROS pipeline, indicating all
    the useful parameters for replicating the reconstruction procedure.
    """
    _init_comment = (
        f'#### OUTPUT FILE FOR IROS RECONSTRUCTION \n'
        f'# This file serves as a container for all the parameters/info to replicate the IROS run.\n'
        f'\n'
        f'# ARGS:\n'
        f'#     - `mask_file`, `simul_data` and `save_path` are the paths to the mask and data files;\n'
        f'#     - `vignetting` and `psfy` represent the active instrumental effects;\n'
        f'#     - `module_cameras` contains the ID for the two coded-mask cameras of the LEM-X module;\n'
        f'#     - `dataset` is the photons reconstruction logic-type of the data;\n'
        f'#     - `start_ups` and `final_ups` are the starting and final images upsampling factors in the (fine, coarse) directions;\n'
        f'#     - `iros_max_iterations` is the max number of IROS iterations;\n'
        f'#     - `iros_snr_threshold` is the significance threshold for the candidates validation;\n'
        f'#     - `sky_compositions` is a flag for the cameras sky images composition;\n'
        f'#     - `smoothing` is a flag for performing a smoothing of the observed sky-field detector image;\n'
        f'#     - `smoothing_thresh` is a SNR threshold for the brightest sources to perform the smoothing with;\n'
        f'#     - `smoothing_baseline_recnstr` is the path to a non-smoothed IROS reconstruction analysis, if present;\n'
        f'#     - `simul_names`, ..., `pipeline_outfile` are the names with which the pipeline files are saved (skies, databases, `.reg`);\n'
        f'#     - `E_min` and `E_max` are the limits on the data energy range (in keV);\n'
        f'#     - `coords` contains the sources RA/Dec coords that have been filtered out from the analysis (filtered out photons);\n'
        f'#     - `F_min` and `F_max` are the limits on the catalogue sources flux range for the candidates comparison (in ph/cm2/s);\n'
        f'\n\n'
    )
    _end_comment = (
        f'\n\n'
        f'# end'
    )
    dict_: dict[str, Any] = {}
    for p in params: dict_.update(p.__dict__.items())
    to_yaml = yaml.dump(dict_, indent=4, sort_keys=False)

    with open(save_to, "w", encoding="utf-8") as f:
        f.write(_init_comment)
        f.write(to_yaml)
        f.write(_end_comment)
    
    return


def config_parameters(
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
    iros_max_iterations: int,
    iros_snr_threshold: int | float,
    sky_compositions: bool,
    smoothing: bool,
    smoothing_thresh: int | float | None,
    smoothing_baseline_recnstr: str | Path | None,
    energy_range: tuple[int | float | None, int | float | None] | None,
    coords: CoordEquatorial | Sequence[CoordEquatorial] | None,
    flux_range: tuple[int | float | None, int | float | None] | None,
    verbose: bool = True,
) -> PipelineParams:
    """
    Configures the IROS pipeline by processing input parameters.
    """
    #### CHECKS
    # - check dataset type
    if dataset not in ["detected", "reconstructed"]:
        raise ValueError("'dataset' must be either 'detected' or 'reconstructed'.")
    # - check is smoothing baseline IROS files exist and if databases can be loaded
    if smoothing and smoothing_baseline_recnstr is not None:
        _db_file = tuple(Path(smoothing_baseline_recnstr).glob('*IROS_sources_db*.fits'))
        if not _db_file:
            raise FileNotFoundError(
                f"Invalid directory for detector smoothing baseline reconstruction: '{smoothing_baseline_recnstr}'. "
                f"Database file with sources info and data NOT present (i.e., 'IROS_sources_database...' FITS file)."
            )
        else:
            smoothing_baseline_recnstr = _db_file[0]
    # - check smoothing significance threshold value (must be at least 5.0)
    if smoothing and smoothing_thresh < 5.0:
        print('Detector smoothing threshold too small. Automathically setting to 5.0.')
        smoothing_thresh = max(5.0, smoothing_thresh)
    
    #### SETUP PIPELINE
    # - file directory paths (mask, simulated, where to save output files) / instrum. effects
    mask_file, simul_data, save_path = config_dirpaths(
        mask=mask,
        skyfield=skyfield,
        simul=skydata,
        runID=analysisID,
    )
    vignetting = not thin_mask
    psfy = False if dataset == "detected" else True
    analysis_params: AnalysisParams = AnalysisParams(
        mask_file=mask_file,
        simul_data=simul_data,
        save_path=save_path,
        vignetting=vignetting,
        psfy=psfy,
        unit_camsID=unit_camsID,
        dataset=dataset,
        start_ups=start_ups,
        final_ups=final_ups,
        sky_compositions=sky_compositions,
    )
    # - IROS params
    iros_params: IROSParams = IROSParams(
        iros_max_iterations=iros_max_iterations,
        iros_snr_threshold=iros_snr_threshold,
        smoothing=smoothing,
        smoothing_thresh=smoothing_thresh,
        smoothing_baseline_recnstr=smoothing_baseline_recnstr,
    )
    # - output files names
    out_filenames: OutFileNames = config_out_filenames(unit_camsID, save_path)
    # - filters params
    E_min, E_max = energy_range if energy_range is not None else (None, None)
    F_min, F_max = flux_range if flux_range is not None else (None, None)
    filters: WMFilters = WMFilters(
        E_min=E_min,
        E_max=E_max,
        coords=coords,
        F_min=F_min,
        F_max=F_max,
    )

    if verbose:
        print(
            f"\n# IROS Pipeline Report\n"
            f"  - Testing skyfield: '{skyfield}'\n"
            f"  - Output folder name: '{analysisID}'\n"
            f"  - Dataset type: '{dataset}'\n"
            f"  - Mask type: '{"ideal" if thin_mask else "realistic"}'\n"
            f"  - Vignetting: {vignetting}\n"
            f"  - Psfy: {psfy}\n"
            f"  - Starting upscaling (x, y): {start_ups}\n"
            f"  - Final upscaling (x, y): {final_ups}\n"
            f"  - Max IROS iteration: {iros_max_iterations}\n"
            f"  - Sky compositions: {sky_compositions}\n"
            f"  - Detector BKG smoothing: {smoothing}\n"
            f"  - Filtered photons energy range [keV]: {energy_range}\n"
            f"  - Excluded photons RA/Dec [deg]: {coords}\n"
            f"  - Catalog sources flux min/range [ph/cm2/s]: {flux_range}\n"
        )

    return PipelineParams(analysis_params, iros_params, out_filenames, filters)


def perform_BKG_smoothing(
    camera: CodedMaskCamera,
    detector: NDArray,
    varmap: NDArray,
    cameraID: str,
    smoothing_baseline_recnstr: str | Path,
    smoothing_thresh: float,
    vignetting: bool = True,
    psfy: bool = True,
    **iros_kwargs: Any,
) -> NDArray:
    """
    Performs the smoothing of the BKG on the detector.
    """
    print("# Initialising detector smoothing...")
    # sky-field brightest sources (SNR > smoothing_thresh)
    # - if is not possible to retrieve the sky-field brightest sources from
    #   a pre-made IROS analysis, then a pre-smoothing IROS reconstruction
    #   will be performed to get the brightest cands in the detector image
    if smoothing_baseline_recnstr is not None:
        # load baseline IROS reconstruction data
        print('Loading baseline IROS reconstruction for smoothing...')
        logA, logB = ds.load_database(smoothing_baseline_recnstr)
        camera_log = (
            logA if cameraID.lower() in logA.name.lower()
            else logB
        )
        brightest_cands = ds.get_candidates(camera_log, smoothing_thresh)
    else:
        print('No baseline IROS reconstruction for smoothing selected')
        brightest_cands = run_IROS_loop(
            detector=detector,
            camera=camera,
            snr_threshold=smoothing_thresh,
            vignetting=vignetting,
            psfy=psfy,
            varmap=varmap,
            **iros_kwargs,
        )
    # perform detector smoothing and run again IROS on the processed data.
    # To do that, we first remove the stored sources from the original
    # detector, and then we perform the smoothing
    print('Performing detector smoothing...')
    detector_ = ds.detector_smoothing(
        detector=detector,
        candidates=brightest_cands,
        camera=camera,
        vignetting=vignetting,
        psfy=psfy,
    )
    print('Detector succesfully smoothed!')
    return detector_
    

# end