"""
Support methods for the IROS pipeline.
"""

import os
from typing import Sequence
from pathlib import Path
from dataclasses import dataclass
import yaml

import numpy as np

from bloodmoon.types import CoordEquatorial

from darksun.data import Log
from darksun.data import CatalogueLoader
from darksun.benchmarking import source_catalogue_data


OS_SELECT = {
    'DEBIAN': '/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data',
    'WSL': '/mnt/d/PhD_AASS/Coding/Images_fits',
}

def _handle_dirpaths(
    mask: str,
    skyfield: str,
    simul: str,
    run_name: str | None = None,
) -> tuple[str]:
    """Handles paths depending on the OS."""
    # TODO: use os.path.join to be more general (still, the root folder has to be written as `/mnt`...)

    # define paths for data and output files
    if Path(base_path := OS_SELECT['DEBIAN']).is_dir():
        mask_path = f"{base_path}/Simulations/{mask}"                 # dirpath to cameras mask file
        data_path = f"{base_path}/Simulations/{skyfield}/{simul}/"    # dirpath with simul files

        if run_name:
            save_path = f"{base_path}/Outputs/Out{skyfield}/{simul}"  # dirpath to save output data
            if not Path(f"{save_path}/{run_name}").is_dir():
                os.mkdir(f"{save_path}/{run_name}")
            save_path += f"/{run_name}/"
        else:
            save_path = None
        
    elif Path(base_path := OS_SELECT['WSL']).is_dir():
        mask_path = f"{base_path}/{mask}"
        data_path = f"{base_path}/{skyfield}/{simul}/"

        if run_name:
            save_path = base_path
            if not Path(f"{save_path}/{run_name}").is_dir():
                os.mkdir(f"{save_path}/{run_name}")
            save_path += f"/{run_name}/"
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


def handle_instrument_effects(
    thin_mask: bool,
    dataset: str,
) -> tuple[bool, bool]:
    """Handles CAI vignetting and psf corrections along y-axis."""

    if dataset not in ["detected", "reconstructed"]:
        raise ValueError("'dataset' must be either 'detected' or 'reconstructed'.")
    
    vignetting = not thin_mask
    psfy = False if dataset == "detected" else True
    return vignetting, psfy


@dataclass(frozen=True)
class PipelineParams:
    """
    Container for the IROS pipeline parameters, which configures the LEM-X coded-mask cameras
    module parameters, the IROS setup parameters, handles the pipeline output files setup
    (skyes, databases, info files), and manage the data photons energy range and catalogue
    fluxes range to perform the IROS sources candidates association with.

    Attributes:
        mask_file (str):
            Directory path to the mask FITS file.
        simul_data (str):
            Directory path to the simulated data FITS file.
        save_path (str):
            Output FITS files directory path.
        vignetting (bool):
            Flag for vignetting effect on the detector.
        psfy (bool):
            Flag for detector PSF effect along the y axis.
        module_cameras (tuple[str, str]):
            Name of the LEM-X module cameras (e.g., `('cam1a', 'cam1b')`).
        dataset (str):
            Photons position reconstruction effects. Either 'detected' or 'reconstructed'.
        start_ups (tuple[int, int]):
            Starting upscaling values (fine, coarse) directions.
        final_ups (tuple[int, int]):
            Final upscaling values (fine, coarse) directions.
        
        iros_max_iterations (int):
            Maximum number of iterations for the IROS loop.
        iros_snr_threshold (int | float):
            Minimum SNR value required to continue the iterative source removal process.
        sky_compositions (bool):
            Flag for LEM-X module sky compositions.
        smoothing (bool):
            Selects if detector smoothing is to be applied.
        smoothing_thresh (int | float | None):
            Significance threshold for brightest sources in sky-field (min is set to `5.0` automathically).
        smoothing_baseline_recnstr (str | Path | None):
            Path to non-smoothed IROS reconstruction directory, if present.
        
        simul_names (tuple[str, str]):
            Names for the simulated skies FITS files.
        simul_comp_name (str):
            Name for the simulated sky composition FITS file.
        iros_output_name (str):
            Name for the output IROS database FITS file.
        res_names (tuple[str, str]):
            Names for the IROS sky residuals FITS files.
        res_comp_name (str):
            Name for the IROS sky residuals composition FITS file.
        iros_data_name (str):
            Name for the output IROS sources parameters database FITS file.
        DB_name (str):
            Name for the output database with the identified sources parameters FITS file.
        out_names (tuple[str, str]):
            Names for the IROS reconstructed skies FITS files.
        out_comp_name (str):
            Name for the IROS reconstructed sky composition FITS file.
        region_outfiles (tuple[str, str]):
            Names for the output `.reg` files of the IROS reconstr for the LEM-X cameras.
        pipeline_outfile (str):
            Name for the output `.yaml` file with the pipeline's parameters.
        
        E_min (int | float | None):
            Minimum photons energy in [keV] for the observed data filtering.
        E_max (int | float | None):
            Maximum photons energy in [keV] for the observed data filtering.
        coords (CoordEquatorial | Sequence[CoordEquatorial] | None):
            Input photons RA/Dec (or sequence of RA/Dec) to filter out.
        n (int | tuple[int, int] | None):
            Filtered interval of sources.
        F_min (int | float | None):
            Minimum flux range in [ph/cm2/s] for the catalogue data filtering.
        F_max (int | float | None):
            Maximum flux range in [ph/cm2/s] for the catalogue data filtering.
    """
    # path to data (mask, WISEMAN events and directory to save output files)
    mask_file: str
    simul_data: str
    save_path: str
    # instrumental effects
    vignetting: bool
    psfy: bool
    # LEM-X module to apply IROS to, events reconstruction type and images sampling
    module_cameras: tuple[str, str]
    dataset: str
    start_ups: tuple[int, int]
    final_ups: tuple[int, int]
    # IROS setup (iterations, threshold, module sky images composition and detector smoothing)
    iros_max_iterations: int
    iros_snr_threshold: int | float
    sky_compositions: bool
    smoothing: bool
    smoothing_thresh: int | float | None
    smoothing_baseline_recnstr: str | Path | None
    # pipeline output files (skyes, databases, info files)
    simul_names: tuple[str, str]
    simul_comp_name: str
    iros_output_name: str
    res_names: tuple[str, str]
    res_comp_name: str
    iros_data_name: str
    DB_name: str
    out_names: tuple[str, str]
    out_comp_name: str
    region_outfiles: tuple[str, str]
    pipeline_outfile: str
    # WISEMAN photons energy range and catalogue fluxes range setup
    E_min: int | float | None
    E_max: int | float | None
    coords: CoordEquatorial | Sequence[CoordEquatorial] | None
    n: int | tuple[int, int] | None
    F_min: int | float | None
    F_max: int | float | None


def config_parameters(
    *,
    mask: str,
    thin_mask: bool,
    skyfield: str,
    skydata: str,
    module_cameras: tuple[str, str],
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
    coords: tuple[float, float] | Sequence[tuple[float, float]] | None,
    n: int | tuple[int, int] | None,
    flux_range: tuple[int | float | None, int | float | None] | None,
) -> PipelineParams:
    """
    Configures the IROS pipeline by processing input parameters.

    Args:
        mask (str):
            Name of the mask FITS file.
        thin_mask (bool):
            Indicates if the mask is infinite thin and/or absorbent or realistic.
        skyfield (str):
            Name of the sky-field simulation (e.g., 'Crab', 'GalacticCenter', ...).
        skydata (str):
            Name of the directory with the sky-data simulation.
        module_cameras (tuple[str, str]):
            Name of the LEM-X cameras module (e.g., `('cam1a', 'cam1b')`).
        dataset (str):
            Photons position reconstruction effects. Either 'detected' or 'reconstructed'.
        start_ups (tuple[int, int]):
            Starting upscaling values (x, y).
        final_ups (tuple[int, int]):
            Final upscaling values (x, y).
        analysisID (str | None):
            Test name.
        iros_max_iterations (int):
            Maximum number of iterations for the IROS loop.
        iros_snr_threshold (int | float):
            Minimum SNR value required to continue the iterative source removal process.
        sky_compositions (bool):
            Flag for LEM-X coded-mask camera module sky compositions.
        smoothing (bool):
            Selects if detector smoothing is to be applied.
        smoothing_thresh (int | float | None):
            Significance threshold for brightest sources in sky-field (min is set to `5.0` automathically).
        smoothing_baseline_recnstr (str | Path | None):
            Path to non-smoothed IROS reconstruction directory, if present.
        energy_range (tuple[int | float | None, int | float | None] | None):
            Energy range in keV for the data filtering, to be interpreted as (`E_min`, `E_max`).
        coords (tuple[float, float] | Sequence[tuple[float, float]] | None):
            Input photons RA/Dec (or sequence of RA/Dec) to filter out.
        n (int | tuple[int, int] | None):
            Filtered interval of sources, up to the n-th brightest
            source or from `n[0]` to `n[1]` if `n` is a tuple.
        flux_range (tuple[int | float | None, int | float | None] | None):
            Flux range in ph/cm2/s for the data filtering, to be interpreted as (`F_min`, `F_max`).
    
    Returns:
        params (PipelineParams):
            Output PipelineParams instance with all the parameters to run the pipeline.
    
    Raises:
        ValueError: If `n` or `flux_range` are both specified for catalogs filtering.
        FileNotFoundError: If invalid `smoothing_baseline_recnstr` path.
    """
    def report(params: PipelineParams) -> None:
        """Prints out IROS pipeline info."""
        print(
            f"\n# IROS Pipeline Report\n"
            f"  - Testing skyfield: '{skyfield}'\n"
            f"  - Output folder name: '{analysisID}'\n"
            f"  - Dataset type: '{dataset}'\n"
            f"  - Mask type: '{"ideal" if thin_mask else "realistic"}'\n"
            f"  - Vignetting: {params.vignetting}\n"
            f"  - Psfy: {params.psfy}\n"
            f"  - Starting upscaling (x, y): {start_ups}\n"
            f"  - Final upscaling (x, y): {final_ups}\n"
            f"  - Max IROS iteration: {iros_max_iterations}\n"
            f"  - Sky compositions: {sky_compositions}\n"
            f"  - Detector smoothing: {smoothing}\n"
            f"  - Filtered photons energy range [keV]: {energy_range}\n"
            f"  - Excluded photons RA/Dec [deg]: {coords}\n"
            f"  - Catalog selected brighest sources: {n}\n"
            f"  - Catalog sources flux min/range [ph/cm2/s]: {flux_range}\n"
        )

    # configure n and flux_range for catalogs filtering
    if n and flux_range:
        raise ValueError("Specify either 'n' or 'flux_range' to filter the catalog.")
    
    # check is smoothing baseline IROS files exist and if databases can be loaded
    if smoothing and smoothing_baseline_recnstr is not None:
        _db_file = tuple(Path(smoothing_baseline_recnstr).glob('IROS_sources_database*.fits'))
        if not _db_file:
            raise FileNotFoundError(
                f"Invalid directory for detector smoothing baseline reconstruction: '{smoothing_baseline_recnstr}'. "
                f"Database file with sources info and data NOT present (i.e., 'IROS_sources_database...' FITS file)."
            )
        else:
            smoothing_baseline_recnstr = _db_file[0]
    # check smoothing significance threshold value (must be at least 5.0)
    if smoothing and smoothing_thresh < 5.0:
        print('Detector smoothing threshold too small. Automathically setting to 5.0.')
        smoothing_thresh = max(5.0, smoothing_thresh)

    # file directory paths (mask, simulated, where to save output files)
    mask_file, simul_data, save_path = _handle_dirpaths(
        mask=mask,
        skyfield=skyfield,
        simul=skydata,
        run_name=analysisID,
    )

    # mask/detector corrections
    vignetting, psfy = handle_instrument_effects(thin_mask, dataset)
    
    # output files names
    #   - simulated TRUE skies
    #   - iros output DB and sky residuals after IROS procedure
    #   - sources and catalog-compared DB
    #   - IROS reconstructed skies
    #   - pipeline output .reg files for camera module and .yaml file with pipeline's params
    cam_a, cam_b = module_cameras

    simul_names = tuple(save_path + f"sky_SIMUL_{cam.upper()}_TEST_{analysisID}.fits" for cam in (cam_a, cam_b))
    simul_comp_name = save_path + f"COMPOSED_sky_SIMUL_{cam_a.upper()}_{cam_b.upper()}_TEST_{analysisID}.fits"

    iros_output_name = save_path + f"IROS_output_TEST_{analysisID}.fits"
    res_names = tuple(save_path + f"skyRES_IROS_{cam.upper()}_TEST_{analysisID}.fits" for cam in (cam_a, cam_b))
    res_comp_name = save_path + f"COMPOSED_skyRES_IROS_{cam_a.upper()}_{cam_b.upper()}_TEST_{analysisID}.fits"

    iros_data_name = save_path + f"IROS_data_TEST_{analysisID}.fits"
    DB_name = save_path + f"IROS_sources_database_TEST_{analysisID}.fits"

    out_names = tuple(save_path + f"OUTsky_IROS_{cam.upper()}_TEST_{analysisID}.fits" for cam in (cam_a, cam_b))
    out_comp_name = save_path + f"COMPOSED_OUTsky_IROS_{cam_a.upper()}_{cam_b.upper()}_TEST_{analysisID}.fits"

    region_outfiles = tuple(save_path + f"OUTregionfile_IROS_{cam.upper()}_TEST_{analysisID}.reg" for cam in (cam_a, cam_b))
    pipeline_outfile = save_path + f"OUTpipelinefile_TEST_{analysisID}.yaml"

    # filters params
    if (energy_range is None) or (not any(energy_range)):
        E_min, E_max = None, None
    elif any(energy_range):
        E_min, E_max = energy_range
    
    if (flux_range is None) or (not any(flux_range)):
        F_min, F_max = None, None
    elif any(flux_range):
        F_min, F_max = flux_range

    params = PipelineParams(
        mask_file=mask_file,
        simul_data=simul_data,
        save_path=save_path,
        vignetting=vignetting,
        psfy=psfy,
        module_cameras=module_cameras,
        dataset=dataset,
        start_ups=start_ups,
        final_ups=final_ups,
        iros_max_iterations=iros_max_iterations,
        iros_snr_threshold=iros_snr_threshold,
        sky_compositions=sky_compositions,
        smoothing=smoothing,
        smoothing_thresh=smoothing_thresh,
        smoothing_baseline_recnstr=smoothing_baseline_recnstr,
        simul_names=simul_names,
        simul_comp_name=simul_comp_name,
        iros_output_name=iros_output_name,
        res_names=res_names,
        res_comp_name=res_comp_name,
        iros_data_name=iros_data_name,
        DB_name=DB_name,
        out_names=out_names,
        out_comp_name=out_comp_name,
        region_outfiles=region_outfiles,
        pipeline_outfile=pipeline_outfile,
        E_min=E_min,
        E_max=E_max,
        coords=coords,
        n=n,
        F_min=F_min,
        F_max=F_max,
    )
    report(params)
    return params


def save_region_file(
    log: Log,
    catalogue: CatalogueLoader,
    save_to: str | Path,
) -> None:
    """
    Saves a `.reg` (region) file with info about the reconstructed
    IROS sources, i.e., their respective RA/Dec coordinates from
    the catalogue in the `fk5` reference frame.

    Args:
        log (Log):
            IROS reconstructed sources database.
        catalogue (CatalogueLoader):
            Catalogue data for the LEM-X coded-mask camera.
        save_to (str | Path):
            Path for where to save the `.reg` file.
    """
    SETUP = {
        'format': 'DS9 version 4.1',
        'refsystem': 'fk5',
        'circles_size': 400.0,

        'options': {
            'color': 'green',
            'dashlist': '8 3',
            'width': 1,
            'font': '"helvetica 10 normal roman"',
            'select': 1,
            'highlite': 1,
            'dash': 0,
            'fixed': 0,
            'edit': 1,
            'move': 1,
            'delete': 1,
            'include': 1,
            'source': 1,
        },
    }

    # populate list with sources location
    source_list = []
    for idx, sourceID in enumerate(log.log['ID']):
        if sourceID in catalogue.DLdata['ID']:
            source_data = source_catalogue_data(sourceID, catalogue.DLdata)
            source_list.append(
                (sourceID, source_data['RA'], source_data['DEC'])
            )
        else:
            source_list.append(
                (sourceID, log.log['ra'][idx], log.log['dec'][idx])
            )
    
    # write .reg file with custom options
    global_options = [
        f'{key}={value} ' for key, value in SETUP['options'].items()
    ]
    tags = [
        f'circle({ra}, {dec}, {SETUP['circles_size']}{'"'}) # text={{{sID}}}\n'
        for (sID, ra, dec) in source_list
    ]
    with open(save_to, "w", encoding="utf-8") as f:
        f.write(f'# Region file format: {SETUP['format']}\n')
        f.write('global ')
        f.writelines(global_options)
        f.write('\n')
        f.write(f'{SETUP['refsystem']}\n')
        f.writelines(tags)

    return None


def save_pipeline_params(
    params: PipelineParams,
    save_to: str | Path,
) -> None:
    """
    Saves a `.yaml` file with info about the IROS pipeline, indicating all
    the useful parameters for replicating the reconstruction procedure.

    Args:
        params (PipelineParams):
            PipelineParams instance with the initialized parameters for the pipeline.
        save_to (str | Path):
            Path for where to save the `.yaml` file.
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

    dict_ = params.__dict__
    to_yaml = yaml.dump(dict_, indent=4, sort_keys=False)

    with open(save_to, "w", encoding="utf-8") as f:
        f.write(_init_comment)
        f.write(to_yaml)
        f.write(_end_comment)
    
    return None


def output_files(
    params: PipelineParams,
    check_out: bool = True,
) -> None:
    """Prints out which pipeline files have been saved."""

    def check(filename: str) -> str:
        return "SAVED" if Path(filename).is_file() else "MISSING"
    
    cam_a, cam_b = params.module_cameras
    pipeline_files = {
        "Simulation Files": [
            (f"Simulated Sky {cam_a.upper()}", params.simul_names[0]),
            (f"Simulated Sky {cam_b.upper()}", params.simul_names[1]),
            ("Simulated Sky (composed)", params.simul_comp_name),
        ],
        "Residuals Files": [
            (f"Residual Sky {cam_a.upper()}", params.res_names[0]),
            (f"Residual Sky {cam_b.upper()}", params.res_names[1]),
            ("Residual Sky (composed)", params.res_comp_name),
        ],
        "IROS Output Files": [
            (f"Output Sky {cam_a.upper()}", params.out_names[0]),
            (f"Output Sky {cam_b.upper()}", params.out_names[1]),
            ("Output Sky (composed)", params.out_comp_name),
        ],
        "Databases Files": [
            ("IROS output DB", params.iros_output_name),
            ("IROS sources parameters", params.iros_data_name),
            ("IROS-catalog comparison", params.DB_name),
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


# end