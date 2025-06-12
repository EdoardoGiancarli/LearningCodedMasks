"""
Support methods for the IROS pipeline.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Sequence

__all__ = [
    "_handle_dirpaths", "handle_simul_corrections",
    "iros_pipeline_report", "PipelineParams",
    "initialize_pipeline", "output_pipeline_files",
]


def _handle_dirpaths(
    mask: str,
    skyfield: str,
    simul: str,
    test_name: str,
) -> tuple[str]:
    """Handles paths depending on the OS."""

    if Path(base_path := "/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data/").is_dir():
        mask_path = base_path + "Simulations/" + mask                                  # dirpath to WFM mask file 
        data_path = base_path + "Simulations/" + skyfield + "/" + simul + "/"          # dirpath with simul files
        save_path = base_path + "Outputs/" + "Out" + skyfield + "/" + simul + "/"      # dirpath to save output data
        if not Path(save_path + test_name).is_dir():
            os.mkdir(save_path + test_name)
        save_path += test_name + "/"
        
    elif Path(base_path := "/mnt/d/PhD_AASS/Coding/Images_fits/").is_dir():
        mask_path = base_path + mask
        data_path = base_path + skyfield + "/" + simul + "/"
        save_path = base_path
        if not Path(save_path + test_name).is_dir():
            os.mkdir(save_path + test_name)
        save_path += test_name + "/"

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


def handle_simul_corrections(
    ideal_mask: bool,
    dataset: str,
) -> tuple[bool, bool]:
    """Handles CAI vignetting and psf corrections along y-axis."""

    if dataset not in ["detected", "reconstructed"]:
        raise ValueError("dataset must be either 'detected' or 'reconstructed'.")
    
    vignetting = not ideal_mask
    psfy = False if dataset == "detected" else True
    return vignetting, psfy


def iros_pipeline_report(
    skyfield: tuple[str],
    test_name: str,
    dataset: str,
    mask_type: bool,
    vignetting: bool,
    psfy: bool,
    start_upscaling: tuple[int],
    final_upscaling: tuple[int],
    iros_iterations: int,
    sky_composition: bool,
    energy_range: int | tuple[int, int] | None = None,
    coords: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
    n: int | tuple[int, int] | None = None,
    flux_range: int | float | tuple[int | float, int | float] | None = None,
) -> None:
    """Prints out some IROS pipeline info."""
    print(
        f"\n# IROS Pipeline Report\n"
        f"  - Testing skyfield: '{skyfield[0]}'\n"
        f"  - Test name: '{test_name}'\n"
        f"  - Dataset type: '{dataset}'\n"
        f"  - Mask type: '{"ideal" if mask_type else "realistic"}'\n"
        f"  - Vignetting: {vignetting}\n"
        f"  - Psfy: {psfy}\n"
        f"  - Starting upscaling (x, y): {start_upscaling}\n"
        f"  - Final upscaling (x, y): {final_upscaling}\n"
        f"  - Max IROS iteration: {iros_iterations}\n"
        f"  - Sky compositions: {sky_composition}\n"
        f"  - Simulated photons energy range [keV]: {energy_range}\n"
        f"  - Excluded photons RA/Dec [deg]: {coords}\n"
        f"  - Catalog selected brighest sources: {n}\n"
        f"  - Catalog sources flux min/range [ph/cm2/s]: {flux_range}\n"
    )


@dataclass(frozen=True)
class PipelineParams:
    """
    Container for the IROS pipeline parameters.

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
        wfm_cameras (tuple[str, str]):
            Name of the WFM cameras (e.g., `('cam1a', 'cam1b')`).
        dataset_type (str):
            Photons position reconstruction effects. Either 'detected' or 'reconstructed'.
        start_ups (tuple[int, int]):
            Starting upscaling values.
        end_ups (tuple[int, int]):
            Final upscaling values.
        iros_max_iterations (int, optional (default=20)):
            Maximum number of iterations for the IROS loop.
        iros_snr_threshold (int | float, optional (default=5)):
            Minimum SNR value required to continue the iterative source removal process.
        sky_compositions (bool, optional (default=False)):
            Flag for WFM sky compositions.
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
        energy_range (int | float | tuple[int | float, int | float] | None):
            Energy range in keV for the data filtering.
        coords (tuple[float, float] | Sequence[tuple[float, float]] | None):
            Input photons RA/Dec (or sequence of RA/Dec) to filter out.
        n (int | tuple[int] | None):
            Filtered interval of sources.
        flux_range (int | float | tuple[int | float, int | float] | None, optional (default=None)):
            Flux range in ph/cm2/s for the data filtering.
    """
    mask_file: str
    simul_data: str
    save_path: str
    vignetting: bool
    psfy: bool
    wfm_cameras: tuple[str, str]
    dataset_type: str
    start_ups: tuple[int, int]
    end_ups: tuple[int, int]
    iros_max_iterations: int
    iros_snr_threshold: int | float
    sky_compositions: bool
    simul_names: tuple[str, str]
    simul_comp_name: str
    iros_output_name: str
    res_names: tuple[str, str]
    res_comp_name: str
    iros_data_name: str
    DB_name: str
    out_names: tuple[str, str]
    out_comp_name: str
    energy_range: int | tuple[int, int] | None
    coords: tuple[float, float] | Sequence[tuple[float, float]] | None
    n: int | tuple[int, int] | None
    flux_range: int | float | tuple[int | float, int | float] | None


def initialize_pipeline(
    mask: str,
    ideal_mask: bool,
    skyfield: str,
    skydata: str,
    wfm_cameras: tuple[str],
    dataset_type: str,
    start_ups: tuple[int],
    end_ups: tuple[int],
    testID: str,
    iros_max_iterations: int = 20,
    iros_snr_threshold: int | float = 5,
    sky_compositions: bool = False,
    energy_range: int | tuple[int, int] | None = None,
    coords: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
    n: int | tuple[int, int] | None = None,
    flux_range: int | float | tuple[int | float, int | float] | None = None,
) -> PipelineParams:
    """
    Initializes the IROS pipeline by processing input parameters.

    Args:
        mask (str):
            Name of the mask FITS file.
        ideal_mask (bool):
            Indicates if the mask is infinite thin and/or absorbent or realistic.
        skyfield (str):
            Name of the sky-field simulation (e.g., 'Crab', 'GalacticCenter', ...).
        skydata (str):
            Name of the directory with the sky-data simulation.
        wfm_cameras (tuple[str]):
            Name of the WFM cameras (e.g., `('cam1a', 'cam1b')`).
        dataset_type (str):
            Photons position reconstruction effects. Either 'detected' or 'reconstructed'.
        start_ups (tuple[int]):
            Starting upscaling values.
        end_ups (tuple[int]):
            Final upscaling values.
        testID (str):
            Test name.
        iros_max_iterations (int, optional (default=20)):
            Maximum number of iterations for the IROS loop.
        iros_snr_threshold (int | float, optional (default=5)):
            Minimum SNR value required to continue the iterative source removal process.
        sky_compositions (bool, optional (default=False)):
            Flag for WFM sky compositions.
        energy_range (int | float | tuple[int | float, int | float] | None, optional (default=None)):
            Energy range in keV for the data filtering. If a specific energy
            is given, this will be considered as the maximum filter value.
            If a tuple is given, it's interpreted as (`E_min`, `E_max`).
        coords (tuple[float, float] | Sequence[tuple[float, float]] | None, optional (default=None)):
            Input photons RA/Dec (or sequence of RA/Dec) to filter out.
        n (int | tuple[int] | None, optional (default=None)):
            Filtered interval of sources, up to the n-th brightest
            source or from `n[0]` to `n[1]` if `n` is a tuple.
        flux_range (int | float | tuple[int | float, int | float] | None, optional (default=None)):
            Flux range in ph/cm2/s for the data filtering. If a specific flux
            is given, this will be considered as the minimum filter value.
            If a tuple is given, it's interpreted as (`F_min`, `F_max`).
    
    Returns:
        params (PipelineParams):
            Output PipelineParams instance with all the parameters to run the pipeline.
    
    Raises:
        ValueError: If `n` or `flux_range` are both specified for catalogs filtering.
    """
    # configure n and flux_range for catalogs filtering
    if n and flux_range:
        raise ValueError("Specify either 'n' or 'flux_range' to filter the catalog.")

    # file directory paths (mask, simulated, where to save output files)
    mask_file, simul_data, save_path = _handle_dirpaths(
        mask=mask,
        skyfield=skyfield,
        simul=skydata,
        test_name=testID,
    )

    # detector corrections
    vignetting, psfy = handle_simul_corrections(ideal_mask, dataset_type)
    
    # output files names (simul skies, iros output DB and sky residuals, sources and catalog-compared DB, IROS skies)
    cam_a, cam_b = wfm_cameras

    simul_names = tuple(save_path + f"sky_SIMUL_{cam.upper()}_TEST_{testID}.fits" for cam in (cam_a, cam_b))
    simul_comp_name = save_path + f"COMPOSED_sky_SIMUL_{cam_a.upper()}_{cam_b.upper()}_TEST_{testID}.fits"

    iros_output_name = save_path + f"IROS_output_TEST_{testID}.fits"
    res_names = tuple(save_path + f"skyRES_IROS_{cam.upper()}_TEST_{testID}.fits" for cam in (cam_a, cam_b))
    res_comp_name = save_path + f"COMPOSED_skyRES_IROS_{cam_a.upper()}_{cam_b.upper()}_TEST_{testID}.fits"

    iros_data_name = save_path + f"IROS_data_TEST_{testID}.fits"
    DB_name = save_path + f"IROS_sources_database_TEST_{testID}.fits"

    out_names = tuple(save_path + f"OUTsky_IROS_{cam.upper()}_TEST_{testID}.fits" for cam in (cam_a, cam_b))
    out_comp_name = save_path + f"COMPOSED_OUTsky_IROS_{cam_a.upper()}_{cam_b.upper()}_TEST_{testID}.fits"

    params = PipelineParams(
        mask_file=mask_file,
        simul_data=simul_data,
        save_path=save_path,
        vignetting=vignetting,
        psfy=psfy,
        wfm_cameras=wfm_cameras,
        dataset_type=dataset_type,
        start_ups=start_ups,
        end_ups=end_ups,
        iros_max_iterations=iros_max_iterations,
        iros_snr_threshold=iros_snr_threshold,
        sky_compositions=sky_compositions,
        simul_names=simul_names,
        simul_comp_name=simul_comp_name,
        iros_output_name=iros_output_name,
        res_names=res_names,
        res_comp_name=res_comp_name,
        iros_data_name=iros_data_name,
        DB_name=DB_name,
        out_names=out_names,
        out_comp_name=out_comp_name,
        energy_range=energy_range,
        coords=coords,
        n=n,
        flux_range=flux_range,
    )

    iros_pipeline_report(
        skyfield=(skyfield, skydata),
        test_name=testID,
        dataset=dataset_type,
        mask_type=ideal_mask,
        vignetting=vignetting,
        psfy=psfy,
        start_upscaling=start_ups,
        final_upscaling=end_ups,
        iros_iterations=iros_max_iterations,
        sky_composition=sky_compositions,
        energy_range=energy_range,
        coords=coords,
        n=n,
        flux_range=flux_range,
    )

    return params


def output_pipeline_files(
    params: PipelineParams,
    check_out: bool = True,
) -> None:
    """Prints out which pipeline files have been saved."""

    def check(_file: str) -> str:
        return "SAVED" if Path(_file).is_file() else "MISSING"
    

    cam_a, cam_b = params.wfm_cameras
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
    
    print("\n")
    if check_out and "MISSING" not in check_list:
        print("All files present!")
        exit()


# end