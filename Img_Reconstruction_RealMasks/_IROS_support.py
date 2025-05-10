"""
Support methods for the IROS pipeline.
"""

import os
from pathlib import Path


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
) -> None:
    """Prints out some IROS pipeline info."""
    print(
        f"\n# IROS Pipeline Report\n"
        f"  - Testing skyfield: '{skyfield[0]}', simulation of: {skyfield[1][:4]}/{skyfield[1][4:6]}/{skyfield[1][6:8]}\n"
        f"  - Test name: '{test_name}'\n"
        f"  - Dataset type: '{dataset}'\n"
        f"  - Mask type: '{"ideal" if mask_type else "realistic"}'\n"
        f"  - Vignetting: {vignetting}\n"
        f"  - Psfy: {psfy}\n"
        f"  - Starting upscaling: {start_upscaling}\n"
        f"  - Final upscaling: {final_upscaling}\n"
        f"  - Max IROS iteration: {iros_iterations}\n"
        f"  - Sky compositions: {sky_composition}\n"
    )


def output_pipeline_files(files: dict) -> None:
    """Prints out which pipeline files have been saved."""
    def check(_file: str) -> str:
        return "SAVED" if Path(_file).is_file() else "MISSING"
    
    check_list = []
    print("\n#### GENERATED FILES")
    for category, files_list in files.items():
        print(f"# {category}")
        for name, path in files_list:
            check_list.append(status := check(path))
            print(f"  - {name}: {status}")
    
    print("\n")
    if "MISSING" not in check_list: exit()


def initialize_pipeline():
    """Initializes the IROS pipeline by processing input parameters."""
    #mask_file, simul_data, save_path = _handle_dirpaths(
    #    mask=mask_FITS,
    #    skyfield=skyfield,
    #    simul=data_FITS,
    #    test_name=N_TEST,
    #)
    #vignetting, psfy = handle_simul_corrections(IDEAL_MASK, dataset)
    raise NotImplementedError

# end