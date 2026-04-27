"""
Module for I/O operations handling.
"""

import os
from pathlib import Path

from darksun.benchmarking import source_catalogue_data
from darksun.data import CatalogueLoader
from darksun.data import Log


def config_dirpaths(
    mask: str,
    skyfield: str,
    simul: str,
    runID: str | None = None,
) -> tuple[str]:
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


def config_filenames(
    runID: str | Path,
    unit_camsID: tuple[str, str] = ('cam1a', 'cam1b'),
) -> dict[str, str]:
    """Configures the names for the pipeline output files."""
    cam_a, cam_b = unit_camsID
    filenames: dict[str, str] = {}

    filenames['SIM_SKY'] = tuple(runID + f"sky_SIMUL_{cam.upper()}.fits" for cam in (cam_a, cam_b))
    filenames['COMP_SIMSKY'] = runID + f"COMPOSED_sky_SIMUL_{cam_a.upper()}_{cam_b.upper()}.fits"

    filenames['IROS_RES'] = tuple(runID + f"skyRES_IROS_{cam.upper()}.fits" for cam in (cam_a, cam_b))
    filenames['COMP_IROSRES'] = runID + f"COMPOSED_skyRES_IROS_{cam_a.upper()}_{cam_b.upper()}.fits"

    filenames['SRCS_DB'] = runID + f"IROS_sources_database.fits"

    filenames['OUT_SKY'] = tuple(runID + f"OUTsky_IROS_{cam.upper()}.fits" for cam in (cam_a, cam_b))
    filenames['COMP_OUTSKY'] = runID + f"COMPOSED_OUTsky_IROS_{cam_a.upper()}_{cam_b.upper()}.fits"

    filenames['OUT_REG'] = tuple(runID + f"OUTregionfile_IROS_{cam.upper()}.reg" for cam in (cam_a, cam_b))

    return filenames


def output_files(
    filenames: dict[str, str],
    unit_camsID: tuple[str, str] = ('cam1a', 'cam1b'),
    check_out: bool = True,
) -> None:
    """
    Prints out which pipeline files have been saved.
    """
    def check(filename: str) -> str:
        return "SAVED" if Path(filename).is_file() else "MISSING"
    
    cam_a, cam_b = unit_camsID
    pipeline_files = {
        "Simulation Files": [
            (f"Simulated Sky {cam_a.upper()}", filenames['SIM_SKY'][0]),
            (f"Simulated Sky {cam_b.upper()}", filenames['SIM_SKY'][1]),
            ("Simulated Sky (composed)", filenames['COMP_SIMSKY']),
        ],
        "Residuals Files": [
            (f"Residual Sky {cam_a.upper()}", filenames['IROS_RES'][0]),
            (f"Residual Sky {cam_b.upper()}", filenames['IROS_RES'][1]),
            ("Residual Sky (composed)", filenames['COMP_IROSRES']),
        ],
        "IROS Output Files": [
            (f"Output Sky {cam_a.upper()}", filenames['OUT_SKY'][0]),
            (f"Output Sky {cam_b.upper()}", filenames['OUT_SKY'][1]),
            ("Output Sky (composed)", filenames['COMP_OUTSKY']),
        ],
        "Databases Files": [
            ("Sources Database", filenames['SRCS_DB']),
            (f"File Region {cam_a.upper()}", filenames['OUT_REG'][0]),
            (f"File Region {cam_b.upper()}", filenames['OUT_REG'][1]),
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

    return


# end