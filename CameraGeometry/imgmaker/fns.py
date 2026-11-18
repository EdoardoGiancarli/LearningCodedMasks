"""
Module with helper funcs.
"""

from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd
from tqdm import tqdm

from bloodmoon.mask import CodedMaskCamera
import darksun as ds
from darksun.data import Log, DataLoader, CatalogueLoader

__all__ = [
    'config_psfy_flag',
    'config_savedata_to',
    'get_srcmap_for_unit',
    'extract_catalogue_angular_coords',
    'get_angularcoords_residues',
    'extract_catalogue_fluences',
    'df2TeXtab',
    'open_ImageHDU',
]


def config_psfy_flag(dataset: str) -> bool:
    if dataset not in ['detected', 'reconstructed']:
        raise ValueError('Invalid dataset type.')
    flag: bool = (True if dataset == 'reconstructed' else False)
    return flag

def config_savedata_to() -> str:
    """Generates the path to directory for saving files."""
    paths = [
        '/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/IROS_paper',
        '/mnt/d/PhD_AASS/Coding/Images_fits/IROS_paper',
    ]
    out = next((p for p in paths if Path(p).is_dir()), None)
    if out is None: raise ValueError('A0 ndo sei finit*?')
    return out




class CameraUnitMap(NamedTuple):
    common: NDArray
    unique: NDArray
    idx_a: NDArray
    idx_b: NDArray

def get_srcmap_for_unit(
    srcA: list[str],
    srcB: list[str],
    uniqueIDs: bool = True,
) -> CameraUnitMap:
    """
    Finds common sources in LEM-X Unit IROS reconstruction from detected sources IDs.
    Returns the sorted, unique src IDs that are in both input lists, and the index map.
    """
    srcA_, srcB_ = map(np.array, (srcA, srcB))
    common, idx_a, idx_b = np.intersect1d(
        srcA_, srcB_, assume_unique=uniqueIDs, return_indices=True,
    )
    unique = np.setxor1d(srcA_, srcB_, assume_unique=uniqueIDs)
    return CameraUnitMap(common, unique, idx_a, idx_b)




def extract_catalogue_angular_coords(
    log: Log,
    catalogue: CatalogueLoader,
    sdl: DataLoader,
    camera: CodedMaskCamera,
) -> tuple[NDArray, NDArray]:
    """
    Extracts the catalogue sources angular coords in [deg]
    along the fine and coarse direction respectively.
    """
    angles_x, angles_y = [], []
    loop = tqdm(log.log['ID'])
    for sourceID in loop:
        loop.set_description(f'Analysing {sourceID.upper()}')
        anglex, angley = (
            ds.source_angular_coords(sourceID, catalogue, sdl, camera)
            if sourceID in catalogue.DLdata['ID']
            else (np.nan, np.nan)
        )
        angles_x.append(anglex)
        angles_y.append(angley)

    return map(np.array, (angles_x, angles_y))

def get_angularcoords_residues(
    log: Log,
    catalogue: CatalogueLoader,
    sdl: DataLoader,
    camera: CodedMaskCamera,
) -> tuple[NDArray, NDArray]:
    """
    Computes the angular coords residues between the IROS
    sources and the catalogue sources in [arcmin], along
    the fine and coarse direction respectively.
    """
    angles_x, angles_y = extract_catalogue_angular_coords(log, catalogue, sdl, camera)
    res_x, res_y = (
        60 * (np.array(log.log['angle_x']) - angles_x),
        60 * (np.array(log.log['angle_y']) - angles_y),
    )
    return res_x, res_y

def extract_catalogue_fluences(
    log: Log,
    catalogue: CatalogueLoader,
    sdl: DataLoader,
    camera: CodedMaskCamera,
    verbose: bool = False,
) -> NDArray:
    """
    Extracts the catalogue sources fluences.
    """
    fluences: list[float] = []
    loop = tqdm(log.log['ID'])
    for sourceID in loop:
        loop.set_description(f'Analysing {sourceID.upper()}')
        fluence = (
            ds.source_fluence(sourceID, catalogue, sdl, camera, verbose)
            if sourceID in catalogue.DLdata['ID']
            else np.nan
        )
        fluences.append(fluence)
        
    return np.array(fluences)




def df2TeXtab(
    df: pd.DataFrame,
    adjust_tabfrmt: Callable[[str], str] | None = None,
    save_to: str | Path | None = None,
    overwrite: bool = False,
    **tex_kws: Any,
) -> str:
    """
    Generates a LaTeX table from input dataframe and saves it.
    """
    tab = df.to_latex(index=False, **tex_kws)
    if adjust_tabfrmt is not None:
        tab = adjust_tabfrmt(tab)

    if save_to is not None:
        if Path(save_to).exists() and not overwrite:
            print('Table already saved!')
            return tab
        with open(save_to, "w", encoding="utf-8") as f:
            f.write(tab)
    
    return tab



def open_ImageHDU(filepath: str, ext: int = 1) -> tuple[NDArray, WCS | None]:
    """
    Opens FITS file and extracts the chosen Image.
    Returns the Image HDU and the associated WCS.
    """
    with fits.open(filepath) as hdu:
        img_hdu = hdu[ext]
        img = img_hdu.data
        try:
            wcs = WCS(img_hdu.header)
        except (KeyError, ValueError) as e:
            print(f'Error: {e}. WCS not extractable.')
            wcs = None
    
    return img, wcs


# end