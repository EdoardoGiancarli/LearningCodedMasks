"""
Script for WISEMAN configuration files editing/generation.
"""

from camgeom.simaker import config_wmsim
from camgeom.types import SimType


if __name__ == '__main__':

    basepath: str = '/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding'
    mask_fpath: str = f'{basepath}/IROS_Data/Simulations/mask_NTHT_20260129_CORRECTED.fits'

    # basepath: str = '/mnt/d/PhD_AASS/Coding/Images_fits'
    # mask_fpath: str = f'{basepath}/mask_NTHT_20260129_CORRECTED.fits'
    
    sims_parent: str = f'{basepath}/cam_geom_wmsims'

    simtypes: list[SimType] = []

    # ---------   MASK   ---------
    # - ROTATIONS
    vals: list[float] = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]  # [arcmin]
    simtypes += [
        SimType('mask', 'rots', 'y', (val, 'arcmin')) for val in vals
    ]
    simtypes += [
        SimType('mask', 'rots', 'z', (val, 'arcmin')) for val in vals
    ]
    # - PLATE WARPING
    vals += [-v for v in vals]
    simtypes += [
        SimType('mask', 'warps', 'y', (val, 'arcmin'), mask_fpath=mask_fpath) for val in vals
    ]
    # - PLATE DEFORMATIONS
    get_mname = lambda val: f"mask_050_1040x17_20260129_ELXDIM_250um_for_wiseman_disp{'m' if val < 0 else 'p'}{f'0{str(int(abs(val)))}'[-3:]}.fits"
    vals = [-25.0, -50.0, -75.0, -100.0, 25.0, 50.0, 75.0, 100.0]
    simtypes += [
        SimType('mask', 'expans', 'x', (val, 'micron'), mask_fpath=get_mname(val)) for val in vals
    ]
    # ---------   DETECTOR PLANE   ---------
    # - DEFORMATIONS
    vals = [25.0, 50.0, 75.0, 100.0]  # [micron]
    simtypes += [
        *[SimType('sdd', 'plane_deforms', 'x', (val, 'micron'), sddplane_deform='external') for val in vals],
        *[SimType('sdd', 'plane_deforms', 'x', (val, 'micron'), sddplane_deform='internal') for val in vals],
        *[SimType('sdd', 'plane_deforms', 'x', (val, 'micron'), sddplane_deform='slide') for val in vals],
        *[SimType('sdd', 'plane_deforms', 'x', (val, 'micron'), sddplane_deform='trapez') for val in vals],
    ]
    # - IN-PLANE SDD ROTATIONS
    vals = [1.0, 2.0, 3.0, 4.0]  # [arcmin]
    simtypes += [
        SimType('sdd', 'plane_inrots', 'z', (val, 'arcmin')) for val in vals
    ]

    config_wmsim(sims_parent, simtypes)


# end