"""
Script for WISEMAN configuration files editing/generation.
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Literal


# ax map mirroring WM configuration file container idxs
AXIS_DMAP: dict[str, int] = {
    'x': 0,
    'y': 1,
    'z': 2,
}
# map for single-SDD rotation directions along given plane axis for detector plane
# warpings; +1 is relative to an anti-clockwise rotation, -1 to a clockwise one
SDD_PLANE_WARP_DMAP: dict[str, dict[str, int]] = {
    'x': {'00': -1, '10': -1, '01': 1, '11': 1},
    'y': {'00': 1, '10': -1, '01': 1, '11': -1},
    'z': {'00': 1, '10': -1, '01': -1, '11': 1},
}


@dataclass
class SimType:
    """Camera geometry simulation type to be performed by WISEMAN."""
    level: str
    stype: str
    axis: str
    value: tuple[float, str]
    sddID: str = '00'


def load_json(filepath: str | Path) -> dict[str, Any]:
    with open(filepath, 'r') as f:
        dmap = json.load(f)
    return dmap

def save_json(data: Any, filepath: str | Path, mode: str = 'w') -> None:
    with open(filepath, mode) as f:
        json.dump(data, f, indent=4)
    return

def update_entry(data: dict, keypath: str, value: Any, sep: str = '.') -> None:
    """Edits input dict entry defined by `keypath` with given `value`, in-place."""
    keys = keypath.split(sep)
    curr = data
    for k in keys[:-1]:
        curr = curr[k]
    curr[keys[-1]] = value
    return

def edit_config_for_mask_rots(data: dict[str, Any], ax: Literal['x', 'y', 'z'], effect: float) -> None:
    """Edits baseline camera parameters data to account for mask rotations."""
    if ax not in AXIS_DMAP.keys():
        raise ValueError(f"Invalid axis '{ax}' for mask rotations.")
    data['mask']['maskRot'][AXIS_DMAP[ax]] = effect
    return

def edit_config_for_sdd_rots(data: dict[str, Any], ax: Literal['x', 'y', 'z'], effect: float, sddID: str = '00') -> None:
    """Edits baseline camera parameters data to account for SDD rotations."""
    if ax not in AXIS_DMAP.keys():
        raise ValueError(f"Invalid axis '{ax}' for SDD rotations.")
    data['focalplane']['detRot'][sddID][AXIS_DMAP[ax]] = effect
    return

def edit_config_for_sdd_shifts(data: dict[str, Any], ax: Literal['x', 'y', 'z'], effect: float, sddID: str = '00') -> None:
    """Edits baseline camera parameters data to account for SDD shifts."""
    if ax not in AXIS_DMAP.keys():
        raise ValueError(f"Invalid axis '{ax}' for SDD shifts on detector plane.")
    data['focalplane']['detPos'][sddID][AXIS_DMAP[ax]] += effect
    return

def edit_config_for_sdd_plane_expans(data: dict[str, Any], ax: Literal['radial'], effect: float) -> None:
    """Edits baseline camera parameters data to account for detector plane expansions."""
    match ax:
        case 'radial':
            detPos = data['focalplane']['detPos']
            for pos in detPos.values():
                r_base = math.hypot(pos[0], pos[1])
                if r_base == 0:
                    print('UserInfo: SDD aligned with detector plane axis (use sdd shifts in this case)')
                    continue

                r_new = r_base + effect
                scale = r_new / r_base
                pos[0] *= scale
                pos[1] *= scale
        case _:
            raise ValueError(f"Invalid expansion '{ax}' for detector plane.")
    return

def edit_config_for_sdd_warps(data: dict[str, Any], ax: Literal['x', 'y', 'z'], effect: float) -> None:
    """Edits baseline camera parameters data to account for detector plane warps, symmetric wrt plane centre."""
    if ax not in AXIS_DMAP.keys():
        raise ValueError(f"Invalid axis '{ax}' for detector plane warps.")
    idx = AXIS_DMAP[ax]
    for sddID, w in SDD_PLANE_WARP_DMAP[ax].items():
        data['focalplane']['detRot'][sddID][idx] = w * effect
    return

def get_dst_path(parent: Path, st: SimType):
    """Handles target directory for given simulation."""
    single_sdd_op = st.level == 'sdd' and st.stype in ('rots', 'shifts')
    prefix = f"{st.level}{st.sddID}" if single_sdd_op else st.level
    
    root = f"{prefix}_{st.stype}"
    ax = f"{prefix}_{st.axis}_{st.stype}"
    sim = f"{st.axis}_{st.stype}_{st.value[0]}{st.value[1]}"
    
    save_to = parent / root / ax / sim
    save_to.mkdir(parents=True, exist_ok=True)
    return save_to


def NotAvailable(*args, **kwargs):
    raise NotImplementedError(kwargs.get('msg', 'Method not available'))

WMSIM_REGISTRY: dict[str, dict[str, Callable]] = {
    'mask': {
        'rots': edit_config_for_mask_rots,
        'shifts': partial(NotAvailable, msg='Mask shifts TBD...'),
        'warps': partial(NotAvailable, msg='Mask warps TBD...'),
    },
    'sdd': {
        'rots': edit_config_for_sdd_rots,
        'shifts': edit_config_for_sdd_shifts,
        'plane_expans': edit_config_for_sdd_plane_expans,
        'plane_warps': edit_config_for_sdd_warps,
    },
}


def config_wmsim(
    parent: str | Path,
    simtype: SimType | tuple[SimType, ...],
    camfile: str = 'camera_parfile.json',
    srcfile: str = 'sources_instr_parfile.json',
    ref_to: str = 'baseline',
) -> None:
    """
    Generates directories with config files for camera geometry sims performed with WISEMAN.
    """
    parent = Path(parent)
    simtype_ = (simtype,) if isinstance(simtype, SimType) else simtype

    basepath = parent / 'baseline' / ref_to
    data = load_json(basepath / camfile)
    src_filepath = basepath / srcfile

    for st in simtype_:
        data_ = deepcopy(data)

        try:
            edit_fn = WMSIM_REGISTRY[st.level][st.stype]
        except KeyError:
            print(f"KeyError: missing config for {st.level} {st.stype}")
            continue

        if st.level == 'sdd' and st.stype in ('rots', 'shifts'):
            edit_fn = partial(edit_fn, sddID=st.sddID)

        try:
            edit_fn(data_, st.axis, st.value[0])
        except Exception as e:
            print(f"{type(e).__name__} on {st.level} {st.stype}: {e}")
            continue

        save_to = get_dst_path(parent, st)
        save_json(data_, save_to / camfile)
        shutil.copyfile(src_filepath, save_to / srcfile)

    return




if __name__ == '__main__':

    simtype = (
        SimType('mask', 'rots', 'x', (1, 'deg')),
        # SimType('mask', 'rots', 'y', (6 / 60, 'arcmin')),
        # SimType('mask', 'rots', 'z', (6 / 60, 'arcmin')),

        # SimType('mask', 'shifts', 'z', (1, 'mm')),
        # SimType('mask', 'warps', 'y', (6 / 60, 'arcmin')),

        # SimType('sdd', 'shifts', 'x', (10, 'mm')),
        # SimType('sdd', 'shifts', 'y', (10, 'mm')),
        # SimType('sdd', 'plane_expans', 'radial', (10, 'mm')),
        # SimType('sdd', 'plane_expans', 'radial', (-10, 'mm')),
        # SimType('sdd', 'rots', 'x', (6 / 60, 'arcmin')),
        # SimType('sdd', 'rots', 'y', (6 / 60, 'arcmin')),
        # SimType('sdd', 'plane_warps', 'x', (6 / 60, 'arcmin')),
        # SimType('sdd', 'plane_warps', 'y', (6 / 60, 'arcmin')),
        # SimType('sdd', 'plane_warps', 'y', (-6 / 60, 'arcmin')),
    )
    config_wmsim("/mnt/d/PhD_AASS/Coding/Images_fits/cam_geom_wmsims", simtype)


# end