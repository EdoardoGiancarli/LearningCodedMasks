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
from typing import Any, Callable, Iterator

from astropy.io import fits


# ax map mirroring WM configuration file container idxs
AXIS_DMAP: dict[str, int] = {
    'x': 0,
    'y': 1,
    'z': 2,
}
# map for coded-mask plate half-size projection for warping; if warp along 'x',
# the half Y-ax mask size must be projected (opposite for warp along 'y') 
HALFMASK_PROJ_WARP_DMAP = {
    'x': 'Y',
    'y': 'X',
}
# map for single-SDD rotation directions along given plane axis for detector plane
# warping; +1 is relative to an anti-clockwise rotation, -1 to a clockwise one
SDD_PLANE_WARP_DMAP: dict[str, dict[str, int]] = {
    'x': {'00': -1, '10': -1, '01': 1, '11': 1},
    'y': {'00': 1, '10': -1, '01': 1, '11': -1},
    'z': {'00': 1, '10': -1, '01': -1, '11': 1},
}

@dataclass(frozen=True)
class SimType:
    """Camera geometry simulation type to be performed by WISEMAN."""
    level: str
    stype: str
    axis: str
    value: tuple[float, str]
    sddID: str = '00'

    def __post_init__(self) -> None:
        avail_units = ('deg', 'arcmin', 'mm', 'micron')
        if self.value[-1] not in avail_units:
            raise ValueError(f"Invalid effect unit '{self.value[-1]}', must be in {avail_units}")

# -----------   I/O operators   -----------

def load_json(filepath: str | Path) -> dict[str, Any]:
    with open(filepath, 'r') as f:
        dmap = json.load(f)
    return dmap

def save_json(data: Any, filepath: str | Path, mode: str = 'w') -> None:
    with open(filepath, mode) as f:
        json.dump(data, f, indent=4)
    return

# -----------   data operators   -----------

def arcmin2deg(val: float, unit: str) -> float:
    avail = ('deg', 'arcmin')
    if unit not in avail:
        raise ValueError(f"Invalid unit '{unit}', must be in {avail}")
    return val if unit == 'deg' else val / 60

def micron2mm(val: float, unit: str) -> float:
    avail = ('mm', 'micron')
    if unit not in avail:
        raise ValueError(f"Invalid unit '{unit}', must be in {avail}")
    return val if unit == 'mm' else 1e-3 * val

def update_maskOff(data: dict[str, Any], idx: int, val: float) -> None:
    """Adds given value offset to mask pos along specified axis."""
    data['mask']['maskOff'][idx] += val
    return

def update_maskRot(data: dict[str, Any], idx: int, val: float) -> None:
    """Overwrites given value to mask rot angle along specified axis."""
    data['mask']['maskRot'][idx] = val
    return

def update_detPos(data: dict[str, Any], idx: int, val: float, sddID: str) -> None:
    """Adds given value offset to SDD pos along specified axis."""
    data['focalplane']['detPos'][sddID][idx] += val
    return

def update_detRot(data: dict[str, Any], idx: int, val: float, sddID: str) -> None:
    """Overwrites given value to SDD rot angle along specified axis."""
    data['focalplane']['detRot'][sddID][idx] = val
    return

# -----------   WM sim files configuration editors   -----------

ConfigEditor = Callable[[dict, SimType], Iterator[tuple[str, dict]]]

def update_entry(data: dict, keypath: str, value: Any, sep: str = '.') -> None:
    """Edits input dict entry defined by `keypath` with given `value`, in-place."""
    keys = keypath.split(sep)
    curr = data
    for k in keys[:-1]:
        curr = curr[k]
    curr[keys[-1]] = value
    return

def edit_config_for_mask_rots(data: dict[str, Any], st: SimType) -> Iterator[tuple[str, dict]]:
    """Edits baseline camera parameters data to account for coded-mask rotations."""
    if st.axis not in AXIS_DMAP:
        raise ValueError(f"Invalid axis '{st.axis}' for mask rotations.")
    val = arcmin2deg(*st.value)
    update_maskRot(data, AXIS_DMAP[st.axis], val)
    yield "", data

def edit_config_for_mask_warps(data: dict[str, Any], st: SimType, ext: int = 2) -> Iterator[tuple[str, dict]]:
    """
    Edits baseline camera parameters data to account for coded-mask plate warping.

    ## Notes
        * Because actual coded-mask warping cannot be currently simulated with WISEMAN, we approximate
          this effect by joining two simulation with opposite mask rotations along the given axis in post.
    """
    if st.axis not in ('x', 'y'):
        raise ValueError(f"Invalid axis '{st.axis}' for coded-mask plate warping (only planar).")

    val = arcmin2deg(*st.value)

    mask_ax2proj = HALFMASK_PROJ_WARP_DMAP[st.axis].upper()
    header = dict(fits.getheader(data['mask']['maskPatternFile'], ext=ext))
    half_mask = (header[f'MAX{mask_ax2proj}'] - header[f'MIN{mask_ax2proj}']) / 2
    zOff = half_mask * math.sin(math.radians(val))

    neg_rot = deepcopy(data)
    update_maskRot(neg_rot, AXIS_DMAP[st.axis], -abs(val))
    update_maskOff(neg_rot, AXIS_DMAP['z'], zOff)
    yield f"neg_{abs(st.value[0])}{st.value[1]}", neg_rot    # NOTE: this refers to the dir name

    pos_rot = deepcopy(data)
    update_maskRot(pos_rot, AXIS_DMAP[st.axis], abs(val))
    update_maskOff(pos_rot, AXIS_DMAP['z'], zOff)
    yield f"pos_{abs(st.value[0])}{st.value[1]}", pos_rot

def edit_config_for_sdd_rots(data: dict[str, Any], st: SimType) -> Iterator[tuple[str, dict]]:
    """Edits baseline camera parameters data to account for SDD rotations."""
    if st.axis not in AXIS_DMAP:
        raise ValueError(f"Invalid axis '{st.axis}' for SDD rotations.")
    val = arcmin2deg(*st.value)
    update_detRot(data, AXIS_DMAP[st.axis], val, st.sddID)
    yield "", data

def edit_config_for_sdd_shifts(data: dict[str, Any], st: SimType) -> Iterator[tuple[str, dict]]:
    """Edits baseline camera parameters data to account for SDD shifts."""
    if st.axis not in AXIS_DMAP:
        raise ValueError(f"Invalid axis '{st.axis}' for SDD shifts on detector plane.")
    val = micron2mm(*st.value)
    update_detPos(data, AXIS_DMAP[st.axis], val, st.sddID)
    yield "", data

def edit_config_for_sdd_plane_expans(data: dict[str, Any], st: SimType) -> Iterator[tuple[str, dict]]:
    """Edits baseline camera parameters data to account for detector plane expansions."""
    if st.axis not in ('radial',):
        raise ValueError(
            f"Invalid expansion '{st.axis}' for detector plane (currently, only 'radial' supported)."
        )

    val = micron2mm(*st.value)
    for pos in data['focalplane']['detPos'].values():
        r_base = math.hypot(pos[0], pos[1])
        if r_base == 0:
            print('UserInfo: SDD aligned with detector plane axis (use sdd shifts in this case)')
            continue
        scale = (r_base + val) / r_base
        pos[0] *= scale
        pos[1] *= scale
        
    yield "", data

def edit_config_for_sdd_warps(data: dict[str, Any], st: SimType) -> Iterator[tuple[str, dict]]:
    """Edits baseline camera parameters data to account for detector plane warping, symmetric wrt plane centre."""
    if st.axis not in AXIS_DMAP:
        raise ValueError(f"Invalid axis '{st.axis}' for detector plane warps.")

    idx = AXIS_DMAP[st.axis]
    val = arcmin2deg(*st.value)
    for sddID, w in SDD_PLANE_WARP_DMAP[st.axis].items():
        update_detRot(data, idx, w * val, sddID)

    yield "", data

# -----------   Interface for WM sim editing   -----------

def NotAvailable(*args, **kwargs):
    raise NotImplementedError(kwargs.get('msg', 'Method not available'))

WMSIM_REGISTRY: dict[str, dict[str, ConfigEditor]] = {
    'mask': {
        'rots': edit_config_for_mask_rots,
        'shifts': partial(NotAvailable, msg='Mask shifts TBD...'),
        'warps': edit_config_for_mask_warps,
        'expans': partial(NotAvailable, msg='Mask plate expansions TBD...'),
    },
    'sdd': {
        'rots': edit_config_for_sdd_rots,
        'shifts': edit_config_for_sdd_shifts,
        'plane_expans': edit_config_for_sdd_plane_expans,
        'plane_warps': edit_config_for_sdd_warps,
    },
}

def get_dst_path(parent: Path, st: SimType) -> Path:
    """Handles target directory for given simulation."""
    single_sdd_op = st.level == 'sdd' and st.stype in ('rots', 'shifts')
    prefix = f"{st.level}{st.sddID}" if single_sdd_op else st.level
    
    root = f"{prefix}_{st.stype}"
    ax = f"{prefix}_{st.axis}_{st.stype}"
    sim = f"{st.axis}_{st.stype}_{st.value[0]}{st.value[1]}"

    return parent / root / ax / sim

def config_wmsim(
    parent: str | Path,
    simtype: SimType | tuple[SimType, ...],
    camfile: str = 'camera_parfile.json',
    srcfile: str = 'sources_instr_parfile.json',
    ref_to: str = 'baseline/baseline',
) -> None:
    """
    Generates directories with config files for camera geometry sims performed with WISEMAN.
    """
    parent = Path(parent)
    simtype_ = (simtype,) if isinstance(simtype, SimType) else simtype

    basepath = parent / ref_to
    data = load_json(basepath / camfile)
    src_filepath = basepath / srcfile

    for st in simtype_:
        try:
            edit_fn = WMSIM_REGISTRY[st.level][st.stype]
        except KeyError:
            print(f"KeyError: missing config for level '{st.level}', stype '{st.stype}'")
            continue

        data_ = deepcopy(data)
        base_save_to = get_dst_path(parent, st)

        try:
            for suffix, mod_data in edit_fn(data_, st):
                save_to = base_save_to / suffix if suffix else base_save_to
                save_to.mkdir(parents=True, exist_ok=True)

                save_json(mod_data, save_to / camfile)
                shutil.copyfile(src_filepath, save_to / srcfile)

        except Exception as e:
            print(f"{type(e).__name__} on {st.level} {st.stype}: {e}")
            continue

    return




if __name__ == '__main__':

    simtype = (
        # SimType('mask', 'rots', 'x', (-1, 'deg')),
        # SimType('mask', 'rots', 'y', (6, 'arcmin')),
        # SimType('mask', 'rots', 'z', (-2, 'arcmin')),

        # SimType('mask', 'warps', 'x', (6, 'arcmin')),
        SimType('mask', 'warps', 'y', (2, 'arcmin')),
        SimType('mask', 'warps', 'y', (-2, 'arcmin')),
        # SimType('mask', 'warps', 'y', (-0.1, 'deg')),

        # SimType('mask', 'shifts', 'z', (1, 'mm')),
        # SimType('mask', 'expans', 'z', (1, 'mm')),

        # SimType('sdd', 'shifts', 'x', (100, 'micron')),
        # SimType('sdd', 'shifts', 'y', (1, 'mm')),
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