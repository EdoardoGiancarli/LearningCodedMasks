"""
Module for WISEMAN configuration files editing/generation.
"""

from copy import deepcopy
from functools import partial
from itertools import product
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Iterator

from astropy.io import fits

from .opers import (
    load_json, save_json,
    arcmin2deg, micron2mm,
    update_maskOff, update_maskRot,
    update_detPos, update_detRot,
)
from .types import SimType


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
# map for single-SDD rotation directions along given plane axis for detector
# plane warping; -1/+1 refers to clockwise/anti-clockwise rotation
SDD_PLANE_WARP_DMAP: dict[str, dict[str, int]] = {
    'x': {'00': -1, '10': -1, '01': 1, '11': 1},
    'y': {'00': 1, '10': -1, '01': 1, '11': -1},
    'z': {'00': 1, '10': -1, '01': -1, '11': 1},
}
# map for detector plane deformation cases, with per-axis weights
# [x axis, y axis], and with -1/+1 referring to neg/pos shifts
SDD_PLANE_DEFORMS_DMAP: dict[str, list[dict[str, int]]] = {
    'external': [
        {'00': -1, '10': 1, '01': -1, '11': 1},
        {'00': -1, '10': -1, '01': 1, '11': 1},
    ],
    'internal': [
        {'00': 1, '10': -1, '01': 1, '11': -1},
        {'00': 1, '10': 1, '01': -1, '11': -1},
    ],
    'slide': [
        {'00': -1, '10': -1, '01': 1, '11': 1},
        {'00': 1, '10': -1, '01': 1, '11': -1},
    ],
    'trapez': [
        {'00': -1, '10': 1, '01': 1, '11': -1},
        {'00': 1, '10': -1, '01': -1, '11': 1},
    ],
}


# -----------   WM sim files configuration editors   -----------

ConfigEditor = Callable[[dict, SimType], Iterator[tuple[str, dict]]]

def edit_config_for_mask_rots(data: dict[str, Any], st: SimType) -> Iterator[tuple[str, dict]]:
    """Edits baseline camera parameters data to account for coded-mask rotations."""
    if st.axis not in AXIS_DMAP:
        raise ValueError(f"Invalid axis '{st.axis}' for mask rotations.")
    val = arcmin2deg(*st.value)
    update_maskRot(data, AXIS_DMAP[st.axis], val)
    yield "", data

def edit_config_for_mask_warps(
    data: dict[str, Any],
    st: SimType,
    ext: int = 2,
) -> Iterator[tuple[str, dict]]:
    """
    Edits baseline camera parameters data to account for coded-mask plate warping.

    ## Notes
        * Because actual coded-mask warping cannot be currently simulated with WISEMAN, we approximate
          this effect by joining two simulation with opposite mask rotations along the given axis in post.
    """
    if st.axis not in ('x', 'y'):
        raise ValueError(f"Invalid axis '{st.axis}' for coded-mask plate warping (only planar).")

    fpath = st.mask_fpath or data['mask']['maskPatternFile']
    header = dict(fits.getheader(fpath, ext=ext))

    val = arcmin2deg(*st.value)
    mask_ax2proj = HALFMASK_PROJ_WARP_DMAP[st.axis].upper()
    half_mask = (header[f'MAX{mask_ax2proj}'] - header[f'MIN{mask_ax2proj}']) / 2
    zOff = half_mask * math.sin(math.radians(val))

    neg_rot = deepcopy(data)
    update_maskRot(neg_rot, AXIS_DMAP[st.axis], -abs(val))
    update_maskOff(neg_rot, AXIS_DMAP['z'], zOff)
    yield f"neg_{abs(st.value[0])}{st.value[1]}", neg_rot

    pos_rot = deepcopy(data)
    update_maskRot(pos_rot, AXIS_DMAP[st.axis], abs(val))
    update_maskOff(pos_rot, AXIS_DMAP['z'], zOff)
    yield f"pos_{abs(st.value[0])}{st.value[1]}", pos_rot

def edit_config_for_mask_expans(data: dict[str, Any], st: SimType) -> Iterator[tuple[str, dict]]:
    """
    Edits baseline camera parameters data to account for coded-mask plate expansion.

    ## Notes
        * In this case the mask pattern file within `data` is updated.
    """
    if st.axis not in ('x', 'y'):
        raise ValueError(f"Invalid axis '{st.axis}' for coded-mask plate expansion (only planar).")
    if not st.mask_fpath:
        raise ValueError(f"Missing deformed coded-mask pattern in SimType obj.")

    in_path, mask_path = map(Path, (data['mask']['maskPatternFile'], st.mask_fpath))
    out_path = mask_path if mask_path.is_absolute() else in_path.parent / mask_path
    data['mask']['maskPatternFile'] = str(out_path)

    yield "", data

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

def edit_config_for_sdd_plane_deforms(data: dict[str, Any], st: SimType) -> Iterator[tuple[str, dict]]:
    """Edits baseline camera parameters data to account for detector plane deformation."""
    if st.axis not in ('x', 'y'):
        raise ValueError(f"Invalid axis '{st.axis}' for detector plane deformation (only planar).")
    if st.sddplane_deform not in SDD_PLANE_DEFORMS_DMAP:
        raise ValueError(f"Invalid detector plane deformation type '{st.sddplane_deform}'.")

    idx = AXIS_DMAP[st.axis]
    val = micron2mm(*st.value)
    for sddID, w in SDD_PLANE_DEFORMS_DMAP[st.sddplane_deform][idx].items():
        update_detPos(data, idx, w * val, sddID)
        
    yield f"{st.sddplane_deform}_{st.value[0]}{st.value[1]}", data

def edit_config_for_sdd_plane_warps(data: dict[str, Any], st: SimType) -> Iterator[tuple[str, dict]]:
    """Edits baseline camera parameters data to account for detector plane warping, symmetric wrt plane centre."""
    if st.axis not in AXIS_DMAP:
        raise ValueError(f"Invalid axis '{st.axis}' for detector plane warps.")

    idx = AXIS_DMAP[st.axis]
    val = arcmin2deg(*st.value)
    for sddID, w in SDD_PLANE_WARP_DMAP[st.axis].items():
        update_detRot(data, idx, w * val, sddID)

    yield "", data

def edit_config_for_sdd_plane_inrots(
    data: dict[str, Any],
    st: SimType,
) -> Iterator[tuple[str, dict]]:
    """
    Edits baseline camera parameters data to account for all-SDD rotations
    wrt selected axis (e.g., for SDD in-plane rotations select 'z' axis).

    ## Notes:
        * `rm_sdd_inrots` refers to SDD rot combinations, e.g. express as
          `['1,1,1,1', '1,1,1,-1', ...]`, with -1/+1 referring to a
          clockwise/anti-clockwise rotation
    """
    if st.axis not in AXIS_DMAP:
        raise ValueError(f"Invalid axis '{st.axis}' for all-SDD in-plane-rotations.")

    idx = AXIS_DMAP[st.axis]
    val = arcmin2deg(*st.value)
    rm_sdd_inrots_ = [tuple(map(int, e.split(','))) for e in st.rm_sdd_inrots]

    sddIDs = ('00', '10', '01', '11')
    states = set(product([-1, 1], repeat=len(sddIDs)))
    for state in states.difference(rm_sdd_inrots_):
        data_ = deepcopy(data)
        for sddID, w in zip(sddIDs, state):
            update_detRot(data_, idx, w * val, sddID)

        yield f"state_{'.'.join(map(str, state))}_{st.value[0]}{st.value[1]}", data_


# -----------   Interface for WM sim editing   -----------

def NotAvailable(*args, **kwargs):
    raise NotImplementedError(kwargs.get('msg', 'Method not available'))

WMSIM_REGISTRY: dict[str, dict[str, ConfigEditor]] = {
    'mask': {
        'rots': edit_config_for_mask_rots,
        'shifts': partial(NotAvailable, msg='Mask shifts TBD...'),
        'warps': edit_config_for_mask_warps,
        'expans': edit_config_for_mask_expans,
    },
    'sdd': {
        'rots': edit_config_for_sdd_rots,
        'shifts': edit_config_for_sdd_shifts,
        'plane_expans': edit_config_for_sdd_plane_expans,
        'plane_deforms': edit_config_for_sdd_plane_deforms,
        'plane_inrots': edit_config_for_sdd_plane_inrots,
        'plane_warps': edit_config_for_sdd_plane_warps,
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


# end