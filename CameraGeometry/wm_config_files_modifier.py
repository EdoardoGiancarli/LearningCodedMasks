"""
Script for WISEMAN configuration files editing/generation.
"""

import json
from pathlib import Path
from typing import Any


def update_entry(data: dict, keypath: str, value: Any, sep: str = '.') -> None:
    """Edits input dict entry defined by `keypath` with given `value`, in-place."""
    keys = keypath.split(sep)
    curr = data
    for k in keys[:-1]:
        curr = curr[k]
    curr[keys[-1]] = value
    return




fname = "camera_parfile.json"
fpath_baseline = f"/mnt/d/PhD_AASS/Coding/Images_fits/cam_geom_wmsims/baseline/baseline/{fname}"

with open(fpath_baseline, 'r') as f:
    dmap = json.load(f)

dmap.keys()




if __name__ == '__main__':

    case = 'mask_rots'
    axis = 'y'
    effect = (0.5, 'arcmin')
    keypath = 'mask.maskRot'
    val = [0, effect[0] / 60, 0]

    case = 'sdd00_rots'
    axis = 'y'
    value = (0.5, 'arcmin')
    keypath = 'focalplane.detPos.00'
    val = [0, effect[0] / 60, 0]

    case = 'sdd00_shifts'
    axis = 'y'
    value = (10, 'um')
    keypath = 'focalplane.detRot.00'
    val = [0, -41.4375 + effect[0], 0]

    make_wmsim_dir()




    case = 'sdd_plane_contrapts'
    axis = 'radial'
    value = (20, 'um')

    case = 'sdd_plane_warps'
    axis = 'y'
    value = (-0.5, 'arcmin')


# end