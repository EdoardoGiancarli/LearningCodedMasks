"""
Module for `camgeom` helper objs.
"""

import json
from pathlib import Path
from typing import Any


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


def update_entry(data: dict, keypath: str, value: Any, sep: str = '.') -> None:
    """Edits input dict entry defined by `keypath` with given `value`, in-place."""
    keys = keypath.split(sep)
    curr = data
    for k in keys[:-1]:
        curr = curr[k]
    curr[keys[-1]] = value
    return


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


# end