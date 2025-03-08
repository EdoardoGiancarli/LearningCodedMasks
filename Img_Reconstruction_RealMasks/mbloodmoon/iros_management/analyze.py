"""
IROS output data management and computation.
"""

from typing import Literal
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from mbloodmoon.io import SimulationDataLoader
from mbloodmoon.mask import CodedMaskCamera

from mbloodmoon.images import _shift
from mbloodmoon.images import argmax
from mbloodmoon.mask import count
from mbloodmoon.mask import decode
from mbloodmoon.optim import iros


def perform_iros(
    camerasID: tuple[str],
    camera: CodedMaskCamera,
    sdl_camA: SimulationDataLoader,
    sdl_camB: SimulationDataLoader,
    max_iterations: int = 25,
    snr_threshold: int | float = 5,
    dataset: Literal["detected", "reconstructed"] = "reconstructed",
) -> tuple[dict, tuple[np.array, np.array]]:
    """Runs IROS loop and stores output."""

    def init_log() -> dict:
        """Initializes the log dict structure."""
        init_keys = {
            "shiftx": [], "shifty": [], "fluence": [],
            "snr": [], "obs_counts": [], "sub_counts": [],
        }
        return {camera: deepcopy(init_keys) for camera in camerasID}
    
    def store_output(
        rec_source: tuple[tuple, tuple],
        obs_counts: tuple[float, float],
        sub_counts: tuple[float, float],
    ) -> None:
        """Stores sources info into log."""
        keys = log_output[camerasID[0]].keys()
        for idx, camera in enumerate(log_output.keys()):
            params = [*rec_source[idx], obs_counts[idx], sub_counts[idx]]
            for key, p in zip(keys, params):
                log_output[camera][key].append(p)
    
    def data_to_array(log) -> dict:
        """Converts the log lists in arrays."""
        keys = log[camerasID[0]].keys()
        for camera in list(log.keys()):
            for key in keys:
                if not isinstance(log[camera][key], np.ndarray):
                    log[camera][key] = np.asarray(log[camera][key])
        return log
    
    log_output = init_log()
    detectors = tuple(count(camera, sdl.data)[0] for sdl in [sdl_camA, sdl_camB])
    skies = tuple(decode(camera, d) for d in detectors)
    skies_max = [tuple(np.max(sky) for sky in skies)]
    skies = [skies]

    print("## Looping around the FOV...")
    loop = iros(
        camera=camera,
        sdl_cam1a=sdl_camA,
        sdl_cam1b=sdl_camB,
        max_iterations=max_iterations,
        snr_threshold=snr_threshold,
        dataset=dataset,
    )

    for sources, residuals in tqdm(loop):
        skies.append(residuals)
        skies_max.append(tuple(np.max(r) for r in residuals))
        obs_counts = skies_max[0]
        sub_counts = tuple(s.max() - r[*argmax(s)] for s, r in zip(skies[0], skies[1]))
        skies.pop(0); skies_max.pop(0)
        store_output(sources, obs_counts, sub_counts)

    return data_to_array(log_output), residuals








# end