"""
Module with corrected source shadowgram template normalisation, applied before array processing.
"""

from typing import Callable

import numpy as np
import numpy.typing as npt

from bloodmoon.optim import _detector_footprint_cached, _shift_mask_pattern, _process_mask_pattern
from bloodmoon.mask import CodedMaskCamera


def _extract_detector(
    camera: CodedMaskCamera,
    shadowgram: npt.NDArray,
    normalise: bool = False,
) -> npt.NDArray:
    """
    Extracts the detector image from the mask pattern projection on the detector plane.
    """
    i_min, i_max, j_min, j_max = _detector_footprint_cached(camera)
    detector = shadowgram[i_min:i_max, j_min:j_max]
    detector *= camera.bulk
    if normalise:
        detector /= np.sum(detector)
    return detector


def model_shadowgram(
    camera: CodedMaskCamera,
    shift_x: float,
    shift_y: float,
    vignetting: bool | Callable[[CodedMaskCamera, npt.NDArray, float, float], npt.NDArray] = True,
    psfy: bool | Callable[[CodedMaskCamera, npt.NDArray], npt.NDArray] = True,
) -> npt.NDArray:
    """
    Generates a normalized shadowgram for a point source.

    The model may feature:
    - Mask pattern projection
    - Vignetting effects
    - PSF convolution over y axis

    Args:
        camera: CodedMaskCamera instance containing all geometric parameters
        shift_x: Source position x-coordinate in sky-shift space (mm)
        shift_y: Source position y-coordinate in sky-shift space (mm)
        vignetting: simulates vignetting effects
        psfy: simulates detector reconstruction effects

    Returns:
        2D array representing the modeled detector image from the source

    Notes:
        * Results are normalized, i.e. sums up to one.
    """
    for key, val in {'vignetting': vignetting, 'psfy': psfy}.items():
        if not (isinstance(val, bool) or callable(val)):
            raise ValueError(f"'{key}' must be bool or Callable, got {type(val)} instead.")
    
    # shift camera mask pattern wrt source local-frame coords
    mask_shifted = _shift_mask_pattern(camera, shift_x, shift_y)
    norm_factor = _extract_detector(camera, mask_shifted).sum()
    # apply instrumental effects
    mask_projected = _process_mask_pattern(
        camera, mask_shifted, shift_x, shift_y, vignetting, psfy,
    )
    # extract source detector image
    #   - the shadowgram must be normalised wrt the shifted pattern
    #     without instr. effects applied on it
    #   - after applying the instr. effects a fraction of the photons
    #     is lost and/or dispersed, and the total sum is reduced, thus
    #     increasing the array px intensities
    detector = _extract_detector(camera, mask_projected)
    detector /= norm_factor

    return detector


# end