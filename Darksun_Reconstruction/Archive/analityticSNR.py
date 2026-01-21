"""
Source significance analytical equation from Evangelista et al., 2025 (CFR also with Skinner et al, 2008).
"""

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from bloodmoon.coords import angle2shift
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.optim import _detector_footprint_cached
from bloodmoon.optim import _mask_pattern_projection


@lru_cache
def pattern_projection(
    camera: CodedMaskCamera,
    shift_x: float,
    shift_y: float,
    vignetting: bool = True,
) -> NDArray:
    """Projects the mask pattern on the detector."""
    # mask pattern projection WTO detector sp. res.
    sg = _mask_pattern_projection(
        camera, shift_x, shift_y, vignetting, False,
    )
    # extract detector
    i_min, i_max, j_min, j_max = _detector_footprint_cached(camera)
    detector = sg[i_min:i_max, j_min:j_max]
    detector *= camera.bulk
    return detector

def get_effective_area(
    camera: CodedMaskCamera,
    shift_x: float,
    shift_y: float,
    vignetting: bool = True,
) -> float:
    """Computes source effective area from detector in [cm^2]."""
    # pixel area in [cm^2]
    pixel_area = (
        1e-2 * camera.specs.mask_deltax * camera.specs.mask_deltay / np.prod(camera.upscale_f)
    )
    # correction factor for SDD QE, MLI and filter photons absorption @ 8keV
    # https://github.com/yuri-evangelista/CodedMasks/blob/main/mask_050_1040x17/Effective_area_and_Sens_2D.ipynb
    #           dead layer *   QE  * 25um Be * 300nm Al * 12.5um Kapton
    qe_factor =   0.974    * 0.999 * 0.99527 *  0.99614 *   0.98945
    # compute mask projection on camera detector
    detector = pattern_projection(camera, shift_x, shift_y, vignetting)
    return detector.sum() * pixel_area * qe_factor

def get_effective_open_fraction(
    camera: CodedMaskCamera,
    shift_x: float,
    shift_y: float,
    vignetting: bool = True,
) -> float:
    """Computes projected mask effective open fraction on detector."""
    detector = pattern_projection(camera, shift_x, shift_y, vignetting)
    return detector.sum() / detector.size


def source_significance(
    camera: CodedMaskCamera,
    flux: float,
    exposure: float,
    angle_x: float,
    angle_y: float,
    vignetting: bool = True,
) -> float:
    """
    Computes a source significance taking into account its position
    and the sky background counts per unit area on the detector.

    The output value is a upper limit estimation, since the source
    is assumed to be isolated in the sky-field. This means that the
    SNR estimate does not include the coding noise from other sources.

    Flux is in [Crab], exposure in [s], angles in [deg].
    """
    #TODO: THIS IS NOT CORRECT, THERE IS SOME MISTAKE SOMEWHERE

    # setup (bkg_fluence and crab cts/area refers to LEM-X single coded-mask camera)
    bkg_fluence: float = 6.3799 * exposure          # bkg counts per detector unit area [ph/cm2/s]
    crab: float = 2.5737                            # [ph/cm2/Crab/s]
    source_fluence: float = flux * crab * exposure  # [ph/cm2]
    # open_fraction and coding power are [adim]
    open_fraction: float = camera.mask.sum() / camera.mask.size
    coding_power: float = 0.85
    
    # retrieve source effective area and effective open fraction on detector
    shift_x, shift_y = map(
        lambda x: angle2shift(camera, x),
        (angle_x, angle_y),
    )
    off_axis_area: float = get_effective_area(camera, shift_x, shift_y, vignetting)
    effective_open_fraction: float = get_effective_open_fraction(camera, shift_x, shift_y, vignetting)

    # bkg counts correction for sensitivity overestimation
    # https://github.com/yuri-evangelista/CodedMasks/blob/main/mask_050_1040x17/Effective_area_and_Sens_2D.ipynb
    on_axis_area: float = get_effective_area(camera, 0.0, 0.0, vignetting)
    bkg_fluence_corr: float = (
        bkg_fluence + (1 - off_axis_area / on_axis_area) * source_fluence
    )

    # compute significance
    a: float = coding_power * source_fluence / open_fraction
    b1: float = off_axis_area * (1.0 - effective_open_fraction)
    b2: float = effective_open_fraction * (source_fluence + bkg_fluence_corr) / open_fraction
    return a * np.sqrt(b1 / b2)


# end