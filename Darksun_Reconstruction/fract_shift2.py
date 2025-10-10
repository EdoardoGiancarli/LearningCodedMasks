"""
Testing sources PSF with fractional array shift and vignetting.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift as ndshift
from scipy.signal import convolve

from bloodmoon.coords import shift2pos, pos2shift
from bloodmoon.mask import _detector_footprint, CodedMaskCamera
from bloodmoon.optim import _wfm_psfy_kernel_cached


def _fshift(
    arr: NDArray,
    shifty: int | float,
    shiftx: int | float,
):
    """
    Applies fractional shift to input array.

    The fractional shifting value is eroded from the end-points
    wrt the shift verse and added to the front (for both rows/cols).
    """
    arr_ = ndshift(
        arr, (shifty, shiftx), output='float', order=1,
        mode='grid-constant', cval=0.0, prefilter=True,
    )
    return arr_


def _erosion(
    arr: NDArray,
    step: float,
    cut: float,
) -> NDArray:
    """
    2D matrix erosion for simulating finite thickness effect in shadow projections.
    It takes a mask array and "thins" the mask elements across the columns' direction.
    """
    # number of bins to cut
    ncuts = int(cut / step)
    arr_mask = (arr > 0) & (_fshift(arr, 0, ncuts) > 0)
    cutted = arr * arr_mask if ncuts else arr

    # array indexes to be fractionally reduced:
    #   - the bin with the decimal values is the one
    #     to the left or right wrt the cutted bins
    erosion_value = abs(cut / step - ncuts)
    cutted_mask = (
        np.array((cutted > 0), dtype=int) - np.array((_fshift(cutted, 0, int(np.sign(cut))) > 0), dtype=int)
    )
    border = (cutted_mask > 0)
    return cutted * (1.0 - border * erosion_value)


def apply_vignetting(
    camera: CodedMaskCamera,
    shadowgram: NDArray,
    shift_x: float,
    shift_y: float,
) -> NDArray:
    """
    Applies vignetting effects to a shadowgram based on source position.
    """
    bins = camera.bins_detector

    angle_x_rad = (-1) * np.arctan(shift_x / camera.mdl["mask_detector_distance"])
    red_factorx = camera.mdl["mask_thickness"] * np.tan(angle_x_rad)
    # since the mask detector distance is defined as the distance between the
    # detector top and the mask top, erosion shall cut on the left-side of the
    # shadowgram when sources have negative `angle_x_rad`.
    # if the mask detector distance was defined as the distance between the
    # detector top and the mask bottom, erosion should have been applied to the
    # right side, i.e. `red_factor` should be multiplied by -1.
    sg1 = _erosion(shadowgram, bins.x[1] - bins.x[0], red_factorx)

    angle_y_rad = (-1) * np.arctan(shift_y / camera.mdl["mask_detector_distance"])
    red_factory = camera.mdl["mask_thickness"] * np.tan(angle_y_rad)
    sg2 = _erosion(shadowgram.T, bins.y[1] - bins.y[0], red_factory)
    return sg1 * sg2.T


def apply_detector_resolution(
    camera: CodedMaskCamera,
    shadowgram: NDArray,
) -> NDArray:
    """
    Applies finite detector spatial resolution effects to a shadowgram.
    """
    return convolve(
        shadowgram, _wfm_psfy_kernel_cached(camera), mode="same",
    )


def _shifts_interp(
    camera: CodedMaskCamera,
    shift_y: float,
    shift_x: float,
) -> tuple[float, float]:
    """
    # NOTE: be careful with sky-shifts sign and array shifting verse
    """
    # convert sky-shifts coords [mm] to px indexes
    # NOTE: px indexes are centered to conserve binning structure
    n, m = camera.shape_sky
    i, j = shift2pos(camera, shift_x, shift_y)
    r, c = (n - 1) // 2 - i, (m - 1) // 2 - j

    # compute pixel indexes decimal parts
    def fpixel(
        coord: float,
        bin_coord: float,
        px_size: float,
    ) -> float:
        """Returns the pixel decimal part for the input sky-shift."""
        # handle on-axis source case
        if float(coord) == 0.0:
            return 0.0
        return (coord - bin_coord) / px_size
    
    binsx, binsy = pos2shift(camera, i, j)
    pxdimx, pxdimy = (
        camera.specs['mask_deltax'] / camera.upscale_f.x,
        camera.specs['mask_deltay'] / camera.upscale_f.y,
    )
    fr, fc = (
        r + fpixel(shift_y, binsy, pxdimy),
        c + fpixel(shift_x, binsx, pxdimx),
    )
    return fr, fc


def model_shadowgram(
    camera: CodedMaskCamera,
    shift_x: float,
    shift_y: float,
    vignetting: bool = True,
    psfy: bool = True,
) -> NDArray:
    """
    Generates a normalized shadowgram for a point source
    with fractional shift of the mask pattern.
    """    
    # apply instrument effects and shift mask pattern
    mask_vignetted = (
        apply_vignetting(camera, camera.mask, shift_x, shift_y)
        if vignetting else camera.mask.astype(float)
    )

    fr, fc = _shifts_interp(camera, shift_y, shift_x)
    mask_shifted = _fshift(mask_vignetted, fr, fc)

    sg = (
        apply_detector_resolution(camera, mask_shifted)
        if psfy else mask_shifted
    )

    # extract normalised detector image
    i_min, i_max, j_min, j_max = _detector_footprint(camera)
    detector = sg[i_min:i_max, j_min:j_max]
    detector *= camera.bulk
    detector /= np.sum(detector)
    return detector


# end