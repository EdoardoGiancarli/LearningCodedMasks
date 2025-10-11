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


def scipy_fshift(
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


def _shift(
    arr: NDArray,
    rows: int,
    cols: int,
) -> NDArray:
    """
    Performs a 2D integer shift of an array using slicing.
    Areas shifted in from outside the frame are filled with zeros.

    Args:
        arr (NDArray): Input 2D numpy array to be shifted.
        rows (int): Shift value along vertical axis.
        cols (int): Shift value along horizontal axis.

    Returns:
        output (NDArray): Shifted array with same shape.
    
    Raises:
        ValueError: If shift values are not integer.
    
    ## Notes:
        * Shift values larger than input array shape
          results in an array full of zeros.
        * This exists because is `~40` times quicker
          than `scipy.ndimage.shift()`.

    Examples:
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> _shift(arr, 1, 0)   # Shift down by 1
        array([[0, 0],
               [1, 2]])
        >>> _shift(arr, 0, -1)  # Shift left by 1
        array([[2, 0],
               [4, 0]])
    """
    if not (
        isinstance(rows, int) and isinstance(cols, int)
    ):
        raise ValueError('Shift values must be integers.')
    
    # zero-shift
    if rows == 0 and cols == 0:
        return arr.copy()
    
    n, m = arr.shape
    # avoid memory overload
    if abs(rows) >= n or abs(cols) >= m:
        return np.zeros_like(arr)
    
    arr_ = np.zeros_like(arr)
    arr_ystart, arr_yend = max(0, -rows), n - max(0, rows)
    arr_xstart, arr_xend = max(0, -cols), m - max(0, cols)

    y_start, y_end = max(0, rows), n + min(0, rows)
    x_start, x_end = max(0, cols), m + min(0, cols)

    arr_[y_start : y_end, x_start : x_end] = (
        arr[arr_ystart : arr_yend, arr_xstart : arr_xend]
    )
    return arr_


def fshift(
    arr: NDArray,
    shifty: int | float,
    shiftx: int | float,
) -> NDArray:
    """
    Shifts a 2D array elements with integer or fractional shifts.
    Areas shifted in from outside the frame are filled with zeros.

    Args:
        arr (NDArray):
            Input 2D array.
        shifty (int | float):
            Shift along the rows axis.
        shiftx (int | float):
            Shift along the columns axis.

    Returns:
        output (NDArray): Shifted 2D array casted to float.
    
    ## Notes:
        * Shift values larger than input array shape
          results in an array full of zeros.
        * This exists because is `~40` times quicker
          than `scipy.ndimage.shift()`.
        * CFR with url: [fshift](
        https://github.com/yuri-evangelista/CodedMasks/blob/26a5bb2fa08e37c645f85d55a3a1ef038fe7497d/mask_utils/image_utils.py#L12
        ).
    
    Examples:
        >>> # INT shifts
        >>> arr = np.ones((2, 2))
        >>> fshift(arr, 1, 0)       # Shift down by 1
        array([[0., 0.],
               [1., 1.]])
        
        >>> fshift(arr, 0, -1)      # Shift left by 1
        array([[1., 0.],
               [1., 0.]])
        
        >>> # FLOAT shifts
        >>> arr = np.array(
        ...     [[0, 0, 0, 0, 0, 0, 0,],
        ...      [0, 0, 0, 0, 0, 0, 0,],
        ...      [0, 0, 1, 1, 1, 0, 0,],
        ...      [0, 0, 1, 1, 1, 0, 0,],
        ...      [0, 0, 0, 0, 0, 0, 0,],
        ...      [0, 0, 0, 0, 0, 0, 0,]],
        ... )

        >>> fshift(arr, 1.1, 0)     # Shift up by 1.1
        array(
            [[0. , 0. , 0. , 0. , 0. , 0. , 0. ],
             [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
             [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
             [0. , 0. , 0.9, 0.9, 0.9, 0. , 0. ],
             [0. , 0. , 1. , 1. , 1. , 0. , 0. ],
             [0. , 0. , 0.1, 0.1, 0.1, 0. , 0. ]],
        )

        >>> fshift(arr, 0, 1.1)     # Shift left by 1.1
        array(
            [[0. , 0. , 0. , 0. , 0. , 0. , 0. ],
             [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
             [0. , 0. , 0. , 0.9, 1. , 1. , 0.1],
             [0. , 0. , 0. , 0.9, 1. , 1. , 0.1],
             [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
             [0. , 0. , 0. , 0. , 0. , 0. , 0. ]]
        )
    """
    # zero-shift
    if float(shifty) == 0.0 and float(shiftx) == 0.0:
        return arr.copy()
    
    n, m = arr.shape
    # avoid memory overload
    if abs(shifty) >= n or abs(shiftx) >= m:
        return np.zeros_like(arr)
    
    # compute int and decimal shifts
    r, c = map(int, (shifty, shiftx))
    rsign, csign = map(
        lambda x: int(np.sign(x)),
        (shifty, shiftx),
    )
    fr, fc = map(abs, (shifty - r, shiftx - c))

    # - to perform the shifting and the decimal interpolation,
    #   the array is divided into four weighted components
    # - the first component is the INT shift, the other three
    #   are the INT-shifted array in each direction, by one pixel
    # - the weights are normalised to 1
    shifted = _shift(arr, r, c)

    # zero-fract shift
    if fr == 0.0 and fc == 0.0:
        return shifted
    
    shifted_01 = _shift(shifted, 0, csign)
    shifted_10 = _shift(shifted, rsign, 0)
    shifted_11 = _shift(shifted, rsign, csign)
    
    w00 = (1.0 - fr) * (1.0 - fc)
    w01 = (1.0 - fr) * fc
    w10 = fr * (1.0 - fc)
    w11 = fr * fc

    shifted_ = (
        w00 * shifted + w01 * shifted_01 + w10 * shifted_10 + w11 * shifted_11
    )
    return shifted_


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
    arr_mask = (arr > 0) & (_shift(arr, 0, ncuts) > 0)
    cutted = arr * arr_mask if ncuts else arr

    # array indexes to be fractionally reduced:
    #   - the bin with the decimal values is the one
    #     to the left or right wrt the cutted bins
    erosion_value = abs(cut / step - ncuts)
    cutted_mask = (
        np.array((cutted > 0), dtype=int) - np.array((_shift(cutted, 0, int(np.sign(cut))) > 0), dtype=int)
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
    # TODO: fix this comment (depends on what shift is considered, from source or on the detector)
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
    # instrumental effects and shift mask pattern
    # - apply vignetting to mask pattern array
    mask_vignetted = (
        apply_vignetting(camera, camera.mask, shift_x, shift_y)
        if vignetting else camera.mask.astype(float)
    )
    # - shift mask array to match source direction
    fr, fc = _shifts_interp(camera, shift_y, shift_x)
    mask_shifted = fshift(mask_vignetted, fr, fc)
    # - apply detector spatial resolution
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