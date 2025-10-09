import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve

from bloodmoon.coords import shift2pos
from bloodmoon.mask import _detector_footprint, CodedMaskCamera
from bloodmoon.optim import _wfm_psfy_kernel_cached, apply_vignetting


def fshift(
    arr: NDArray,
    shifty: int | float,
    shiftx: int | float,
) -> NDArray:
    """
    Shifts a 2D array elements with integer or fractional shifts.

    Args:
        arr (NDArray):
            Input 2D array.
        shifty (int | float):
            Shift along the y-axis.
        shiftx (int | float):
            Shift along the x-axis.

    Returns:
        output (NDArray): Shifted 2D array casted to float.
    
    Examples:
        >>> arr = np.ones((2, 2))
        >>> # INT shifts
        >>> fshift(arr, 1, 0)       # Shift down by 1
        ... array([[0., 0.],
                   [1., 1.]])
        >>> fshift(arr, 0, -1)      # Shift left by 1
        ... array([[1., 0.],
                   [1., 0.]])
        >>> # FLOAT shifts
        >>> fshift(arr, 1.1, 0)     # Shift up by 1.1
        ... array([[0.9, 0.9],
                   [0.,  0. ]])
        >>> fshift(arr, 0, 1.1)     # Shift right by 1.1
        ... array([[0., 0.9],
                   [0., 0.9]])
        >>> fshift(arr, 1.3, -1.2)  # Shift (down, left) by (1.3, 1.2)
        ... array([[0.  , 0.],
                   [0.56, 0.]])

    ## Notes:
        - If `shifty` or `shiftx` are larger than input `arr.shape`, `fshift()`
          returns a float array full of zeroes to avoid memory overload.
        - CFR with url: [fshift](
        https://github.com/yuri-evangelista/CodedMasks/blob/26a5bb2fa08e37c645f85d55a3a1ef038fe7497d/mask_utils/image_utils.py#L12
        ).
        - CFR with url: [_shift](
        https://github.com/peppedilillo/bloodmoon/blob/976102e8558d2a4b2eeecf0817131f525a82c266/bloodmoon/images.py#L402
        ).
        - This exists because the `scipy.ndimage` one is `~10` times slower.
    """
    n, m = arr.shape
    r, c = map(int, (shifty, shiftx))

    # check no shift
    if (float(shifty) == 0.0) and (float(shiftx) == 0.0):
        return arr
    # avoid memory overload
    if abs(r) >= n or abs(c) >= m:
        return np.zeros_like(arr)
    
    # apply integer array shift
    #   - first, the shift is applied vertically
    #   - then, the shift is applied horizontally
    vpadded = np.pad(arr, ((0 if r < 0 else r, 0 if r >= 0 else -r), (0, 0)))
    vpadded = vpadded[:n, :] if r > 0 else vpadded[-n:, :]
    hpadded = np.pad(
        vpadded,
        ((0, 0), (0 if c < 0 else c, 0 if c >= 0 else -c)),
    )
    shifted = hpadded[:, :m] if c > 0 else hpadded[:, -m:]
    shifted = shifted.astype(float)

    # correct edges for decimal shift
    decr, decc = map(
        lambda x: 1.0 - abs(x),
        (shifty - r, shiftx - c),
    )
    fr, fc = map(
        lambda x: x if x >= 0 else x - 1,
        (r, c),
    )
    shifted[fr, :] *= decr
    shifted[:, fc] *= decc
    return shifted


def _shifts_interp(
    camera: CodedMaskCamera,
    shift_y: float,
    shift_x: float,
) -> tuple[float, float]:
    """
    # NOTE: be careful with sky-shifts sign and array shifting verse
    """
    UPX, UPY = camera.upscale_f
    PXX, PXY = (
        camera.specs['mask_deltax'] / UPX,
        camera.specs['mask_deltay'] / UPY,
    )
    # convert sky-shifts coords [mm] to px indexes
    # NOTE: px indexes are centered to conserve binning structure
    n, m = camera.shape_sky
    i, j = shift2pos(camera, shift_x, shift_y)
    r, c = n // 2 - i, m // 2 - j

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
    
    binsx, binsy = camera.bins_sky
    fr, fc = (
        r + fpixel(shift_y, binsy[i], PXY),
        c + fpixel(shift_x, binsx[j], PXX),
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
    Generates a normalized shadowgram for a point source. In this version,
    the mask pattern is shifted by considering a fractional shift, rather
    than computing the four shadowrgram components to simulate an intra-pixel
    source true position.

    The modeled source detector image may feature:
        * Mask pattern projection;
        * Vignetting effects;
        * PSF convolution over y axis.

    Args:
        camera (CodedMaskCamera):
            CodedMaskCamera instance containing all geometric parameters.
        shift_x (float):
            Source position x-coordinate in sky-shift space [mm].
        shift_y (float):
            Source position y-coordinate in sky-shift space [mm].
        vignetting (bool):
            Flag for simulating vignetting effects.
        psfy (bool):
            Flag for simulating detector reconstruction effects.

    Returns:
        output (NDArray):
            2D array representing the modeled detector image from the source.

    ## Notes:
        - The output source shadowgram is normalised, i.e. sums up to one.
        - CFR with url: [model_shadowgram](
        https://github.com/peppedilillo/bloodmoon/blob/976102e8558d2a4b2eeecf0817131f525a82c266/bloodmoon/optim.py#L171
        ).
    """
    def process_mask(sx: float, sy: float) -> NDArray:
        mask_maybe_vignetted = (
            apply_vignetting(
                camera, camera.mask, sx, sy,
            )
            if vignetting else camera.mask
        )
        mask_maybe_vignetted_maybe_psfy = (
            convolve(
                mask_maybe_vignetted, _wfm_psfy_kernel_cached(camera), mode="same",
            )
            if psfy else mask_maybe_vignetted
        )
        return mask_maybe_vignetted_maybe_psfy
    
    # apply instrument effects and shift mask pattern
    mask_p = process_mask(shift_x, shift_y).astype(float)
    r, c = _shifts_interp(camera, shift_y, shift_x)
    sg = fshift(mask_p, r, c)

    # extract normalised detector image
    i_min, i_max, j_min, j_max = _detector_footprint(camera)
    detector = sg[i_min:i_max, j_min:j_max]
    detector *= camera.bulk
    detector /= np.sum(detector)
    return detector


# end