import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve

from bloodmoon.coords import shift2pos
from bloodmoon.images import _shift
from bloodmoon.mask import _detector_footprint, CodedMaskCamera
from bloodmoon.optim import _wfm_psfy_kernel_cached, apply_vignetting


def _shifts_interp(
    camera: CodedMaskCamera,
    shift_y: float,
    shift_x: float,
) -> tuple[float, float]:
    """
    Interpolates the amount of pixels to shift the coded-mask camera mask
    pattern by from the input source local-frame sky shifts coordinates.

    Args:
        camera (CodedMaskCamera):
            CodedMaskCamera instance containing all geometric parameters.
        shift_y (float):
            Source position x-coordinate in sky-shift space [mm].
        shift_x (float):
            Source position y-coordinate in sky-shift space [mm].
    
    Returns:
        output (tuple[float, float]):
            Tuple of decimal pixel for the mask pattern shifting.
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
        shift: float,
        bin_shift: float,
        px_size: float,
    ) -> float:
        """Returns the pixel decimal part for the input sky-shift."""
        # handle on-axis source case
        if float(shift) == 0.0:
            return 0.0
        return (shift - bin_shift) / px_size
    
    binsx, binsy = camera.bins_sky
    fr, fc = (
        r + fpixel(shift_y, binsy[i], PXY),
        c + fpixel(shift_x, binsx[j], PXX),
    )
    return fr, fc


def _apply_decimal_correction(
    mask: NDArray,
    decimal_i: float,
    decimal_j: float,
) -> NDArray:
    """
    Applies the decimal correction to the mask pattern due to the
    disalignment of the source wrt the pixel binning structure.
    """
    def source_decimal_correction(arr: NDArray, dec: float) -> NDArray:
        """Applies source-binning structure disalignment correction."""
        raise NotImplementedError
        edges = (arr - _shift(arr, (0, int(np.sign(dec))))) > 0 # PROBLEM: since mask is not INT, there are residual positive values
        arr[edges] *= dec # meh, what if arr is modified? TO CHECK
        return arr
    
    sgy = source_decimal_correction(mask.T, decimal_i)
    sgx = source_decimal_correction(mask, decimal_j)
    return sgy.T * sgx


def shift_mask(
    mask: NDArray,
    shifty: int | float,
    shiftx: int | float,
) -> NDArray:
    """
    Shifts the coded-mask camera mask pattern array by `(shifty, shiftx)`.
    The input shifts represent the number of pixels the mask pattern is
    shifted by (**NOT** the source local-frame sky shifts coordinates).

    Args:
        mask (NDArray):
            Input 2D array.
        shifty (int | float):
            Shift along the y-axis.
        shiftx (int | float):
            Shift along the x-axis.

    Returns:
        output (NDArray): Shifted 2D mask array casted to float.
    
    Examples:
        >>> mask = np.ones((2, 2))
        >>> # INT shifts
        >>> shift_mask(mask, 1, 0)       # Shift down by 1
        ... array([[0., 0.],
                   [1., 1.]])
        >>> shift_mask(mask, 0, -1)      # Shift left by 1
        ... array([[1., 0.],
                   [1., 0.]])
        >>> # FLOAT shifts
        >>> shift_mask(mask, 1.1, 0)     # Shift up by 1.1
        ... array([[0.9, 0.9],
                   [0.,  0. ]])
        >>> shift_mask(mask, 0, 1.1)     # Shift right by 1.1
        ... array([[0., 0.9],
                   [0., 0.9]])
        >>> shift_mask(mask, 1.3, -1.2)  # Shift (down, left) by (1.3, 1.2)
        ... array([[0.  , 0.],
                   [0.56, 0.]])

    ## Notes:
        - If `shifty` or `shiftx` are larger than input `mask.shape`, a
          float array full of zeroes is returned to avoid memory overload.
    """
    if (float(shifty) == 0.0) and (float(shiftx) == 0.0):
        return mask
    
    r, c = map(int, (shifty, shiftx))
    mask_ = _apply_decimal_correction(mask, shifty - r, shiftx - c)
    shifted = _shift(mask_, (r, c)).astype(float)
    return shifted


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
    sg = shift_mask(mask_p, r, c)

    # extract normalised detector image
    i_min, i_max, j_min, j_max = _detector_footprint(camera)
    detector = sg[i_min:i_max, j_min:j_max]
    detector *= camera.bulk
    detector /= np.sum(detector)
    return detector


# end