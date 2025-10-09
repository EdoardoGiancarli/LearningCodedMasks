"""
Testing sources PSF with fractional array shift and vignetting.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift as ndshift


def _shift(
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
    cutted = arr * (arr & _shift(arr, 0, ncuts)) if ncuts else arr

    # array indexes to be fractionally reduced:
    #   - the bin with the decimal values is the one
    #     to the left or right wrt the cutted bins
    erosion_value = abs(cut / step - ncuts)
    border = (cutted - _shift(cutted, 0, int(np.sign(cut)))) > 0
    return cutted * (1.0 - border * erosion_value)


# end