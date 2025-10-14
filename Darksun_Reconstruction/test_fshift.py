"""
Tests for arrays upscaling and downscaling.
"""

import unittest
from unittest import TestCase

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift as ndshift

from fract_shift2 import _shift, fshift


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


class TestFractShift(TestCase):
    """Test for the `fshift()` method in `images.py`."""

    def test_equal2scipyfshift(self):
        """
        Test if `fshift()` has the same output of scipy's `shift()`,
        with shifts greater than 1.
        """
        arr = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 1, 1, 1, 0, 0,],
                [0, 0, 1, 1, 1, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
            ],
        )

        shifty, shiftx = 2.1, 3.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

        shifty, shiftx = -2.1, 3.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

        shifty, shiftx = 2.1, -3.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

        shifty, shiftx = -2.1, -3.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

    def test_equal2scipyfshift_decimal(self):
        """
        Test if `fshift()` has the same output of scipy's `shift()`,
        with shifts in (0, 1).
        """
        arr = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 1, 1, 1, 0, 0,],
                [0, 0, 1, 1, 1, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
            ],
        )

        shifty, shiftx = 0.1, 0.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

        shifty, shiftx = -0.1, 0.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

        shifty, shiftx = 0.1, -0.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

        shifty, shiftx = -0.1, -0.4
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = fshift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)
    
    def test_int_shift(self):
        """
        Test if `_shift()` has the same output of scipy's `shift()`.
        """
        arr = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 1, 1, 1, 0, 0,],
                [0, 0, 1, 1, 1, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
            ],
        )

        shifty, shiftx = 1, 1
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = _shift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

        shifty, shiftx = -1, 1
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = _shift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

        shifty, shiftx = 1, -1
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = _shift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)

        shifty, shiftx = -1, -1
        expected = scipy_fshift(arr, shifty, shiftx)
        shifted = _shift(arr, shifty, shiftx)
        np.testing.assert_array_almost_equal(expected, shifted)


    

if __name__ == "__main__":
    unittest.main()


# end