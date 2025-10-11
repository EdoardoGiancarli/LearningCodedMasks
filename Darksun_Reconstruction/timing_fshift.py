from timeit import timeit

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift as ndshift

from fract_shift2 import _shift, fshift, scipy_fshift


def _shift_old(a: NDArray, i: int, j: int) -> NDArray:
    """Shifts 2D array of `i` rows and `j` cols."""
    u, v = a.shape
    # avoid memory overload
    if abs(i) >= u or abs(j) >= v:
        return np.zeros_like(a)
    
    # vertical shift
    vpadded = np.pad(
        a,
        ((0 if i < 0 else i, 0 if i >= 0 else -i), (0, 0))
    )
    vpadded = vpadded[:u, :] if i > 0 else vpadded[-u:, :]
    # horizontal shift
    hpadded = np.pad(
        vpadded,
        ((0, 0), (0 if j < 0 else j, 0 if j >= 0 else -j)),
    )
    hpadded = hpadded[:, :v] if j > 0 else hpadded[:, -v:]
    return hpadded


def fshift_old(
    arr: NDArray,
    shifty: int | float,
    shiftx: int | float,
) -> NDArray:
    """
    """
    def apply_decimal_correction(a: NDArray, dec: float) -> NDArray:
        """Applies decimal correction along columns."""
        end_mask = (
            np.array((a > 0), dtype=int) - np.array((_shift_old(a, 0, int(np.sign(dec))) > 0), dtype=int)
        ) > 0
        front_mask = (
            np.array((_shift_old(a, 0, int(np.sign(dec))) > 0), dtype=int) - np.array((a > 0), dtype=int)
        ) > 0
        return (
            a * (1.0 - abs(dec) * end_mask) + _shift_old(a, 0, int(np.sign(dec))) * abs(dec) * front_mask
        )

    # check no shift
    if (float(shifty) == 0.0) and (float(shiftx) == 0.0):
        return arr
        
    # apply integer array shift
    r, c = map(int, (shifty, shiftx))
    shifted = _shift_old(arr, r, c).astype(float)
    # correct edges for decimal shift (end-elements and front-elements)
    rdec, cdec = (shifty - r, shiftx - c)
    shifted_ = apply_decimal_correction(shifted.T, rdec).T
    shifted_ = apply_decimal_correction(shifted_, cdec)
    return shifted_



if __name__ == '__main__':

    REP = 100

    a = np.ones((650, 1040))
    sy, sx = 100, -100

    t1 = timeit('_shift_old(a, sy, sx)', globals=globals(), number=REP)
    t2 = timeit('_shift(a, sy, sx)', globals=globals(), number=REP)

    t3 = timeit('fshift_old(a, sy, sx)', globals=globals(), number=REP)
    t4 = timeit('scipy_fshift(a, sy, sx)', globals=globals(), number=REP)
    t5 = timeit('fshift(a, sy, sx)', globals=globals(), number=REP)

    print(
        f'_shift_old: {t1 / REP}s',
        f'_shift: {t2 / REP}s',

        f'fshift_old: {t3 / REP}s', 
        f'scipy_fshift: {t4 / REP}s',
        f'fshift: {t5 / REP}s',
    )


# end