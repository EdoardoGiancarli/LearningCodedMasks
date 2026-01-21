import numpy as np
from numpy.typing import NDArray

from darksun.utils import benchmark_func

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



if __name__ == '__main__':

    REP = 50

    a = np.ones((300, 5000))
    sy, sx = 100, -100

#    t1 = timeit('_shift_old(a, sy, sx)', globals=globals(), number=REP)
#    t2 = timeit('_shift(a, sy, sx)', globals=globals(), number=REP)
#
#    t4 = timeit('scipy_fshift(a, sy, sx)', globals=globals(), number=REP)
#    t5 = timeit('fshift(a, sy, sx)', globals=globals(), number=REP)

    t1, dt1, _ = benchmark_func(_shift_old, a, sy, sx, iterations=REP)
    t2, dt2, _ = benchmark_func(_shift, a, sy, sx, iterations=REP)
    t3, dt3, _ = benchmark_func(scipy_fshift, a, sy, sx, iterations=REP)
    t4, dt4, _ = benchmark_func(fshift, a, sy, sx, iterations=REP)

    print(
        f'_shift_old: {t1} +/- {dt1} s\n',
        f'_shift: {t2} +/- {dt2} s\n',

        f'scipy_fshift: {t3} +/- {dt3} s\n',
        f'fshift: {t4} +/- {dt4} s',
    )


# end