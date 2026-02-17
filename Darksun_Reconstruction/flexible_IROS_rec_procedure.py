"""
Temporary module for new IROS reconstruction procedure.
In this version, the sub-logics are made flexible by allowing for customisation.
The IROS routine is now intended as a "wrapper" for the main logics (source finding process, parameters fitting and source subtraction).
"""

# search for updated versions of:
#   - finder, fitter, subtractor methods
#   - optimiser method
#
# NOTE: inside the finder there is the sky pos masking
# NOTE: the optimiser is called inside the fitter
#
# NOTE (optimiser): custom obj for `curve_fit` output for `verbose` func input
# NOTE (optimiser): general custom obj for optimising procedure? Some scipy routine have their own output obj...
# NOTE (optimiser): make `verbose` func flexible by again giving it as input



from typing import Callable, Iterable, NamedTuple
import warnings

import numpy as np
from numpy.typing import NDArray

from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import decode
from bloodmoon.mask import variance
from bloodmoon.mask import snratio

type EnergyRange = tuple[float, float]
type CoordShifts = tuple[float, float]

class Source(NamedTuple):
    """
    Source candidate parameters container.
    """
    coords: dict[EnergyRange, CoordShifts]
    cts: float
    snr: float


def iros_singleCAM(
    detector: NDArray,
    camera: CodedMaskCamera,
    max_iterations: int = 5,
    finder: Callable[[NDArray, NDArray], tuple[int, int] | bool] | None = None,
    fitter: Callable[[tuple[int, int], NDArray, NDArray], Source] | None = None,
    subtractor: Callable[[Source, NDArray], NDArray] | None = None,
    varmap: NDArray | None = None,
) -> Iterable[tuple[Source, NDArray]]:
    """
    Performs the Iterative Removal of Sources (IROS) algorithm for a single coded-mask
    camera of the Wide Field Monitor observations.
    """
    # arrs setup
    detector_ = detector.copy()
    skymap = decode(camera, detector)
    varmap = (
        varmap if varmap is not None
        else variance(camera, detector)
    )
    # looping as there's no tomorrow
    for i in range(max_iterations):
        snrmap = snratio(skymap, varmap)
        candidate_pos = finder(skymap, snrmap)

        if not candidate_pos:
            print("\nNo candidates left...")
            break
        try:
            source = fitter(candidate_pos, skymap, snrmap)
        except RuntimeError as e:
            warnings.warn(f"Optimizer failed at iteration {i}:\n\n{e}")
            continue

        detector_ = subtractor(source, detector_)
        skymap = decode(camera, detector_)
        yield (source, skymap)


# end