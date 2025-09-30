import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate

from bloodmoon.mask import CodedMaskCamera
from darksun.images import unframe


def solid_angle(
    x: float | NDArray,
    y: float | NDArray,
    width: float,
    height: float,
    distance: float,
) -> NDArray:
    """
    Computes the solid angle covered by a rectangular plate of physical
    dimension `width` X `height` at a given `distance` from an observer
    located at `(x, y)` wrt the plate bottom-left corner.

                        |-------- width --------|
                                |
                         _______|_______________       _ _
                        |       |               |       |
                        |   4   |       3       |       |
                        |       |               |       |
                     ___|_______|_______________|____   |height
                        |       |               |       |
                        |   1  y|       2       |       |    
                        |_______|_______________|      _|_
                            x   |
                                |                 
    
    To compute the solid angle, the plate is divided in four sub-portions,
    with the observer located on one corner of each one. The plate solid
    angle can be computed by averaging the sub-portions solid angles seen
    by an on-axis observer.

    The sub-rectangules have areas:
        * `A_r1 = x * y`
        * `A_r2 = (width - x) * y`
        * `A_r3 = (width - x) * (height - y)`
        * `A_r4 = x * (height - y)`

    Args:
        x (float | NDArray):
            Observer coord along the x-axis wrt plate bottom-left corner.
            A non-zero solid angle requires that `0 <= x <= width / 2`.
        y (float | NDArray):
            Observer coord along the y-axis wrt plate bottom-left corner.
            A non-zero solid angle requires that `0 <= y <= height / 2`.
        width (float):
            Width of the rectangular plate.
        height (float):
            Height of the rectangular plate.
        distance (float):
            Distance between the plate and the observer.

    Returns:
        output (float | NDArray):
            Solid angle on the plate seen by the observer.
    
    Raises:
        ValueError: If input coords (x, y) not in the range [0, width / 2] x [0, height / 2].

    ## Notes
        - CFR with:
            * https://github.com/yuri-evangelista/CodedMasks/blob/26a5bb2fa08e37c645f85d55a3a1ef038fe7497d/mask_utils/imaging_utils.py#L58
            * https://vixra.org/pdf/2001.0603v2.pdf [Eq. 27, 34]
    """
    def on_axis_solid_angle(
        a: float | NDArray,
        b: float | NDArray,
    ) -> float | NDArray:
        """
        Computes the solid angle covered by a plate of dimension
        `a` X `b` wrt an observer at a given ox-axis `distance`.
        """
        alpha = a / (2 * distance)
        beta = b / (2 * distance)
        return 4 * np.arctan((alpha * beta) / np.sqrt(1 + alpha**2 + beta**2))
    
    if (
        np.any((x < 0) | (x > width / 2) | (y < 0) | (y > height / 2))
    ):
        raise ValueError(
            f"Invalid coords (x, y). Coords must be in the range [0, {width / 2}] x [0, {height / 2}]."
        )
    
    sub_portions = (
        (x, y),
        ((width - x), y),
        ((width - x), (height - y)),
        (x, (height - y)),
    )
    Omega = tuple(
        on_axis_solid_angle(2 * a, 2 * b) for a, b in sub_portions
    )
    return sum(Omega) / len(Omega)


def detector_solid_angle(camera: CodedMaskCamera) -> NDArray:
    """
    Computes the sky solid angle profile seen by each active element
    of the coded-mask camera detector.

    The solid angle is computed by considering only the instrument
    geometry, taking into account the mask physical dimension, which
    represents the base of a pyramid whose vertex sits on the center
    of each active element of the detector plane, and all the active
    elements coordinates (along the plane).
    The final array is masked with the detector bulk profile.

    Args:
        camera (CodedMaskCamera):
            Camera instance containing the system geometry info.
    
    Returns:
        output (NDArray):
            2D array representing the solid angle profile.
    
    ## Notes:
        - CFR with:
            * https://github.com/yuri-evangelista/CodedMasks/blob/26a5bb2fa08e37c645f85d55a3a1ef038fe7497d/mask_utils/imaging_utils.py#L104
    """
    UPX, UPY = camera.upscale_f

    # define mask physical dim (width, height)
    d = camera.specs['mask_detector_distance']
    maxx, minx = camera.specs['mask_maxx'], camera.specs['mask_minx']
    maxy, miny = camera.specs['mask_maxy'], camera.specs['mask_miny']
    maskplate_physdim = (maxx - minx, maxy - miny)

    # compute x- and y-coords for the detector elements solid angle
    #   - first define detector plane active elements positions
    #   - elements coords are clipped to remove binning artefact
    #   - the solid angle is masked with the bulk active elements
    _binsx, _binsy = camera.bins_detector
    centers_x, centers_y = (
        _binsx[:-1] + camera.specs['mask_deltax'] / (2 * UPX),
        _binsy[:-1] + camera.specs['mask_deltay'] / (2 * UPY),
    )
    x = np.clip(maxx - np.abs(centers_x[np.newaxis, :]), a_min=0, a_max=None)
    y = np.clip(maxy - np.abs(centers_y[:, np.newaxis]), a_min=0, a_max=None)
    return solid_angle(x, y, *maskplate_physdim, d) * (camera.bulk > 0)


def sky_variance(
    camera: CodedMaskCamera,
    detector: NDArray,
) -> NDArray:
    """
    Reconstructs balanced sky variance from detector image by using
    the expected photon counts, to neglect Poisson fluctuations.

    Args:
        camera (CodedMaskCamera):
            Camera instance containing mask and decoder patterns.
        detector (NDArray):
            2D array of detector counts.

    Returns:
        output (NDArray):
            Balanced variance map of the reconstructed sky image.
            To conserve the observed total counts, the output array
            is clipped in the range `[1e-8, detector.sum()]`.
    
    ## Notes:
        - CFR with:
            * https://github.com/yuri-evangelista/CodedMasks/blob/26a5bb2fa08e37c645f85d55a3a1ef038fe7497d/mask_utils/imaging_utils.py#L134
    """
    # retrieve total detector counts and total active elements
    sum_det, sum_bulk = map(np.sum, (detector, camera.bulk))

    # compute expected counts for the detector image
    # - the expected counts array `Lambda` can be built as the product between
    #   a normalised matrix `Omega` representing the solid angle seen by each
    #   active pixel; and the total observed detector counts (i.e., `sum_det`)
    Omega = detector_solid_angle(camera)
    Lambda = sum_det * Omega / Omega.sum()

    # balanced variance components
    var = correlate(np.square(camera.decoder), Lambda, mode="full")
    bal = np.square(camera.balancing) * sum_det / np.square(sum_bulk)
    covar = correlate(camera.decoder, Lambda, mode="full")

    var_bal = (
        var + bal - 2 * covar * camera.balancing / sum_bulk
    )
    return np.clip(var_bal, a_min=1e-8, a_max=detector.sum())


def sky_significance(
    sky: NDArray,
    var: NDArray,
    *,
    ycut: int | None = None,
    xcut: int | None = None,
) -> NDArray:
    """
    Computes signal-to-noise ratio from sky signal and variance arrays.
    It's possible to unframe the significance with `ycut` and `xcut` to
    remove boundary effects due to low variance values (non-Poisson).

    Args:
        sky (NDArray):
            Sky signal values.
        var (NDArray):
            Sky variance values.
        ycut (int | None, optional (default=`None`)):
            Unframing factor over the y axis (rows).
        xcut (int | None, optional (default=`None`)):
            Unframing factor over the x axis (columns).

    Returns:
        output (NDArray):
            Sky significance calculated as `sky / sqrt(variance)`. If
            `ycut` and/or `xcut` are specified, an array view is returned.
    """
    return unframe(sky / np.sqrt(var), ycut, xcut)


# end