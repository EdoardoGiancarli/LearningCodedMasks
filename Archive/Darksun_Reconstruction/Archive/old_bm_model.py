"""
Module with old bm source model.
"""

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve

from bloodmoon.images import _rbilinear
from bloodmoon.mask import _bisect_interval
from bloodmoon.mask import _detector_footprint
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import decode


def _shift_bm(
    a: NDArray,
    shift_ext: tuple[int, int],
) -> NDArray:
    """Shifts a 2D numpy array by the specified amount in each dimension.
    This exists because the scipy.ndimage one is slow.

    Args:
        a: Input 2D numpy array to be shifted.
        shift_ext: Tuple of (row_shift, column_shift) where positive values shift down/right
            and negative values shift up/left. Values larger than array dimensions
            result in an array of zeros.

    Returns:
        np.array: A new array of the same shape as the input, with elements shifted
            and empty spaces filled with zeros.

    Examples:
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> _shift(arr, (1, 0))  # Shift down by 1
        array([[0, 0],
               [1, 2]])
        >>> _shift(arr, (0, -1))  # Shift left by 1
        array([[2, 0],
               [4, 0]])
    """
    n, m = a.shape
    shift_i, shift_j = shift_ext
    if abs(shift_i) >= n or abs(shift_j) >= m:
        # won't load into memory 66666666 x 66666666 bullshit matrix
        return np.zeros_like(a)
    vpadded = np.pad(a, ((0 if shift_i < 0 else shift_i, 0 if shift_i >= 0 else -shift_i), (0, 0)))
    vpadded = vpadded[:n, :] if shift_i > 0 else vpadded[-n:, :]
    hpadded = np.pad(
        vpadded,
        ((0, 0), (0 if shift_j < 0 else shift_j, 0 if shift_j >= 0 else -shift_j)),
    )
    hpadded = hpadded[:, :m] if shift_j > 0 else hpadded[:, -m:]
    return hpadded


def _modsech_bm(
    x: NDArray,
    norm: float,
    center: float,
    alpha: float,
    beta: float,
) -> NDArray:
    """
    PSF fitting function template.

    Args:
        x: a numpy array or value, in millimeters
        norm: normalization parameter
        center: center parameter
        alpha: alpha shape parameter
        beta: beta shape parameter

    Returns:
        numpy array or value, depending on the input
    """
    return norm / np.cosh(np.abs((x - center) / alpha) ** beta)


def _wfm_psfy_bm(x: NDArray) -> NDArray:
    """
    PSF function in y direction as fitted from WFM simulations.

    Args:
        x: a numpy array or value, in millimeters

    Returns:
        numpy array or value
    """
    PSFY_WFM_PARAMS = {
        "center": 0,
        "alpha": 0.3214,
        "beta": 0.6246,
    }
    return _modsech_bm(x, norm=1, **PSFY_WFM_PARAMS)


def _wfm_psfy_kernel_bm(camera: CodedMaskCamera) -> NDArray:
    """
    Returns PSF convolution kernel.
    At present, it ignores the `x` direction, since PSF characteristic lenght is much shorter
    than typical bin size, even at moderately large upscales.

    Args:
        camera: a CodedMaskCamera object.

    Returns:
        A column array convolution kernel.
    """
    bins = camera.bins_detector
    min_bin, max_bin = _bisect_interval(bins.y, -camera.specs.slit_deltay, camera.specs.slit_deltay)
    bin_edges = bins.y[min_bin : max_bin + 1]
    midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
    kernel = _wfm_psfy_bm(midpoints).reshape(len(midpoints), -1)
    kernel = kernel / np.sum(kernel)
    return kernel


def _erosion_bm(
    arr: NDArray,
    step: float,
    cut: float,
) -> NDArray:
    """
    2D matrix erosion for simulating finite thickness effect in shadow projections.
    It takes a mask array and "thins" the mask elements across the columns' direction.

    Comes with NO safeguards: setting cuts larger than step may remove slits or make them negative.

    ⢯⣽⣿⣿⣿⠛⠉⠀⠀⠉⠉⢛⢟⡻⣟⡿⣿⢿⣿⣿⢿⣻⣟⡿⣟⡿⣿⣻⣟⣿⣟⣿⣻⣟⡿⣽⣻⠿⣽⣻⢟⡿⣽⢫⢯⡝
    ⢯⣞⣷⣻⠤⢀⠀⠀⠀⠀⠀⠀⠀⠑⠌⢳⡙⣮⢳⣭⣛⢧⢯⡽⣏⣿⣳⢟⣾⣳⣟⣾⣳⢯⣽⣳⢯⣟⣷⣫⢿⣝⢾⣫⠗⡜
    ⡿⣞⡷⣯⢏⡴⢀⠀⠀⣀⣤⠤⠀⠀⠀⠀⠑⠈⠇⠲⡍⠞⡣⢝⡎⣷⠹⣞⢧⡟⣮⢷⣫⢟⡾⣭⢷⡻⢶⣏⣿⢺⣏⢮⡝⢌
    ⢷⣹⢽⣚⢮⡒⠆⠀⢰⣿⠁⠀⠀⠀⢱⡆⠀⠀⠈⠀⠀⠄⠁⠊⠜⠬⡓⢬⠳⡝⢮⠣⢏⡚⢵⢫⢞⡽⣏⡾⢧⡿⣜⡣⠞⡠
    ⢏⣞⣣⢟⡮⡝⣆⢒⠠⠹⢆⡀⠀⢀⠼⠃⣀⠄⡀⢠⠠⢤⡤⣤⢀⠀⠁⠈⠃⠉⠂⠁⠀⠉⠀⠃⠈⠒⠩⠘⠋⠖⠭⣘⠱⡀
    ⡚⡴⣩⢞⣱⢹⠰⡩⢌⡅⠂⡄⠩⠐⢦⡹⢜⠀⡔⢡⠚⣵⣻⢼⡫⠔⠀⠀⠀⠀⠀⠀⠀⠀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⡄
    ⡑⠦⡑⢎⡒⢣⢣⡑⢎⡰⢁⡒⢰⢠⢣⠞⢁⠢⡜⢢⢝⣺⡽⢮⠑⡈⠀⠀⠀⢀⡀⠀⣾⡟⠁⠀⠀⠠⡀⠀⠀⠀⠀⠀⠀⠐
    ⢘⠰⡉⢆⠩⢆⠡⠜⢢⢡⠣⡜⢡⢎⠧⡐⢎⡱⢎⡱⢊⣾⡙⢆⠁⡀⠄⡐⡈⢦⢑⠂⠹⣇⠀⠀⠀⢀⣿⡀⠀⠀⠀⢀⠀⠄
    ⠈⢆⠱⢈⠒⡈⠜⡈⢆⠢⢱⡘⣎⠞⡰⣉⠎⡴⢋⢰⣻⡞⣍⠂⢈⠔⡁⠆⡑⢎⡌⠎⢡⠈⠑⠂⠐⠋⠁⠀⠀⡀⢆⠠⣉⠂
    ⡉⠔⡨⠄⢂⡐⠤⡐⣄⢣⢧⡹⡜⢬⡑⡌⢎⡵⢋⣾⡳⡝⠤⢀⠊⡔⡈⢆⡁⠮⡜⠬⢠⢈⡐⡉⠜⡠⢃⠜⣠⠓⣌⠒⠤⡁
    ⢌⠢⢡⠘⡄⢎⡱⡑⢎⡳⢎⠵⡙⢆⠒⡍⡞⣬⢛⡶⡹⠌⡅⢂⠡⠐⠐⠂⠄⡓⠜⡈⢅⠢⠔⡡⢊⠔⡡⢚⠤⣋⠤⡉⠒⠠
    ⢢⢑⢢⠱⡘⢦⠱⣉⠞⡴⢫⣜⡱⠂⡬⠜⣵⢊⠷⡸⠥⠑⡌⢂⠠⠃⢀⠉⠠⢜⠨⠐⡈⠆⡱⢀⠣⡘⠤⣉⠒⠄⠒⠠⢁⠡
    ⢌⡚⡌⢆⠳⣈⠦⣛⠴⣓⠮⣝⠃⠐⡁⠖⣭⢚⡴⢃⠆⢢⠑⡌⠀⠀⠌⠐⠠⢜⠢⡀⠡⠐⠡⠘⠠⢁⠂⡉⠐⡀⠂⠄⡈⠄
    ⠦⡱⡘⣌⠳⣌⠳⣌⠳⣍⠞⣥⢣⠀⠈⠑⠢⢍⠲⢉⠠⢁⠊⠀⠁⠀⠄⠡⠈⢂⠧⡱⣀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠐⠀⡀⠂
    ⠂⠥⠑⡠⢃⠌⡓⢌⠳⢌⡹⢄⠣⢆⠀⠀⠀⠈⠀⠀⠀⠀⠀⠈⠀⠀⡌⢢⡕⡊⠔⢡⠂⡅⠂⠀⠀⠀⠀⠀⠐⠈⠀⢀⠀⠀
    ⠈⠄⠡⠐⠠⠈⠔⣈⠐⢂⠐⡨⠑⡈⠐⡀⠀⠀⠀⠀⠀⠀⠀⡀⢤⡘⠼⣑⢎⡱⢊⠀⠐⡀⠁⠀⠀⠀⠐⠀⠀⢀⠀⠀⠀⠀
    ⠀⠈⠄⡈⠄⣁⠒⡠⠌⣀⠒⠠⠁⠄⠡⢀⠁⠀⢂⠠⢀⠡⢂⠱⠢⢍⠳⣉⠖⡄⢃⠀⠀⠄⠂⠀⢀⠈⠀⢀⠈⠀⠀⠀⠀⠀
    ⠀⡁⠆⠱⢨⡐⠦⡑⢬⡐⢌⢢⡉⢄⠃⡄⠂⠁⠠⠀⠄⠂⠄⠡⢁⠊⡑⠌⡒⢌⠢⢈⠀⠄⠂⠁⡀⠀⠂⡀⠄⠂⠀⠀⠀⠀
    ⠤⠴⣒⠦⣄⠘⠐⠩⢂⠝⡌⢲⡉⢆⢣⠘⠤⣁⢂⠡⠌⡐⠈⠄⢂⠐⡀⠂⢀⠂⠐⠠⢈⠀⡐⠠⠀⠂⢁⠀⠀⠀⠀⠀⠀⠀
    ⠌⠓⡀⠣⠐⢩⠒⠦⠄⣀⠈⠂⠜⡈⠦⠙⡒⢤⠃⡞⣠⠑⡌⠢⠄⢂⠐⠀⠀⠀⠀⠀⠀⠂⠀⠐⡀⠁⠠⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠁⡀⢈⠈⡑⠢⡙⠤⢒⠆⠤⢁⣀⠂⠁⠐⠁⠊⠔⠡⠊⠄⠂⢀⠀⠀⠀⠀⠀⠂⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠁⠀⠀⠀⡀⠀⠀⠀⠈⠁⠊⠅⠣⠄⡍⢄⠒⠤⠤⢀⣀⣀⣀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠁⠀⠀⠁⠀⠂⠀⠄⠀⠀⠀⠈⠀⠉⠀⠁⠂⠀⠀⠉⠉⠩⢉⠢⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠂⠀⠀⠀⠀⠐⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠄⠁⠄⠀⠀⠀

    Args:
        arr: 2D input array of integers representing the projected shadow.
        step: The projection bin step.
        cut: Maximum cut width.

    Returns:
        Modified array with shadow effects applied

    Notes:
        * See tests for usage examples.
    """
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("Input array must be of integer type.")

    # number of bins to cut
    ncuts = int(cut / step)
    cutted = arr * (arr & _shift_bm(arr, (0, ncuts))) if ncuts else arr

    # array indexes to be fractionally reduced:
    #   - the bin with the decimal values is the one
    #     to the left or right wrt the cutted bins
    erosion_value = abs(cut / step - ncuts)
    border = (cutted - _shift_bm(cutted, (0, int(np.sign(cut))))) > 0
    return cutted - border * erosion_value


def apply_vignetting_bm(
    camera: CodedMaskCamera,
    shadowgram: NDArray,
    shift_x: float,
    shift_y: float,
) -> NDArray:
    r"""
    Apply vignetting effects to a shadowgram based on source position.
    Vignetting occurs when mask thickness causes partial shadowing at off-axis angles.
    This function models this effect by applying erosion operations in both x and y
    directions based on the source's angular displacement from the optical axis.


                                    <--------> MASK APERTURE

                                  \       \  \ 
                        ___________\       \  \____________
                                   |\       \ |             MASK ELEMENT
                        ___________| \       \|_____________
                                      \       \  \ 
                                       \       \  \ 
                                        \       \  \ 
                         ________________\_______\__\_________  DETECTOR
                         <--------------->        <->
                               SHIFT             EROSION

    Args:
        camera: CodedMaskCamera instance containing mask and detector geometry
        shadowgram: 2D array representing the detector shadowgram before vignetting
        shift_x: Source displacement from optical axis in x direction (mm)
        shift_y: Source displacement from optical axis in y direction (mm)

    Returns:
        2D array representing the detector shadowgram with vignetting effects applied.
        Values are float between 0 and 1, where lower values indicate stronger vignetting.

    Notes:
        - The vignetting effect increases with larger off-axis angles
        - The effect is calculated separately for x and y directions then combined
        - The mask thickness parameter from the camera model determines the strength
          of the effect
    """
    bins = camera.bins_detector

    angle_x_rad = np.arctan(shift_x / camera.specs.mask_detector_distance)
    red_factor = camera.specs.mask_thickness * np.tan(angle_x_rad)
    # since the mask detector distance is defined as the distance between the
    # detector top and the mask top, erosion shall cut on the left-side of the
    # shadowgram when sources have negative `angle_x_rad`.
    # if the mask detector distance was defined as the distance between the
    # detector top and the mask bottom, erosion should have been applied to the
    # right side, i.e. `red_factor` should be multiplied by -1.
    sg1 = _erosion_bm(shadowgram, bins.x[1] - bins.x[0], red_factor)

    angle_y_rad = np.arctan(shift_y / camera.specs.mask_detector_distance)
    red_factor = camera.specs.mask_thickness * np.tan(angle_y_rad)
    sg2 = _erosion_bm(shadowgram.T, bins.y[1] - bins.y[0], red_factor)
    return sg1 * sg2.T


def model_shadowgram_bm(
    camera: CodedMaskCamera,
    shift_x: float,
    shift_y: float,
    vignetting: bool = True,
    psfy: bool = True,
) -> NDArray:
    """
    Generates a normalized shadowgram for a point source.

    The model may feature:
    - Mask pattern projection
    - Vignetting effects
    - PSF convolution over y axis

    Args:
        shift_x: Source position x-coordinate in sky-shift space (mm)
        shift_y: Source position y-coordinate in sky-shift space (mm)
        camera: CodedMaskCamera instance containing all geometric parameters
        vignetting: simulates vignetting effects
        psfy: simulates detector reconstruction effects

    Returns:
        2D array representing the modeled detector image from the source

    Notes:
        * Results are normalized, i.e. sums up to one.
    """

    def process_mask(shift_x, shift_y):
        mask_maybe_vignetted = (
            apply_vignetting_bm(
                camera,
                camera.mask,
                shift_x,
                shift_y,
            )
            if vignetting
            else camera.mask
        )
        mask_maybe_vignetted_maybe_psfy = (
            convolve(
                mask_maybe_vignetted,
                _wfm_psfy_kernel_bm(camera),
                mode="same",
            )
            if psfy
            else mask_maybe_vignetted
        )
        return mask_maybe_vignetted_maybe_psfy

    # relative component map
    components = _rbilinear(shift_x, shift_y, camera.bins_sky.x, camera.bins_sky.y)
    n, m = camera.shape_sky
    detector = np.zeros(camera.shape_detector)
    i_min, i_max, j_min, j_max = _detector_footprint(camera)
    for (c_i, c_j), weight in components.items():
        r, c = (n // 2 - c_i), (m // 2 - c_j)
        mask_p = process_mask(camera.bins_sky.x[c_j], camera.bins_sky.y[c_i])  # mask processed
        sg = _shift_bm(mask_p, (r, c))  # mask shifted processed
        detector += sg[i_min:i_max, j_min:j_max] * weight
    detector *= camera.bulk
    detector /= np.sum(detector)
    return detector


def model_sky_bm(
    camera: CodedMaskCamera,
    shift_x: float,
    shift_y: float,
    fluence: float,
    vignetting: bool = True,
    psfy: bool = True,
) -> NDArray:
    """
    Generate a model of the reconstructed sky image for a point source.

    The model may feature:
    - Mask pattern projection
    - Vignetting effects
    - PSF convolution over y axis
    - Flux scaling

    Args:
        shift_x: Source position x-coordinate in sky-shift space (mm)
        shift_y: Source position y-coordinate in sky-shift space (mm)
        fluence: Source intensity/fluence value
        camera: CodedMaskCamera instance containing all geometric parameters
        vignetting: simulates vignetting effects
        psfy: simulates detector reconstruction effects

    Returns:
        2D array representing the modeled sky reconstruction after all effects
        and processing steps have been applied

    Notes:
        - For optimization, consider using the dedicated, cached function of `optim.py`
    """
    return decode(camera, model_shadowgram_bm(camera, shift_x, shift_y, vignetting=vignetting, psfy=psfy)) * fluence


# end