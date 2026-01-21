"""
Module for PSFy kernel definition, to simulate finite detector spatial resolution effects.
"""

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from bloodmoon.mask import CodedMaskCamera


def _modsech(
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


def _wfm_psfy(x: NDArray) -> NDArray:
    """
    PSF function in y direction as fitted from WFM simulations.

    Args:
        x: a numpy array or value, in millimeters

    Returns:
        numpy array or value
    """
    PSFY_WFM_PARAMS = {
        "norm": 1.0,
        "center": 0.0,
        "alpha": 0.5459735904725987,
        "beta": 0.7363355668833482,
    }
    return _modsech(x, **PSFY_WFM_PARAMS)


def _wfm_psfy_kernel(camera: CodedMaskCamera) -> NDArray:
    """
    Returns not normalised PSF convolution kernel.
    """
    # we take a whole slit to have a good kernel spatial extension
    px_ydim = camera.mdl['mask_deltay'] / camera.upscale_f.y
    slit_dim = camera.mdl['slit_deltay']
    # the kernel must have the same binning as the mask elements
    bins = np.linspace(-slit_dim, slit_dim, int(2 * slit_dim / px_ydim) + 1)
    kernel = _wfm_psfy(bins).reshape(len(bins), -1)
    #kernel = kernel / np.sum(kernel)
    return kernel


def wfm_psfy_kernel(camera: CodedMaskCamera) -> NDArray:
    """
    Returns PSF convolution kernel.
    At present, it ignores the `x` direction, since PSF characteristic lenght is much shorter
    than typical bin size, even at moderately large upscales.

    Args:
        camera: a CodedMaskCamera object.

    Returns:
        A column array convolution kernel.
    """
    PSFY_GAUSS_PARAMS = {
        "sigma": 0.17749677955602094,
    }

    kernel = _wfm_psfy_kernel(camera)
    psfy = gaussian_filter(
        kernel, PSFY_GAUSS_PARAMS['sigma'], mode='constant', cval=0.0,
    )
    return psfy / np.sum(psfy)


@lru_cache(maxsize=1)
def _wfm_psfy_kernel_cached(camera: CodedMaskCamera):
    """Caching helper."""
    return wfm_psfy_kernel(camera)


# end