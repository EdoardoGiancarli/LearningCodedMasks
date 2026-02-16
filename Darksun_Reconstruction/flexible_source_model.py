"""
Temporary module for new bm source model generation.
In this version (3.0) the instr. effects can be given as input to the model as custom funcs. 
"""

from typing import Callable

from numpy.typing import NDArray
import numpy as np

from bloodmoon.images import fshift
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import decode
from bloodmoon.optim import _detector_footprint_cached
from bloodmoon.optim import apply_vignetting
from bloodmoon.optim import apply_detector_resolution


def _shift_mask_pattern(
    camera: CodedMaskCamera,
    shift_x: float,
    shift_y: float,
) -> NDArray:
    """Shifts the camera mask pattern matching the source direction."""
    pxdimy, pxdimx = (
        camera.specs.mask_deltay / camera.upscale_f.y,
        camera.specs.mask_deltax / camera.upscale_f.x,
    )
    fr, fc = (
        (-1.0) * shift_y / pxdimy,
        (-1.0) * shift_x / pxdimx,
    )
    mask_shifted = fshift(camera.mask.astype(float), fr, fc)
    return mask_shifted

def _process_mask_pattern(
    camera: CodedMaskCamera,
    shadowgram: NDArray,
    shift_x: float,
    shift_y: float,
    vignetting: bool | Callable[[CodedMaskCamera, NDArray, float, float], NDArray],
    psfy: bool | Callable[[CodedMaskCamera, NDArray], NDArray],
) -> NDArray:
    """Applies instrumental effects to the mask pattern projection."""
    # vignetting effect
    if vignetting is True:
        shadowgram = apply_vignetting(camera, shadowgram, shift_x, shift_y)
    elif callable(vignetting):
        shadowgram = vignetting(camera, shadowgram, shift_x, shift_y)
    # detector spatial resolution effect
    if psfy is True:
        shadowgram = apply_detector_resolution(camera, shadowgram)
    elif callable(psfy):
        shadowgram = psfy(camera, shadowgram)
    return shadowgram

def _extract_detector(
    camera: CodedMaskCamera,
    shadowgram: NDArray,
) -> NDArray:
    """
    Extracts the detector image from the mask pattern projection on the detector plane.
    """
    i_min, i_max, j_min, j_max = _detector_footprint_cached(camera)
    detector = shadowgram[i_min:i_max, j_min:j_max]
    detector *= camera.bulk
    detector /= np.sum(detector)
    return detector


def model_shadowgram(
    camera: CodedMaskCamera,
    shift_x: float,
    shift_y: float,
    vignetting: bool | Callable[[CodedMaskCamera, NDArray, float, float], NDArray] = True,
    psfy: bool | Callable[[CodedMaskCamera, NDArray], NDArray] = True,
) -> NDArray:
    """Generates a normalized shadowgram for a point source."""
    for key, val in {'vignetting': vignetting, 'psfy': psfy}.items():
        if not (isinstance(val, bool) or callable(val)):
            raise ValueError(f"'{key}' must be bool or Callable, got {type(val)} instead.")
    # shift camera mask pattern wrt source local-frame coords
    mask_shifted = _shift_mask_pattern(camera, shift_x, shift_y)
    # apply instrumental effects
    mask_projected = _process_mask_pattern(
        camera, mask_shifted, shift_x, shift_y, vignetting, psfy,
    )
    # extract normalised source detector image
    detector = _extract_detector(camera, mask_projected)
    return detector

def model_sky(
    camera: CodedMaskCamera,
    shift_x: float,
    shift_y: float,
    counts: float,
    vignetting: bool | Callable[[CodedMaskCamera, NDArray, float, float], NDArray] = True,
    psfy: bool | Callable[[CodedMaskCamera, NDArray], NDArray] = True,
) -> NDArray:
    """Computes the source sky model given the PSFY kernel params."""
    detector = model_shadowgram(camera, shift_x, shift_y, vignetting, psfy)
    sky = decode(camera, counts * detector)
    return sky




# --------- TESTING MODEL ------------
import unittest
from unittest import TestCase

from bloodmoon.mask import codedmask
from bloodmoon.optim import model_sky as bm_sky

def array_isclose(res: NDArray, atol: float = 1e-8) -> None:
    np.testing.assert_allclose(res, np.zeros_like(res), atol=atol)


class TestingSrcModelling(TestCase):
    """
    Tests for the v3.0 `model_sky()` method.
    """
    def setUp(self) -> None:
        PATH_MASK_TEST: str = '/mnt/d/PhD_AASS/Coding/Images_fits/wfm_mask_NTHT_20250725.fits'

        self.cts = 1e6
        self.shift_x = 73.85  # [mm] (~20 deg)
        self.shift_y = 54.37  # [mm] (~15 deg)
        self.wfm = codedmask(PATH_MASK_TEST, 5, 2)
        return None

    def test_default_instr_effs(self) -> None:
        """
        Tests if default vignetting and PSFY are the same as in bm.
        """
        def do_test(vignetting: bool, psfy: bool) -> None:
            bm_model = bm_sky(self.wfm, self.shift_x, self.shift_y, self.cts, vignetting, psfy)
            new_model = model_sky(self.wfm, self.shift_x, self.shift_y, self.cts, vignetting, psfy)
            array_isclose(bm_model - new_model)
        
        VIGNETTING, PSFY = False, False
        do_test(VIGNETTING, PSFY)
        VIGNETTING, PSFY = True, True
        do_test(VIGNETTING, PSFY)
        return None
    
    def test_custom_logics1(self) -> None:
        """
        Tests if custom vignetting and PSFY logics work: external funcs logic.
        """
        bm_model = bm_sky(self.wfm, self.shift_x, self.shift_y, self.cts, True, True)
        new_model = model_sky(self.wfm, self.shift_x, self.shift_y, self.cts, apply_vignetting, apply_detector_resolution)
        array_isclose(bm_model - new_model)
        return None
    
    def test_custom_logics2(self) -> None:
        """
        Tests if custom vignetting and PSFY logics work: external custom funcs.
        """
        def custom_vignetting(camera: CodedMaskCamera, sg: NDArray, shift_x: float, shift_y: float) -> NDArray:
            """Applies custom vignetting mask effect to given shadowgram."""
            return sg

        def custom_psfy(camera: CodedMaskCamera, sg: NDArray) -> NDArray:
            """Applies custom detector sp. res. effect to given shadowgram."""
            return sg
        
        # NOTE: custom funcs simulate NOT-active vignetting AND detector sp. res. effects
        bm_model = bm_sky(self.wfm, self.shift_x, self.shift_y, self.cts, False, False)
        new_model = model_sky(self.wfm, self.shift_x, self.shift_y, self.cts, custom_vignetting, custom_psfy)
        array_isclose(bm_model - new_model)
        return None
    
    def test_custom_logics3(self) -> None:
        """
        Tests if custom vignetting and PSFY logics work: external custom funcs with WRONG input.
        """
        def custom_vignetting(sg: NDArray, *args, **kwargs) -> NDArray:
            """Applies custom vignetting mask effect to given shadowgram."""
            return sg

        def custom_psfy(sg: NDArray, *args, **kwargs) -> NDArray:
            """Applies custom detector sp. res. effect to given shadowgram."""
            return sg
        
        with self.assertRaises(Exception):
            model_sky(self.wfm, self.shift_x, self.shift_y, self.cts, custom_vignetting, custom_psfy)
        return None   




if __name__ == "__main__":
    unittest.main()


# end