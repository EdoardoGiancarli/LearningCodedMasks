"""
Module for testing the mask vignetting instr. effect in the new architecture with supporting
mechanics upon the mask ribs.
"""

import unittest
from unittest import TestCase
from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np

from bloodmoon.mask import CodedMaskCamera, codedmask
from bloodmoon.coords import shift2angle, angle2shift
from bloodmoon.images import _erosion
from bloodmoon.optim import apply_vignetting as std_vignetting

@dataclass
class RibsStructure:
    """
    Container for camera mask ribs structure specifics in [mm].
    """
    ribs_dim: float
    supp_heigth: float
    supp_equiv_thickness: float

def get_ribs_struct(
    ribs_dim: float = 2.5,
    supp_heigth: float = 1.75,
    supp_equiv_thickness: float = 0.25,
) -> RibsStructure:
    ribs = RibsStructure(
        ribs_dim=ribs_dim,
        supp_heigth=supp_heigth,
        supp_equiv_thickness=supp_equiv_thickness,
    )
    return ribs


def apply_vignetting(
    camera: CodedMaskCamera,
    shadowgram: NDArray,
    shift_x: float,
    shift_y: float,
    ribs_struct: RibsStructure,
) -> NDArray:
    """
    Apply vignetting effects to a shadowgram based on source position.
    """
    px_dim_y = camera.specs.mask_deltay / camera.upscale_f.y

    def project_ribs_support(dist: float) -> float:
        """
        Projects the mask ribs support on the mask binning elements, and
        corrects the projection value to allows for correct erosion
        of the mask physical elements starting from the pixels' edges.
        """
        shift_px = shift_y / px_dim_y
        bin_erosion_start = (
            abs(shift_px - int(shift_px)) if (shift_px > 0)
            else abs(shift_px - int(shift_px)) - 1.0
        )
        return dist + bin_erosion_start * px_dim_y

    def apply_ribs_vignetting(sg: NDArray, step: float, cut: float) -> NDArray:
        """Applies camera mask ribs structure correction for vignetting."""
        eroded = _erosion(sg.T, step, cut)
        return eroded.T

    # apply vignetting for standard mask structure
    sg_vignetted = std_vignetting(camera, shadowgram, shift_x, shift_y)

    # correct vignetted sg for the ribs structure
    # - compute critical angle for correction activation
    #
    #                    \ 
    #                  <-->  
    #                  _t__\     
    #                 |   | \  crit_angle
    #                 |   |__\
    #              h  |   |   \      |      p = 0.5 * (L - t) 
    #                 |   |    \     |      crit_angle = atan(p / h)
    #                 |   |     \    |
    #                 |   |      \   |
    #         --------------------\__|
    #         <------------------> \ |  crit_angle
    #                    p <----->  \|    
    #                                |
    #
    struct_eff_base = 0.5 * (ribs_struct.ribs_dim - ribs_struct.supp_equiv_thickness)
    crit_angle_y = abs(np.rad2deg(np.atan(struct_eff_base / ribs_struct.supp_heigth)))
    angle_y = shift2angle(camera, shift_y)
    # - compute ribs structure effect
    if (abs(angle_y) > crit_angle_y):
        erosion_dist_y = ribs_struct.supp_heigth * np.tan(np.deg2rad(abs(angle_y))) - struct_eff_base
        # since the ribs correction is applied before the mask close elements,
        # the shift of the array must be anti-parallel, and we apply a minus
        erosion_dist_y *= (-np.sign(angle_y))
        dist_y = project_ribs_support(erosion_dist_y)
        sg_corrected = apply_ribs_vignetting(sg_vignetted, px_dim_y, dist_y)
    else:
        sg_corrected = sg_vignetted

    return sg_corrected





class TestMaskVignetting(TestCase):
    """Class for testing mask vignetting with new architecture."""

    def setUp(self):
        #BASEPATH: str = '...'
        BASEPATH: str = '/mnt/d/PhD_AASS/Coding/Images_fits'
        MASK_FITS: str = "wfm_mask_NTHT_20260129_CORRECTED.fits"

        UPS_X: int = 5
        UPS_Y: int = 1
        self.wfm = codedmask(f'{BASEPATH}/{MASK_FITS}', UPS_X, UPS_Y)
        self.ribs_struct = get_ribs_struct()
    
    def test_ribs_corr_activation(self) -> None:
        """
        Tests if ribs correction is correctly activated.
        """
        def do_test(angle_x: float, angle_y: float) -> None:
            sx, sy = map(lambda x: angle2shift(self.wfm, x), (angle_x, angle_y))
            std_vignetted = std_vignetting(self.wfm, self.wfm.mask, sx, sy)
            ribs_vignetted = apply_vignetting(self.wfm, self.wfm.mask, sx, sy, self.ribs_struct)
            res = std_vignetted - ribs_vignetted
            np.testing.assert_allclose(res, np.zeros_like(std_vignetted))
            return None

        struct_eff_base = 0.5 * (self.ribs_struct.ribs_dim - self.ribs_struct.supp_equiv_thickness)
        crit_angle_y = abs(np.rad2deg(np.atan(struct_eff_base / self.ribs_struct.supp_heigth)))
        
        angle_x, angle_y = 0.0, crit_angle_y - 1e-6
        do_test(angle_x, angle_y)
        with self.assertRaises(AssertionError):
            angle_x, angle_y = 0.0, crit_angle_y + 1e-6
            do_test(angle_x, angle_y)
        return None






if __name__ == '__main__':
    unittest.main()


# end