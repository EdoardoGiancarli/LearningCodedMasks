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
from bloodmoon.images import _erosion, fshift


def correct_erosion_value(cut_px: float, shift_px: float) -> float:
    """
    Corrects cut to account for bin edge start-point erosion.\n
    When performing the mask erosion during the projection on the detector
    due to the finite mask thickness (or other structures on the camera),
    we have to account for the mask array pixels value, since the erosion
    mechanism is `px_value * (1 - erosion)`.\n
    If the mask pattern array is fshifted, the pixels at the edge of the
    mask physical elements assume values linked to `f_shift`, and have
    to be accounted for when applying the erosion.
    """
    f_cut = abs(cut_px) % 1
    f_shift = abs(shift_px) % 1

    # if no fshift of the mask pattern array, no need for correction
    if not f_shift:
        return cut_px
    
    # compute effective pixel val after mask is fshifted
    # if the erosion is concordant wrt the mask shifting, then the pixel
    # value is `1.0 - f_shift`, otherwise is just `f_shift`
    concordant = (shift_px > 0) == (cut_px > 0)
    val_px = 1.0 - f_shift if concordant else f_shift

    # base condition for using linear scaling vs capping px val
    use_linear = f_cut <= val_px
    # define linear and capping erosion correction
    cap = 1.0 - val_px
    linear = cap * f_cut / val_px

    # std erosion value correction
    corr = linear if use_linear else cap

    # handle px overstepping cases 
    if int(cut_px) and (f_cut <= f_shift):
        if not concordant:
            corr = cap if use_linear else linear
        elif use_linear:
            corr = cap

    return cut_px + np.sign(cut_px) * corr


def apply_std_vignetting(
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
               |\       \ |x            MASK ELEMENT
    ___________| \       \|_x___________
                  \       \  x
                   \       \  x
                    \       \  x
     ________________\_______\__x_________  DETECTOR
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
    def project_mask_thickness(shift: float, bin_dim: float) -> float:
        """
        Projects the mask thickness on the mask binning elements, and
        corrects the projection value to allows for correct erosion
        of the mask physical elements starting from the pixels' edges.
        """
        # - since the mask detector distance is defined as the distance between the
        #   detector top and the mask top, erosion shall cut on the left-side of the
        #   shadowgram when sources have negative `angle`.
        # - if the mask detector distance was defined as the distance between the
        #   detector top and the mask bottom, erosion should have been applied to the
        #   right side, i.e. `proj` should be multiplied by -1.
        angle = np.arctan(shift / camera.specs.mask_detector_distance)
        proj = camera.specs.mask_thickness * np.tan(angle)
        shift_px = shift / bin_dim
        # - the mask thickness projection has to be corrected by considering the
        #   erosion pixel start point, due to the discretisation of the projection
        #   https://github.com/yuri-evangelista/CodedMasks/blob/main/mask_050_1040x17/new_erosion_20251024.ipynb
        # - when generating the source shadowgram, the mask array is shifted in the
        #   opposite direction wrt the coords values, so here we have to multiply
        #   the shift (in px) by -1.0 to compute the exact array shifting
        return correct_erosion_value(proj / bin_dim, -1.0 * shift_px) * bin_dim
    
    bins = camera.bins_detector
    bin_dim_x, bin_dim_y = (
        bins.x[1] - bins.x[0],
        bins.y[1] - bins.y[0],
    )

    red_factor_x = project_mask_thickness(shift_x, bin_dim_x)
    sg_x = _erosion(shadowgram, bin_dim_x, red_factor_x)

    # - we apply the y-axis erosion to `sg_x`, otherwise the decimal
    #   values of the input shifted shadowgram would be squared
    # - the erosion on the two axes is still independent, as it must be
    red_factor_y = project_mask_thickness(shift_y, bin_dim_y)
    sg_y = _erosion(sg_x.T, bin_dim_y, red_factor_y)

    return sg_y.T










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

    Along the coarse axis we have to account for the ribs structure correction:
        * the ribs design provides a double triangular-shaped support
          system aimed at sustaining efficiently the coded-mask sheet;
        * the total vignetting is applied by accounting for the
          effect of both the up and bottom ribs supports. 

    NOTE: This vignetting correction for the ribs structure is linked to the
          definition of the camera focal distance (top detector - top mask).
          If the focal distance is to be changed, the std vignetting effect
          AND this correction logic have to be modified accordingly by
          changing the verse of the effect application.
    """    
    bins = camera.bins_detector
    bin_dim_x, bin_dim_y = (
        bins.x[1] - bins.x[0],
        bins.y[1] - bins.y[0],
    )
    # - since the mask detector distance is defined as the distance between the
    #   detector top and the mask top, erosion shall cut on the left-side of the
    #   shadowgram when sources have negative `angle`.
    # - if the mask detector distance was defined as the distance between the
    #   detector top and the mask bottom, erosion should have been applied to the
    #   right side, i.e. `proj` should be multiplied by -1.
    angle_x = shift2angle(camera, shift_x)
    mask_thick_proj_x = camera.specs.mask_thickness * np.tan(np.deg2rad(angle_x))
    # - the mask thickness projection has to be corrected by considering the
    #   erosion pixel start point, due to the discretisation of the projection
    #   https://github.com/yuri-evangelista/CodedMasks/blob/main/mask_050_1040x17/new_erosion_20251024.ipynb
    # - when generating the source shadowgram, the mask array is shifted in the
    #   opposite direction wrt the coords values, so here we have to multiply
    #   the shift (in px) by -1.0 to compute the exact array shifting
    red_factor_x = correct_erosion_value(mask_thick_proj_x / bin_dim_x, -1.0 * shift_x / bin_dim_x) * bin_dim_x
    sg_x = _erosion(shadowgram, bin_dim_x, red_factor_x)

    # - we apply the y-axis erosion to `sg_x`, otherwise the decimal
    #   values of the input shifted shadowgram would be squared
    # - the erosion on the two axes is still independent, as it must be
    # - to account for the ribs support system, we first compute the
    #   expected mask sheet thickness projection, and then we check if
    #   the source is displaced enough to activate the ribs correction
    angle_y = shift2angle(camera, shift_y)
    mask_thick_proj_y = camera.specs.mask_thickness * np.tan(np.deg2rad(angle_y))

    # * compute critical angle for correction activation
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
    struct_eff_base = 0.5 * (ribs_struct.ribs_dim - ribs_struct.supp_equiv_thickness)  # NB: this refers to `p`
    crit_angle_y = abs(np.rad2deg(np.atan(struct_eff_base / ribs_struct.supp_heigth)))

    # * if (angle_y < crit_angle_y) the ribs support do not block the photons
    if (abs(angle_y) <= crit_angle_y):
        red_factor_y = correct_erosion_value(mask_thick_proj_y / bin_dim_y, -1.0 * shift_y / bin_dim_y) * bin_dim_y
        sg_y = _erosion(sg_x.T, bin_dim_y, red_factor_y)
    
    # * apply ribs support correction
    else:
        # - upper support correction: induce erosion on following mask element projection (along coarse axis)
        upsupp_erosion_dist = ribs_struct.supp_heigth * np.tan(np.deg2rad(abs(angle_y))) - struct_eff_base
        # since the ribs correction is applied before the mask close elements, `upsupp_erosion_dist` must be
        # concordant with the source displacement (i.e. same sign of mask array shift, opposite to coord)
        upsupp_erosion_dist *= (-1.0 * np.sign(angle_y))
        red_factor_upsupp = correct_erosion_value(upsupp_erosion_dist / bin_dim_y, -1.0 * shift_y / bin_dim_y) * bin_dim_y
        
        # - inferior support correction: increases mask thickness erosion effect
        infsupp_erosion_dist = ribs_struct.supp_heigth - struct_eff_base / np.tan(np.deg2rad(abs(angle_y)))
        tot_proj_erosion = mask_thick_proj_y + np.sign(mask_thick_proj_y) * infsupp_erosion_dist
        red_factor_y = correct_erosion_value(tot_proj_erosion / bin_dim_y, -1.0 * shift_y / bin_dim_y) * bin_dim_y

        # - final sg along y
        sg_y_upcorr = _erosion(sg_x.T, bin_dim_y, red_factor_upsupp)
        sg_y = _erosion(sg_y_upcorr, bin_dim_y, red_factor_y)
    
    return sg_y.T




# -----------------------------------------------------------------------------------------------------------------
# --------------------------                  TESTING FUNCS                  --------------------------------------
# -----------------------------------------------------------------------------------------------------------------

def assert_array_almost_equal(a: NDArray, b: NDArray, atol=1e-7) -> None:
    """Asserts that given arrays are closer than `atol`."""
    np.testing.assert_allclose(a - b, np.zeros_like(a), atol=atol)
    return None

def fshift_arr(arr: NDArray, shift_x: float) -> NDArray:
    """Shifts given array fractionally along columns."""
    return fshift(arr, 0, shift_x)


class TestErosionCorrection(TestCase):
    """Class for testing erosion value correction for effective application."""

    def setUp(self) -> None:
        self.bin_dim = 1.0
        self.arr = np.array(
            [
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            ]
        )
        return None
    
    def test_subpx_erosion_corr(self) -> None:
        """Tests erosion correction for sub-pixel effect."""
        shift = 0.4 / self.bin_dim

        # `1 - shift` sx edge, `shift` dx edge
        shifted_dx = fshift_arr(self.arr, shift)

        # `1 - shift - cut_sx` sx edge
        cut_sx = 0.2 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_sx, shift)
        eroded_sx = _erosion(shifted_dx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.0, 0.4, 1.0, 1.0, 1.0, 0.4, 0.0],
                [0.0, 0.0, 0.4, 1.0, 1.0, 1.0, 0.4, 0.0],
                [0.0, 0.0, 0.4, 1.0, 1.0, 1.0, 0.4, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_sx, expected)

        # `shift - abs(cut_dx)` dx edge
        cut_dx = -0.2 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_dx, shift)
        eroded_dx = _erosion(shifted_dx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.0, 0.6, 1.0, 1.0, 1.0, 0.2, 0.0],
                [0.0, 0.0, 0.6, 1.0, 1.0, 1.0, 0.2, 0.0],
                [0.0, 0.0, 0.6, 1.0, 1.0, 1.0, 0.2, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_dx, expected)


        # `shift` sx edge, `1 - shift` dx edge
        shift = -shift
        shifted_sx = fshift_arr(self.arr, shift)

        # `shift - cut_sx` sx edge
        cut_sx = 0.2 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_sx, shift)
        eroded_sx = _erosion(shifted_sx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.2, 1.0, 1.0, 1.0, 0.6, 0.0, 0.0],
                [0.0, 0.2, 1.0, 1.0, 1.0, 0.6, 0.0, 0.0],
                [0.0, 0.2, 1.0, 1.0, 1.0, 0.6, 0.0, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_sx, expected)

        # `1 - shift - abs(cut_dx)` dx edge
        cut_dx = -0.2 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_dx, shift)
        eroded_dx = _erosion(shifted_sx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.4, 1.0, 1.0, 1.0, 0.4, 0.0, 0.0],
                [0.0, 0.4, 1.0, 1.0, 1.0, 0.4, 0.0, 0.0],
                [0.0, 0.4, 1.0, 1.0, 1.0, 0.4, 0.0, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_dx, expected)

        return None
    
    def test_overpx_decimal_erosion_corr(self) -> None:
        """Tests erosion correction for sub-pixel effect."""
        shift = 0.4 / self.bin_dim

        # `1 - shift` sx edge, `shift` dx edge
        shifted_dx = fshift_arr(self.arr, shift)

        # `1 - shift - cut_sx` sx edge
        cut_sx = 0.7 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_sx, shift)
        eroded_sx = _erosion(shifted_dx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.9, 1.0, 1.0, 0.4, 0.0],
                [0.0, 0.0, 0.0, 0.9, 1.0, 1.0, 0.4, 0.0],
                [0.0, 0.0, 0.0, 0.9, 1.0, 1.0, 0.4, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_sx, expected)

        # `shift - abs(cut_dx)` dx edge
        cut_dx = -0.7 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_dx, shift)
        eroded_dx = _erosion(shifted_dx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.0, 0.6, 1.0, 1.0, 0.7, 0.0, 0.0],
                [0.0, 0.0, 0.6, 1.0, 1.0, 0.7, 0.0, 0.0],
                [0.0, 0.0, 0.6, 1.0, 1.0, 0.7, 0.0, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_dx, expected)


        # `shift` sx edge, `1 - shift` dx edge
        shift = -shift
        shifted_sx = fshift_arr(self.arr, shift)

        # `shift - cut_sx` sx edge
        cut_sx = 0.7 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_sx, shift)
        eroded_sx = _erosion(shifted_sx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.0, 0.7, 1.0, 1.0, 0.6, 0.0, 0.0],
                [0.0, 0.0, 0.7, 1.0, 1.0, 0.6, 0.0, 0.0],
                [0.0, 0.0, 0.7, 1.0, 1.0, 0.6, 0.0, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_sx, expected)

        # `1 - shift - abs(cut_dx)` dx edge
        cut_dx = -0.7 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_dx, shift)
        eroded_dx = _erosion(shifted_sx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.4, 1.0, 1.0, 0.9, 0.0, 0.0, 0.0],
                [0.0, 0.4, 1.0, 1.0, 0.9, 0.0, 0.0, 0.0],
                [0.0, 0.4, 1.0, 1.0, 0.9, 0.0, 0.0, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_dx, expected)

        return None
    
    def test_overpx_erosion_corr1(self) -> None:
        """Tests erosion correction for sub-pixel effect."""
        shift = 0.4 / self.bin_dim

        # `1 - shift` sx edge, `shift` dx edge
        shifted_dx = fshift_arr(self.arr, shift)

        # `1 - shift - cut_sx` sx edge
        cut_sx = 1.7 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_sx, shift)
        eroded_sx = _erosion(shifted_dx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.9, 1.0, 0.4, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.9, 1.0, 0.4, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.9, 1.0, 0.4, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_sx, expected)

        # `shift - abs(cut_dx)` dx edge
        cut_dx = -1.7 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_dx, shift)
        eroded_dx = _erosion(shifted_dx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.0, 0.6, 1.0, 0.7, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.6, 1.0, 0.7, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.6, 1.0, 0.7, 0.0, 0.0, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_dx, expected)


        # `shift` sx edge, `1 - shift` dx edge
        shift = -shift
        shifted_sx = fshift_arr(self.arr, shift)

        # `shift - cut_sx` sx edge
        cut_sx = 1.7 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_sx, shift)
        eroded_sx = _erosion(shifted_sx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.7, 1.0, 0.6, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.7, 1.0, 0.6, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.7, 1.0, 0.6, 0.0, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_sx, expected)

        # `1 - shift - abs(cut_dx)` dx edge
        cut_dx = -1.7 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_dx, shift)
        eroded_dx = _erosion(shifted_sx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.4, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.4, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.4, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_dx, expected)

        return None
    
    def test_overpx_erosion_corr2(self) -> None:
        """Tests erosion correction for sub-pixel effect."""
        shift = 0.4 / self.bin_dim

        # `1 - shift` sx edge, `shift` dx edge
        shifted_dx = fshift_arr(self.arr, shift)

        # `1 - shift - cut_sx` sx edge
        cut_sx = 1.3 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_sx, shift)
        eroded_sx = _erosion(shifted_dx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.3, 1.0, 1.0, 0.4, 0.0],
                [0.0, 0.0, 0.0, 0.3, 1.0, 1.0, 0.4, 0.0],
                [0.0, 0.0, 0.0, 0.3, 1.0, 1.0, 0.4, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_sx, expected)

        # `shift - abs(cut_dx)` dx edge
        cut_dx = -1.3 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_dx, shift)
        eroded_dx = _erosion(shifted_dx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.0, 0.6, 1.0, 1.0, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.6, 1.0, 1.0, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.6, 1.0, 1.0, 0.1, 0.0, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_dx, expected)


        # `shift` sx edge, `1 - shift` dx edge
        shift = -shift
        shifted_sx = fshift_arr(self.arr, shift)

        # `shift - cut_sx` sx edge
        cut_sx = 1.3 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_sx, shift)
        eroded_sx = _erosion(shifted_sx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.0, 0.1, 1.0, 1.0, 0.6, 0.0, 0.0],
                [0.0, 0.0, 0.1, 1.0, 1.0, 0.6, 0.0, 0.0],
                [0.0, 0.0, 0.1, 1.0, 1.0, 0.6, 0.0, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_sx, expected)

        # `1 - shift - abs(cut_dx)` dx edge
        cut_dx = -1.3 / self.bin_dim
        cut_corrected = correct_erosion_value(cut_dx, shift)
        eroded_dx = _erosion(shifted_sx, self.bin_dim, cut_corrected * self.bin_dim)
        expected = np.array(
            [
                [0.0, 0.4, 1.0, 1.0, 0.3, 0.0, 0.0, 0.0],
                [0.0, 0.4, 1.0, 1.0, 0.3, 0.0, 0.0, 0.0],
                [0.0, 0.4, 1.0, 1.0, 0.3, 0.0, 0.0, 0.0],
            ]
        )
        assert_array_almost_equal(eroded_dx, expected)

        return None



class TestMaskVignetting(TestCase):
    """Class for testing mask vignetting with new architecture."""

    def setUp(self):
        BASEPATH: str = '/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data/Simulations'
        #BASEPATH: str = '/mnt/d/PhD_AASS/Coding/Images_fits'
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
            std_vignetted = apply_std_vignetting(self.wfm, self.wfm.mask, sx, sy)
            ribs_vignetted = apply_vignetting(self.wfm, self.wfm.mask, sx, sy, self.ribs_struct)
            assert_array_almost_equal(std_vignetted, ribs_vignetted)
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