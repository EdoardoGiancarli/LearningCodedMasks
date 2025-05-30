import numpy as np
import numpy.typing as npt
from mbloodmoon.images import _shift
from mbloodmoon.mask import CodedMaskCamera


def _erosion(
    arr: npt.NDArray,
    step: float,
    cut: float,
) -> npt.NDArray:
    """
    2D matrix erosion for simulating finite thickness effect in shadow projections.
    It takes a mask array and "thins" the mask elements across the columns' direction.
    The erosion is performed only on the correct side of open mask elements:\n
        - right side, if cut is negative (negative angle wrt camera optical axis)
        - left side, if cut is positive (positive angle wrt camera optical axis)
    The function erodes all integer bins (replacing 1s with 0s). If cut is not integer,
    then the function applies a fractional transparency to the last eroded bin.

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
    """
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("Input array must be of integer type.")
    
    # number of bins to cut
    ncuts = int(cut / step)

    # array indexes to cut (completely)
    shifted_int = _shift(arr, (0, ncuts)) if ncuts else arr
    eroded_int = arr * ((arr - shifted_int) > 0)
    
    # number of edge bins to erode
    decimal = abs(cut / step - ncuts)

    # array indexes to be fractionally reduced:
    #   - the bin with the decimal values is the one
    #     to the left or right wrt the cutted bins
    if decimal:
        shifted_frac = _shift(arr, (0, ncuts + np.sign(ncuts)))
        eroded_frac = arr * ((shifted_frac | ~shifted_int) < -1)
    else:
        eroded_frac = arr * 0
    
    return arr * (eroded_int < 1) - eroded_frac * decimal


def apply_vignetting(
    camera: CodedMaskCamera,
    shadowgram: npt.NDArray,
    shift_x: float,
    shift_y: float,
) -> npt.NDArray:
    """
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

    angle_x_rad = np.arctan(shift_x / camera.mdl["detector_topmask_dist"])
    red_factor = camera.mdl["mask_thickness"] * np.tan(angle_x_rad)
    sg1 = _erosion(shadowgram, bins.x[1] - bins.x[0], red_factor)

    angle_y_rad = np.arctan(shift_y / camera.mdl["detector_topmask_dist"])
    red_factor = camera.mdl["mask_thickness"] * np.tan(angle_y_rad)
    sg2 = _erosion(shadowgram.T, bins.y[1] - bins.y[0], red_factor)
    
    return sg1 * sg2.T







import unittest


class TestErosionPositive(unittest.TestCase):

    def setUp(self):
        self.assertArrayAlmostEqual = lambda x, y: np.testing.assert_array_almost_equal(x, y, decimal=2)
        self.erosion_value = lambda cut: 1 - divmod(abs(cut), 1)[1]

    def test_basic_erosion_1(self):
        arr = np.array(
            [
                [1, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 1],
            ]
        )
        step = 0.5
        cut = 0.25
        e = self.erosion_value(cut)
        expected = np.array(
            [
                [e, 1.0, 1.0, 0.0, 0.0, 0.0, e],
                [e, 1.0, 1.0, 0.0, 0.0, 0.0, e],
                [e, 1.0, 1.0, 0.0, 0.0, 0.0, e],
            ]
        )
        result = _erosion(arr, step, cut)
        self.assertArrayAlmostEqual(result, expected)

    #def test_wide_pattern(self):
    #    arr = np.array(
    #        [
    #            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    #            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    #            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    #        ]
    #    )
    #    step = 1.0
    #    cut = 4.5
    #    expected = np.array(
    #        [
    #            [0.0, 0.0, 0.0, 0.0, 0.75, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
    #            [0.0, 0.0, 0.0, 0.0, 0.75, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
    #            [0.0, 0.0, 0.0, 0.0, 0.75, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
    #        ]
    #    )
    #    result = _erosion(arr, step, cut)
    #    self.assertArrayAlmostEqual(result, expected)
#
    #def test_small_step_small_cut(self):
    #    arr = np.array(
    #        [
    #            [0, 0, 1, 0, 0, 0],
    #            [0, 0, 1, 0, 0, 0],
    #            [0, 0, 1, 0, 0, 0],
    #        ]
    #    )
    #    step = 0.5
    #    cut = 0.45
    #    expected = np.array(
    #        [
    #            [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
    #            [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
    #            [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
    #        ]
    #    )
    #    result = _erosion(arr, step, cut)
    #    self.assertArrayAlmostEqual(result, expected)
#
    #def test_double_ones(self):
    #    arr = np.array(
    #        [
    #            [0, 0, 1, 1, 0, 0],
    #            [0, 0, 1, 1, 0, 0],
    #            [0, 0, 1, 1, 0, 0],
    #        ]
    #    )
    #    step = 1.0
    #    cut = 0.5
    #    expected = np.array(
    #        [
    #            [0.0, 0.0, 0.75, 0.75, 0.0, 0.0],
    #            [0.0, 0.0, 0.75, 0.75, 0.0, 0.0],
    #            [0.0, 0.0, 0.75, 0.75, 0.0, 0.0],
    #        ]
    #    )
    #    result = _erosion(arr, step, cut)
    #    self.assertArrayAlmostEqual(result, expected)
#
    #def test_triple_ones_with_large_cut(self):
    #    arr = np.array(
    #        [
    #            [1, 1, 1, 0, 0, 0],
    #            [1, 1, 1, 0, 0, 0],
    #            [1, 1, 1, 0, 0, 0],
    #        ]
    #    )
    #    step = 0.5
    #    cut = 1.0
    #    expected = np.array(
    #        [
    #            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    #            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    #            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    #        ]
    #    )
    #    result = _erosion(arr, step, cut)
    #    self.assertArrayAlmostEqual(result, expected)
#
    #def test_complex_pattern(self):
    #    arr = np.array(
    #        [
    #            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
    #            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
    #            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
    #        ]
    #    )
    #    step = 0.5
    #    cut = 0.49
    #    expected = np.array(
    #        [
    #            [0.0, 0.0, 0.0, 0.51, 1.0, 0.51, 0.0, 0.0, 0.0, 0.51, 1.0, 0.51, 0.0, 0.02, 0.0],
    #            [0.0, 0.0, 0.0, 0.51, 1.0, 0.51, 0.0, 0.0, 0.0, 0.51, 1.0, 0.51, 0.0, 0.02, 0.0],
    #            [0.0, 0.0, 0.0, 0.51, 1.0, 0.51, 0.0, 0.0, 0.0, 0.51, 1.0, 0.51, 0.0, 0.02, 0.0],
    #        ]
    #    )
    #    result = _erosion(arr, step, cut)
    #    self.assertArrayAlmostEqual(result, expected)
#
    #def test_large_pattern_with_large_cut(self):
    #    arr = np.array(
    #        [
    #            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    #            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    #            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    #        ]
    #    )
    #    step = 0.5
    #    cut = 1.2
    #    expected = np.array(
    #        [
    #            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 1.0, 0.8, 0.0, 0.0],
    #            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 1.0, 0.8, 0.0, 0.0],
    #            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 1.0, 0.8, 0.0, 0.0],
    #        ]
    #    )
    #    result = _erosion(arr, step, cut)
    #    self.assertArrayAlmostEqual(result, expected)


class TestErosionNegative(unittest.TestCase):

    def setUp(self):
        self.assertArrayAlmostEqual = lambda x, y: np.testing.assert_array_almost_equal(x, y, decimal=2)
        self.erosion_value = lambda cut: 1 - divmod(abs(cut), 1)[1]
    
    def test_basic_erosion_1(self):
        arr = np.array(
            [
                [1, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 1],
            ]
        )
        step = 0.5
        cut = -0.25
        e = self.erosion_value(cut)
        expected = np.array(
            [
                [1.0, 1.0, e, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, e, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, e, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        result = _erosion(arr, step, cut)
        self.assertArrayAlmostEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
