import unittest
import numpy as np

from mbloodmoon.mask import _bisect_interval
from temp_camera import codedmask


mask_path = "/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data/Simulations/wfm_mask.fits"

class TestCamera(unittest.TestCase):
    """
    Test for the `CodedMaskCamera` class in `mask.py`.

    The test is performed over the typical oversampling that
    will be taken into account in the WFM analyses.
    """
    def setUp(self):
        self.upscale_to = 10
        self.wfm_base = codedmask(mask_path, 1, 1)

    def test_binning_alignment(self):
        """Test for binning steps and superimposition."""

        print("## Testing binning alignment...")
        for ups_y, ups_x in tuple(
            (i + 1, i + 1) for i in range(self.upscale_to)
        ):
            print(f"# Testing upscaling {ups_y, ups_x}")
            wfm = codedmask(mask_path, ups_x, ups_y)
            mask_bins = wfm.bins_mask
            detector_bins = wfm.bins_detector
            sky_bins = wfm.bins_sky

            for ax, b, up in zip(
                ("y", "x"), (1, 0), (ups_y, ups_x),
            ):
                print(f"    - testing {ax} axis")
                # testing edges
                np.testing.assert_equal(
                    np.array((mask_bins[b][0], mask_bins[b][-1])),
                    np.array((wfm.specs["mask_min" + ax], wfm.specs["mask_max" + ax])),
                    strict=False,
                )
                np.testing.assert_almost_equal(
                    np.array((detector_bins[b][0], detector_bins[b][-1])),
                    np.array((self.wfm_base.bins_detector[b][0], self.wfm_base.bins_detector[b][-1])),
                    decimal=7,
                )

                # testing bins step
                step = wfm.specs["mask_delta" + ax]
                self.assertAlmostEqual(
                    up * (mask_bins[b][1] - mask_bins[b][0]), step, places=7,
                )
                self.assertAlmostEqual(
                    up * (detector_bins[b][1] - detector_bins[b][0]), step, places=7,
                )
                self.assertAlmostEqual(
                    up * (sky_bins[b][1] - sky_bins[b][0]), step, places=7,
                )

                # testing superimposition
                np.testing.assert_almost_equal(
                    mask_bins[b],
                    sky_bins[b][len(detector_bins[b]) // 2 - 1 : -len(detector_bins[b]) // 2 + 1],
                    decimal=7,
                )
                np.testing.assert_almost_equal(
                    detector_bins[b],
                    sky_bins[b][len(mask_bins[b]) // 2 - 1 : -len(mask_bins[b]) // 2 + 1],
                    decimal=7,
                )

    def test_arrays_shape1(self):
        """Test for mask, detector and sky shapes after upscaling."""

        print("## Testing arrays shape - 1...")
        for ups_y, ups_x in tuple(
            (i + 1, i + 1) for i in range(self.upscale_to)
        ):
            print(f"# Testing upscaling {ups_y, ups_x}")
            wfm = codedmask(mask_path, ups_x, ups_y)
            
            for (idx, ax), up in zip(
                enumerate(("y", "x")), (ups_y, ups_x),
            ):
                print(f"    - testing {ax} axis")
                self.assertEqual(wfm.mask_shape[idx], up * self.wfm_base.mask_shape[idx])
                self.assertEqual(wfm.detector_shape[idx], up * self.wfm_base.detector_shape[idx])
                self.assertEqual(
                    wfm.sky_shape[idx],
                    up * (self.wfm_base.mask_shape[idx] + self.wfm_base.detector_shape[idx]) - 1,
                )

    def test_arrays_shape2(self):
        """Test for mask, decoder, bulk and balancing shapes wrt binning."""
        
        print("## Testing arrays shape - 2...")
        for ups_y, ups_x in tuple(
            (i + 1, i + 1) for i in range(self.upscale_to)
        ):
            print(f"# Testing upscaling {ups_y, ups_x}")
            wfm = codedmask(mask_path, ups_x, ups_y)
            self.assertEqual(wfm.mask.shape, wfm.mask_shape)
            self.assertEqual(wfm.decoder.shape, wfm.mask_shape)
            self.assertEqual(wfm.bulk.shape, wfm.detector_shape)
            self.assertEqual(wfm.balancing.shape, wfm.sky_shape)

    def test_bulk_binning(self):
        """
        Test for detector binning and bulk binning ('static' vs 'dynamic').

        Notes:
            - 'static' binning meaning that the edges of the detector binning
              are fixed wrt the base binning (i.e. upscaling = (1, 1)), so that
              after the upscaling the length of the structure is self-consistent.
            - 'dynamic' binning meaning that the edges of the detector binning
              are NOT fixed wrt the base binning. In this case, after an upsampling,
              the new structure is not an int multiple of the base binning, since
              this kind of structure converges to the detector physical edges.

        """
        print("## Testing bulk binning...")
        for ups_y, ups_x in tuple(
            (i + 1, i + 1) for i in range(self.upscale_to)
        ):
            print(f"# Testing upscaling {ups_y, ups_x}")
            wfm = codedmask(mask_path, ups_x, ups_y)

            # static detector/bulk binning
            base_bins = self.wfm_base.bins_mask
            xmin, _ = _bisect_interval(base_bins.x, wfm.specs["detector_minx"], wfm.specs["detector_maxx"])
            ymin, _ = _bisect_interval(base_bins.y, wfm.specs["detector_miny"], wfm.specs["detector_maxy"])
            
            # dynamic detector/bulk binning
            test_xmin, test_xmax = _bisect_interval(wfm.bins_mask.x, wfm.specs["detector_minx"], wfm.specs["detector_maxx"])
            test_ymin, test_ymax = _bisect_interval(wfm.bins_mask.y, wfm.specs["detector_miny"], wfm.specs["detector_maxy"])

            n_zero_resp_pxs_x = int(
                np.abs((base_bins.x[xmin] - wfm.specs["detector_minx"]) * ups_x / wfm.specs["mask_deltax"])
            )
            n_zero_resp_pxs_y = int(
                np.abs((base_bins.y[ymin] - wfm.specs["detector_miny"]) * ups_y / wfm.specs["mask_deltay"])
            )

            self.assertEqual(
                2 * n_zero_resp_pxs_y,
                len(wfm.bins_detector.y) - len(wfm.bins_mask.y[test_ymin : test_ymax + 1]),
            )
            self.assertEqual(
                2 * n_zero_resp_pxs_x,
                len(wfm.bins_detector.x) - len(wfm.bins_mask.x[test_xmin : test_xmax + 1]),
            )
    
    def test_array_values(self):
        """Tests the unique values in the mask, decoder and bulk."""
        print("## Testing unique values...")
        for ups_y, ups_x in tuple(
            (i + 1, i + 1) for i in range(self.upscale_to)
        ):
            print(f"# Testing upscaling {ups_y, ups_x}")
            wfm = codedmask(mask_path, ups_x, ups_y)

            np.testing.assert_equal(
                np.unique(wfm.mask),
                np.array([0, 1]),
                strict=False,
            )
            np.testing.assert_almost_equal(
                np.unique(wfm.decoder),
                np.array([-wfm.specs["real_open_fraction"] / (1 - wfm.specs["real_open_fraction"]), 0, 1]),
                decimal=7,
            )
            np.testing.assert_equal(
                np.unique(wfm.bulk),
                np.array([0, 1]),
                strict=False,
            )


if __name__ == "__main__":
    unittest.main()
