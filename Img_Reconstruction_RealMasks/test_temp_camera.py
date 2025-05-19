import unittest
import numpy as np

from mbloodmoon.mask import _bisect_interval
from temp_camera import codedmask


mask_path = "/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data/Simulations/wfm_mask.fits"


from mbloodmoon.mask import decode, encode, psf, variance
class TestWFM(unittest.TestCase):
    def setUp(self):
        self.wfm = codedmask(mask_path, upscale_x=2, upscale_y=1)

    def test_shape_bulk(self):
        self.assertEqual(self.wfm.bulk.shape, self.wfm.shape_detector)

    def test_shape_detector(self):
        self.assertFalse(self.wfm.shape_detector == self.wfm.shape_mask)

    def test_sky_bins(self):
        xbins, ybins = self.wfm._bins_sky()
        assert len(np.unique(xbins)) == len(xbins)
        assert len(np.unique(ybins)) == len(ybins)
        assert len(np.unique(np.round(np.diff(xbins), 7))) == 1
        assert len(np.unique(np.round(np.diff(ybins), 7))) == 1

    def test_encode_shape(self):
        sky = np.zeros(self.wfm.shape_sky)
        self.assertEqual(encode(self.wfm, sky).shape, self.wfm.shape_detector)

    def test_encode_decode(self):
        n, m = self.wfm.shape_sky
        sky = np.zeros((n, m))
        sky[n // 2, m // 2] = 10000
        detector = encode(self.wfm, sky)
        decoded_sky = decode(self.wfm, detector)
        self.assertTrue(np.any(decoded_sky))

    def test_decode_shape(self):
        detector = np.zeros(self.wfm.shape_detector)
        cc = decode(self.wfm, detector)
        var = variance(self.wfm, detector)
        self.assertEqual(cc.shape, self.wfm.shape_sky)
        self.assertEqual(var.shape, self.wfm.shape_sky)

    def test_psf_shape(self):
        self.assertEqual(psf(self.wfm).shape, self.wfm.shape_mask)


class TestCamera(unittest.TestCase):
    """
    Test for the `CodedMaskCamera` class in `mask.py`.

    The test is performed over the typical oversampling that
    will be taken into account for the WFM analyses.
    """
    def setUp(self):
        self.upscale_to = 3
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
                hbin = (mask_bins[b][1] - mask_bins[b][0]) / 2
                np.testing.assert_almost_equal(
                    mask_bins[b] + hbin,
                    sky_bins[b][len(detector_bins[b]) // 2 : -len(detector_bins[b]) // 2 + 2],
                    decimal=7,
                )
                np.testing.assert_almost_equal(
                    detector_bins[b] + hbin,
                    sky_bins[b][len(mask_bins[b]) // 2 : -len(mask_bins[b]) // 2 + 2],
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
                self.assertEqual(wfm.shape_mask[idx], up * self.wfm_base.shape_mask[idx])
                self.assertEqual(wfm.shape_detector[idx], up * self.wfm_base.shape_detector[idx])
                self.assertEqual(
                    wfm.shape_sky[idx],
                    up * (self.wfm_base.shape_mask[idx] + self.wfm_base.shape_detector[idx]) - 1,
                )

    def test_arrays_shape2(self):
        """Test for mask, decoder, bulk and balancing shapes wrt binning."""
        
        print("## Testing arrays shape - 2...")
        for ups_y, ups_x in tuple(
            (i + 1, i + 1) for i in range(self.upscale_to)
        ):
            print(f"# Testing upscaling {ups_y, ups_x}")
            wfm = codedmask(mask_path, ups_x, ups_y)
            self.assertEqual(wfm.mask.shape, wfm.shape_mask)
            self.assertEqual(wfm.decoder.shape, wfm.shape_mask)
            self.assertEqual(wfm.bulk.shape, wfm.shape_detector)
            self.assertEqual(wfm.balancing.shape, wfm.shape_sky)

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








from mbloodmoon.coords import shift2pos, pos2shift

class TestShift2Pos(unittest.TestCase):
    """Test for the `shift2pos()` function in `mask.py`."""

    def setUp(self):
        self.wfm = codedmask(mask_path, upscale_x=3, upscale_y=3)

    def test_binning_boundaries(self):
        """Test for allowed and not allowed shifts wrt the binning."""
        # shifts with "_in" suffix refer to shifts inside binning
        # shifts with "_out" suffix refer to shifts outside binning
        f = 1e-5
        shiftx_sx_in = self.wfm.bins_sky.x[0] + f
        shiftx_sx_out = self.wfm.bins_sky.x[0] - f
        shiftx_dx_in = self.wfm.bins_sky.x[-1] - f
        shiftx_dx_out = self.wfm.bins_sky.x[-1] + f

        shifty_up_in = self.wfm.bins_sky.y[0] + f
        shifty_up_out = self.wfm.bins_sky.y[0] - f
        shifty_bm_in = self.wfm.bins_sky.y[-1] - f
        shifty_bm_out = self.wfm.bins_sky.y[-1] + f

        # test for the allowed shifts at the edges of the binning
        comb_yes = [
            (shiftx_sx_in, shifty_up_in),
            (shiftx_sx_in, shifty_bm_in),
            (shiftx_dx_in, shifty_up_in),
            (shiftx_dx_in, shifty_bm_in),
        ]
        testing = tuple(shift2pos(self.wfm, *shifts) for shifts in comb_yes)

        with self.assertRaises(ValueError):
            # test for the shifts outside the binning
            comb_no = [
                (shiftx_sx_in, shifty_up_out),
                (shiftx_sx_in, shifty_bm_out),
                (shiftx_dx_in, shifty_up_out),
                (shiftx_dx_in, shifty_bm_out),
                (shiftx_sx_out, shifty_up_in),
                (shiftx_dx_out, shifty_up_in),
                (shiftx_sx_out, shifty_bm_in),
                (shiftx_dx_out, shifty_bm_in),
                (shiftx_sx_out, shifty_up_out),
                (shiftx_sx_out, shifty_bm_out),
                (shiftx_dx_out, shifty_up_out),
                (shiftx_dx_out, shifty_bm_out),
            ]
            testing = tuple(shift2pos(self.wfm, *shifts) for shifts in comb_no)


class TestPos2Shift(unittest.TestCase):
    """Test for the `pos2shift()` function in `coords.py`."""

    def setUp(self):
        self.wfm = codedmask(mask_path, upscale_x=3, upscale_y=3)

    def test_p2s_and_s2p_are_inverse(self):
        """
        Tests if computed shifts through `pos2shift()` refer to the
        same pixel indexes obtained with `shift2pos()`.
        """
        n, m = self.wfm.shape_sky
        for _ in range(10000):
            y, x = (np.random.randint(0, n), np.random.randint(0, m))
            self.assertEqual((y, x), shift2pos(self.wfm, *pos2shift(self.wfm, x, y)))

    def test_positive_and_negative_idxs(self):
        """Tests if positive and negative idxs refer to the same shifts."""
        n, m = self.wfm.shape_sky
        in_pos = [
            ((m, n), (-1, -1)),
            ((3 * m // 4, n), (-m // 4 - 1, -1)),
            ((m, 3 * n // 4), (-1, -n // 4 - 1)),
            ((0, 0), (-m - 1, -n - 1)),
        ]
        # `in_pos` contains array positions expressed with positive
        #  idxs and respective negative idxs
        for pos in in_pos:
            self.assertEqual(pos2shift(self.wfm, *pos[0]), pos2shift(self.wfm, *pos[1]))

    def test_idxs_boundaries(self):
        """Test for out-of-bound elements."""
        n, m = self.wfm.shape_sky
        with self.assertRaises(IndexError):
            out_pos = [
                (m + 1, n + 1),
                (-m - 1, n + 1),
                (m + 1, -n - 1),
                (-m - 2, -n - 2),
            ]
            for pos in out_pos:
                pos2shift(self.wfm, *pos)



if __name__ == "__main__":
    unittest.main()
