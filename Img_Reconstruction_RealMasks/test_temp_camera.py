import unittest
import numpy as np

from temp_camera import CodedMaskCamera, codedmask


class TestCamera(unittest.TestCase):
    """
    Test for the `CodedMaskCamera` class in `mask.py`.

    The test is performed over the typical oversampling that
    will be taken into account in the WFM analyses.
    """

    def test_camera(self):
        """
        Test for:
            - binning structure and alignment (mask, detector, sky)
            - arrays shape (mask, detector, sky)
            - arrays shape after upscaling (mask, detector, sky)
            - unique values (mask, decoder, bulk)
        """

        

