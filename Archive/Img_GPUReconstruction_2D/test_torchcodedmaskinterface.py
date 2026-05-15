"""
@Title: Test for URA/MURA Coded Mask Interface with PyTorch
@Author: Edoardo Giancarli
@Date: 13/12/24
@Content:
    - TestCodedMaskInterface: Tests the CodedMaskInterface class in torchcodedmaskinterface.py.
"""

import unittest
from unittest import TestCase
import torch
from torchcodedmaskinterface import CodedMaskInterface
from torchmaskpattern import URAMaskPattern, MURAMaskPattern


class TestCodedMaskInterface(TestCase):
    """Tests the CodedMaskInterface class in torchcodedmaskinterface.py."""

    def setUp(self):
        self.cmi_ura = CodedMaskInterface('ura', 0, False)
        self.cmi_mura = CodedMaskInterface('mura', 0, False)
    

    def test_initialization(self):
        self.assertEqual(self.cmi_ura.mask_type.pattern_type, "URA")
        self.assertEqual(self.cmi_mura.mask_type.pattern_type, "MURA")

    def test_get_mask_type(self):
        with self.assertRaises(ValueError):
            self.assertEqual(self.cmi_ura._get_mask_type('ura', -2))
            self.assertEqual(self.cmi_ura._get_mask_type('mura', -2))
    
    def test_get_mask_pattern(self):
        self.assertTrue(torch.all(self.cmi_ura.mask == URAMaskPattern(0).basic_pattern))
        self.assertTrue(0 < self.cmi_ura.open_fraction and self.cmi_ura.open_fraction < 1)
        self.assertTrue(torch.all(self.cmi_mura.mask == MURAMaskPattern(0).basic_pattern))
        self.assertTrue(0 < self.cmi_mura.open_fraction and self.cmi_mura.open_fraction < 1)
        self.assertTrue(torch.all(self.cmi_ura.decoder == URAMaskPattern(0).basic_decoder))
        self.assertTrue(torch.all(self.cmi_mura.decoder == MURAMaskPattern(0).basic_decoder))
    
    def test_cmi_properties(self):
        self.assertTrue(torch.all(self.cmi_ura.basic_pattern == URAMaskPattern(0).basic_pattern))
        self.assertTrue(torch.all(self.cmi_mura.basic_pattern == MURAMaskPattern(0).basic_pattern))

        test_attr = ["basic_pattern", "basic_pattern_shape", "mask_shape",
                     "basic_decoder", "decoder_shape", "sky_image_shape",
                     "detector_image_shape", "sky_reconstruction_shape"]
        
        _ = [getattr(self.cmi_ura, attr) for attr in test_attr[:5]]
        _ = [getattr(self.cmi_mura, attr) for attr in test_attr[:5]]

        with self.assertRaises(AttributeError):
            _ = [getattr(self.cmi_ura, attr) for attr in test_attr[-4:]]
            _ = [getattr(self.cmi_mura, attr) for attr in test_attr[-4:]]

    def test_psf(self):
        urapsf = self.cmi_ura.psf()
        murapsf = self.cmi_mura.psf()
        self.assertTrue(torch.abs(urapsf.sum() - 1) < 1e-6)
        self.assertTrue(torch.abs(murapsf.sum() - 1) < 1e-6)

    def test_SNR(self):
        with self.assertRaises(AssertionError):
            self.cmi_ura.snr()
            self.cmi_mura.snr()
    
    def test_cai(self):
        ura_cmi_dummy = CodedMaskInterface('ura', 0)
        mura_cmi_dummy = CodedMaskInterface('mura', 0)
        S_ura = torch.randint(1, 11, (5, 3))
        S_mura = torch.randint(1, 11, (5, 5))
        # encoding
        D_ura = ura_cmi_dummy.encode(S_ura)
        D_mura = mura_cmi_dummy.encode(S_mura)
        # decoding
        S_hat_ura = ura_cmi_dummy.decode()
        S_hat_mura = mura_cmi_dummy.decode()
        # sky and detector images shapes
        self.assertEqual(D_ura.size(), S_ura.size())
        self.assertEqual(D_mura.size(), S_mura.size())
        self.assertEqual(S_hat_ura.size(), S_ura.size())
        self.assertEqual(S_hat_mura.size(), S_mura.size())
        # SNR
        ura_cmi_dummy.snr()
        mura_cmi_dummy.snr()
        # attributes
        test_attr = ["sky_image_shape", "detector_image_shape", "sky_reconstruction_shape"]
        _ = [getattr(ura_cmi_dummy, attr) for attr in test_attr]
        _ = [getattr(mura_cmi_dummy, attr) for attr in test_attr]
    
    def test_get_mask_pattern_padding(self):
        # test padded mask shape
        ura_cmi_dummy = CodedMaskInterface('ura', 2, True)
        mura_cmi_dummy = CodedMaskInterface('mura', 2, True)
        self.assertEqual(ura_cmi_dummy.basic_pattern_shape, (13, 11))
        self.assertEqual(ura_cmi_dummy.mask_shape, (25, 21))
        self.assertTrue(0 < ura_cmi_dummy.open_fraction and ura_cmi_dummy.open_fraction < 1)
        self.assertEqual(mura_cmi_dummy.basic_pattern_shape, (17, 17))
        self.assertEqual(mura_cmi_dummy.mask_shape, (33, 33))
        self.assertTrue(0 < mura_cmi_dummy.open_fraction and mura_cmi_dummy.open_fraction < 1)
        # test padded mask content
        ura_cmi_dummy2 = CodedMaskInterface('ura', 0, True)
        self.assertTrue(torch.all(ura_cmi_dummy2.mask == self._get_padded_mask()))
    
    def test_cai_padding(self):
        ura_cmi_dummy = CodedMaskInterface('ura', 2, True)
        mura_cmi_dummy = CodedMaskInterface('mura', 2, True)
        S_ura = torch.randint(1, 11, (13, 11))
        S_mura = torch.randint(1, 11, (17, 17))
        # encoding
        D_ura = ura_cmi_dummy.encode(S_ura)
        D_mura = mura_cmi_dummy.encode(S_mura)
        # decoding
        S_hat_ura = ura_cmi_dummy.decode()
        S_hat_mura = mura_cmi_dummy.decode()
        # sky and detector images shapes
        self.assertEqual(D_ura.size(), S_ura.size())
        self.assertEqual(D_mura.size(), S_mura.size())
        self.assertEqual(S_hat_ura.size(), S_ura.size())
        self.assertEqual(S_hat_mura.size(), S_mura.size())
        # SNR
        ura_cmi_dummy.snr()
        mura_cmi_dummy.snr()
        # attributes
        test_attr = ["sky_image_shape", "detector_image_shape", "sky_reconstruction_shape"]
        _ = [getattr(ura_cmi_dummy, attr) for attr in test_attr]
        _ = [getattr(mura_cmi_dummy, attr) for attr in test_attr]

    def _get_padded_mask(self):
        padded_mask = torch.zeros((9, 5))
        a, b = torch.tensor([1, 1, 0, 1, 1]), torch.tensor([0, 1, 1, 0, 1])
        for i in range(9):
            if i in [0, 4, 5]: padded_mask[i, :] = a
            elif i in [1, 3, 6, 8]: padded_mask[i, :] = b
        return padded_mask




if __name__ == "__main__":
    unittest.main()


# end