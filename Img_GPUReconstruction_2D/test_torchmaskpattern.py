"""
@Title: Test for URA/MURA Coded Mask Pattern Simulation with PyTorch
@Author: Edoardo Giancarli
@Date: 19/12/24
@Content:
    - TestURAMaskPattern: Tests the URAMaskPattern class in torchmaskpattern.py.
    - TestMURAMaskPattern: Tests the MURAMaskPattern class in torchmaskpattern.py.
"""

import unittest
from unittest import TestCase
import torch
from torchmaskpattern import URAMaskPattern, MURAMaskPattern


class TestURAMaskPattern(TestCase):
    """Tests the URAMaskPattern class in torchmaskpattern.py."""
    
    def setUp(self):
        self.ura = URAMaskPattern(rank=0)
    
    def test_torch_implementation(self):
        self.assertTrue(torch.is_tensor(self.ura.basic_pattern))
        self.assertTrue(torch.is_tensor(self.ura.basic_decoder))
    
    def test_initialization(self):
        self.assertEqual(self.ura.pattern_type, "URA")
        self.assertEqual(self.ura.rank, 0)
        self.assertEqual(self.ura.prime_pair, (5, 3))
    
    def test_get_prime_pair(self):
        self.assertEqual(self.ura._get_prime_pair(0), (5, 3))
        self.assertEqual(self.ura._get_prime_pair(1), (7, 5))
        self.assertEqual(self.ura._get_prime_pair(2), (13, 11))

        with self.assertRaises(ValueError):
            URAMaskPattern(rank=-4)
            self.ura._get_prime_pair(-3)
            self.ura._get_prime_pair(100)
    
    def test_get_pattern_root(self):
        C_r_i, C_s_j = self.ura._get_pattern_root()
        C_r_i2, C_s_j2 = URAMaskPattern(rank=2)._get_pattern_root()

        self.assertTrue(torch.all(C_r_i == torch.Tensor([-1, 1, -1, -1, 1])))
        self.assertTrue(torch.all(C_s_j == torch.Tensor([-1, 1, -1])))

        self.assertTrue(torch.all(C_r_i2 == torch.Tensor([-1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1])))
        self.assertTrue(torch.all(C_s_j2 == torch.Tensor([-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1])))
    
    def test_basic_pattern(self):
        self.assertEqual(self.ura.basic_pattern.shape, (5, 3))
        self.assertEqual(self.ura.basic_pattern[0, 0], 0)
        self.assertEqual(self.ura.basic_pattern[1, 1], 1)

        basic_pattern_rank = torch.linalg.matrix_rank(self.ura.basic_pattern)
        self.assertEqual(basic_pattern_rank, 2,
                         "The basic pattern matrix is not URA.")
        
        self.assertEqual(self.ura.basic_decoder.shape, self.ura.basic_pattern.shape)
        self.assertEqual(self.ura.basic_decoder[0, 0], -1/self.ura.basic_pattern.sum())
        self.assertEqual(self.ura.basic_decoder[1, 1], 1/self.ura.basic_pattern.sum())
        decoder_rank = torch.linalg.matrix_rank(self.ura.basic_decoder)
        self.assertEqual(decoder_rank, 3)




class TestMURAMaskPattern(TestCase):
    """Tests the MURAMaskPattern class in torchmaskpattern.py."""

    def setUp(self):
        self.mura = MURAMaskPattern(rank=0)

    def test_torch_implementation(self):
        self.assertTrue(torch.is_tensor(self.mura.basic_pattern))
        self.assertTrue(torch.is_tensor(self.mura.basic_decoder))
    
    def test_initialization(self):
        self.assertEqual(self.mura.pattern_type, "MURA")
        self.assertEqual(self.mura.rank, 0)
        self.assertEqual(self.mura.l, 5)
    
    def test_get_prime(self):
        self.assertEqual(self.mura._get_prime(0), 5)
        self.assertEqual(self.mura._get_prime(1), 13)
        self.assertEqual(self.mura._get_prime(2), 17)

        with self.assertRaises(ValueError):
            MURAMaskPattern(rank=-4)
            self.mura._get_prime(-3)
    
    def test_get_pattern_root(self):
        C_r_i, C_s_j = self.mura._get_pattern_root()
        C_r_i2, C_s_j2 = MURAMaskPattern(rank=2)._get_pattern_root()

        self.assertTrue(torch.all(C_r_i == C_s_j))
        self.assertTrue(torch.all(C_s_j == torch.Tensor([-1, 1, -1, -1, 1])))

        self.assertTrue(torch.all(C_r_i2 == C_s_j2))
        self.assertTrue(torch.all(C_s_j2 == torch.Tensor([-1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1])))
    
    def test_basic_pattern(self):
        self.assertEqual(self.mura.basic_pattern.shape, (5, 5))
        self.assertEqual(self.mura.basic_pattern[0, 0], 0)
        self.assertEqual(self.mura.basic_pattern[1, 1], 1)

        basic_pattern_rank = torch.linalg.matrix_rank(self.mura.basic_pattern)
        self.assertEqual(basic_pattern_rank, 2,
                         "The basic pattern matrix is not MURA.")
        
        self.assertEqual(self.mura.basic_decoder.shape, self.mura.basic_pattern.shape)
        self.assertEqual(self.mura.basic_decoder[0, 0], 1/self.mura.basic_pattern.sum())
        self.assertEqual(self.mura.basic_decoder[1, 1], 1/self.mura.basic_pattern.sum())
        decoder_rank = torch.linalg.matrix_rank(self.mura.basic_decoder)
        self.assertEqual(decoder_rank, 3)




if __name__ == "__main__":
    unittest.main()


# end