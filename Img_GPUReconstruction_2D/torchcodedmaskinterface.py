"""
@Title: URA/MURA Coded Mask Interface with PyTorch
@Author: Edoardo Giancarli
@Date: 19/12/24
@Content:
    - CodedMaskInterface: Interface for the URA/MURA coded mask camera analysis.
"""

import collections.abc as c
import torch
from torch.nn.functional import conv2d
from torchmaskpattern import URAMaskPattern, MURAMaskPattern


class CodedMaskInterface:
    """Interface for the URA/MURA coded mask camera analysis."""

    def __init__(self,
                 pattern_type: str,
                 rank: int,
                 padding: bool = True):
        
        self.padding = padding
        self.mask_type = self._get_mask_type(pattern_type, rank)
        self.mask, self.decoder, self.open_fraction = self._get_mask_pattern(self.padding)

        self.detector_image = torch.zeros(*self.basic_pattern_shape)
        self.sky_reconstruction = torch.zeros(*self.basic_pattern_shape)
    
    def __getattr__(self, name):
        attribute_map = {"basic_pattern": lambda: self.mask_type.basic_pattern,
                         "basic_pattern_shape": lambda: self.mask_type.basic_pattern.size(),
                         "mask_shape": lambda: self.mask.size(),
                         "basic_decoder": lambda: self.mask_type.basic_decoder,
                         "decoder_shape": lambda: self.decoder.size(),
                         "sky_image_shape": lambda: self.sky_image.size(),
                         "detector_image_shape": lambda: self.detector_image.size(),
                         "sky_reconstruction_shape": lambda: self.sky_reconstruction.size()}
        
        if name not in attribute_map:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        else:
            return attribute_map[name]()
    

    def encode(self,
               sky_image: torch.tensor,
               detector_background_rate: None | float = None) -> torch.tensor:
        """Returns the detector image from the simulated sky image."""

        self.sky_image = sky_image
        assert self.sky_image_shape == self.basic_pattern_shape

        if self.padding:
            self.detector_image = self._correlation(self.mask, self.sky_image)
        else:
            zero_pad_mask = self._get_padded_tensor(self.basic_pattern)
            self.detector_image = self._correlation(zero_pad_mask, self.sky_image)

        if detector_background_rate:
            rates = torch.full(self.detector_image_shape, detector_background_rate).float()
            self.detector_image += torch.poisson(rates)
        
        assert self.detector_image_shape == self.basic_pattern_shape
        
        return self.detector_image


    def decode(self) -> torch.tensor:
        """Returns the reconstructed sky image from the detector image."""
        
        if self.padding:
            self.sky_reconstruction = self._correlation(self.decoder, self.detector_image)
        else:
            zero_pad_decoder = self._get_padded_tensor(self.decoder)
            self.sky_reconstruction = self._correlation(zero_pad_decoder, self.detector_image)
        
        assert self.sky_reconstruction_shape == self.sky_image_shape 

        return self.sky_reconstruction
    

    def psf(self) -> torch.tensor:
        """Returns the mask PSF."""

        pad_decoder = self.decoder if self.padding else self._get_padded_tensor(self.decoder, mode='wrap')
        torch_psf = self._correlation(pad_decoder, self.basic_pattern)
        
        assert torch_psf.size() == self.basic_pattern_shape

        return torch_psf
    

    def snr(self) -> torch.tensor:
        """Returns the SNR for the reconstructed image."""

        assert torch.abs(self.sky_reconstruction.sum()) > 0

        return self.sky_reconstruction/torch.sqrt(torch.abs(self.sky_reconstruction.sum()))
    

    def _get_mask_type(self, pattern_type, rank) -> c.Callable:

        if pattern_type not in ['ura', 'mura']:
            raise ValueError(f"Invalid pattern_type = {pattern_type}, must be 'ura' or 'mura'.")
        elif pattern_type == 'ura':
            mask_pattern = URAMaskPattern(rank)
        else:
            mask_pattern = MURAMaskPattern(rank)
        
        assert mask_pattern.pattern_type == pattern_type.upper()
        
        return mask_pattern

    def _get_mask_pattern(self, padding) -> tuple[torch.tensor, torch.tensor, float]:

        if padding:
            mask = self._get_padded_tensor(self.basic_pattern, mode='wrap')
            decoder = self._get_padded_tensor(self.basic_decoder, mode='wrap')
        else:
            mask = self.basic_pattern
            decoder = self.basic_decoder
            
        open_fraction = mask.sum()/torch.prod(torch.tensor(mask.size()))
        
        return mask, decoder, open_fraction
    
    def _get_padded_tensor(self, tensor, mode='zero') -> torch.tensor:

        n, m = tensor.shape
        pad_n, pad_m = (n - 1)//2, (m - 1)//2

        if mode not in ['zero', 'wrap']:
            raise ValueError(f"invalid mode {mode}. Mode must be 'zero' or 'wrap'.")
        
        elif mode == 'zero':
            pad_mode = torch.nn.ConstantPad2d((pad_m, pad_m, pad_n, pad_n), 0)
            padded_tensor = pad_mode(tensor.reshape(1, 1, *tensor.size()))
        
        else:
            pad_mode = torch.nn.CircularPad2d((pad_m, pad_m, pad_n, pad_n))
            padded_tensor = pad_mode(tensor.reshape(1, 1, *tensor.size()))
        
        assert padded_tensor.squeeze(0, 1).size() == (2*n - 1, 2*m - 1)

        return padded_tensor.squeeze(0, 1)
    
    def _correlation(self,
                     A: torch.tensor,
                     B: torch.tensor) -> torch.tensor:
        
        C = conv2d(A.reshape(1, 1, *A.size()).float(), B.reshape(1, 1, *B.size()).float())

        return C.squeeze(0, 1)


# end