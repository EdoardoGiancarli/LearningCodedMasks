"""
@Title: Coded Aperture Imaging for Sky Image Analysis with PyTorch and GPU acceleration
@Author: Edoardo Giancarli
@Date: 23/12/24
@Content:
    - sky_image_simulation(): Simulates the sky image given the sources flux.
    - CodedApertureImagingGPU: Simulates the reconstruction of the sky for a coded mask camera.
"""

import collections.abc as c
import subprocess
import torch
from torchcodedmaskinterface import CodedMaskInterface


def sky_image_simulation(sky_image_shape: tuple[int, int],
                         sources_flux: c.Sequence[int],
                         sources_pos: None | c.Sequence[tuple[int, int]] = None,
                         sky_background_rate: None | int = None,
                         ) -> tuple[torch.tensor, None | c.Sequence]:
    """Simulates the sky image given the sources flux."""

    sky_image = torch.zeros(sky_image_shape)

    # assign fluxes to point-like sources
    for i, pos in enumerate(sources_pos):
        sky_image[pos[0], pos[1]] = sources_flux[i]

    # add sky background
    if sky_background_rate is not None:
        rates = torch.full(sky_image.size(), sky_background_rate).float()
        sky_background = torch.poisson(rates)
        sky_image += sky_background
    else:
        sky_background = None
    
    return sky_image, sky_background


class CodedApertureImagingGPU:
    """Simulates the reconstruction of the sky for a coded mask camera."""

    def __init__(self,
                 pattern_type: str,
                 rank: int,
                 mask_padding: bool = True,
                 use_gpu: bool = False):
        
        print(f"{pattern_type.upper()} mask realization...")
        self.cai = CodedMaskInterface(pattern_type, rank, mask_padding)

        if use_gpu:
            self._check_cuda()
            self._check_gpu_memory()
        else:
            self.gpu = use_gpu
            self.device = torch.device("cpu")
            print("Using CPU...")


    def cai_simulation(self,
                       sources_flux: c.Sequence[int],
                       sky_background_rate: None | int = None,
                       sources_pos: None | c.Sequence[tuple[int, int]] = None,
                       detector_background_rate: None | float = None,
                       print_info: bool = False,
                       ) -> tuple[torch.tensor, torch.tensor, dict]:
        """Simulates the reconstruction of the sky for a coded mask camera."""
        
        print("Begin CAI pipeline...")

        # sky image simulation
        sky_img_shape = self.cai.basic_pattern_shape

        if sources_pos is None:
            sources_pos = [(self._rand(sky_img_shape[0]), self._rand(sky_img_shape[1]))
                           for _ in range(len(sources_flux))]

        sky_image, sky_background = sky_image_simulation(sky_img_shape, sources_flux,
                                                         sources_pos, sky_background_rate)

        # perform sky image reconstruction
        print("Begin CAI reconstruction...")
        detector_image, sky_reconstruction = self._perform_CAI(sky_image, detector_background_rate)
        print("End CAI reconstruction...")

        # source info
        source_info = {
            'sources_pos': sources_pos,
            'sources_transmitted_flux': self.cai.open_fraction*sources_flux,
            'sky_image_fluxes': self._wrap_sources(sky_image, sources_pos),
            'sky_image_shape': self.cai.sky_image_shape,
            'sky_image': sky_image,
            'sky_background': sky_background,
            'reconstructed_fluxes': self._wrap_sources(sky_reconstruction, sources_pos),
            'coded_mask_interface': self.cai,
            'sky_reconstruction_SNR': self.cai.snr(),
            'mask_pattern': self.cai.mask,
            'mask_decoder': self.cai.decoder,
            'mask_PSF': self.cai.psf()}
        
        # print mask info
        if print_info: self._print_info(self.cai)
        print("End CAI pipeline...")

        return detector_image, sky_reconstruction, source_info


    def _check_cuda(self) -> None:

        self.gpu = torch.cuda.is_available()

        if self.gpu:
            self.device = torch.device("cuda:0")
            print("Using GPU...")
        else:
            print("No GPU available, redirecting to CPU...\n")
            user_input = input("Continue on CPU? (y/n): ")

            if user_input.lower() == "n":
                raise Exception("Pipeline interrupted")
            else:
                print("Using CPU...")
                self.device = torch.device("cpu")

    def _check_gpu_memory(self) -> None:
        
        try:
            result = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
            memory_usage_info = ""

            for line in result.splitlines():
                if "MiB / " in line:
                    memory_usage_info = line.strip()
                    break
    
            if memory_usage_info:
                print("-"*(len(line) + len(line)//4 - 4),
                      f"\nGPU Memory Usage: {memory_usage_info}\n"
                      f"{'-'*(len(line) + len(line)//4 - 4)}")
            else:
                print("Unable to find GPU memory usage information.")
        
        except subprocess.CalledProcessError:
            print("Error running nvidia-smi command. Make sure it is installed and accessible.")
    
    def _perform_CAI(self, sky_image, detector_background_rate) -> tuple[torch.tensor, torch.tensor]:
        
        a = [sky_image, self.cai.mask, self.cai.decoder,
             self.cai.detector_image, self.cai.sky_reconstruction]

        with torch.no_grad():
            if self.gpu: self._send_to_gpu(a)
            self.cai.encode(sky_image, detector_background_rate)

            if self.gpu: self._send_to_gpu(a)
            self.cai.decode()
            if self.gpu: torch.cuda.synchronize()

        if self.gpu:
            for tensor in a:
                assert not tensor.is_cuda

        return self.cai.detector_image, self.cai.sky_reconstruction
    
    def _print_info(self, obj) -> None:
        print(f"Mask pattern type: {obj.mask_type.pattern_type}\n"
              f"Basic pattern shape: {obj.basic_pattern_shape}\n"
              f"Mask shape: {obj.mask_shape}\n"
              f"Decoder shape: {obj.decoder_shape}\n"
              f"Detector image shape: {obj.detector_image_shape}\n"
              f"Sky reconstruction image shape: {obj.sky_reconstruction_shape}")

    def _wrap_sources(self, sky, sources_pos) -> torch.tensor:
        sources = torch.tensor([sky[x[0], x[1]] for x in sources_pos])
        return sources
    
    def _rand(self, high) -> int:
        return torch.randint(0, high, (1,)).item()
    
    def _send_to_gpu(self, a) -> None:
        for tensor in a:
            tensor = tensor.to(self.device)
            assert tensor.is_cuda


# end