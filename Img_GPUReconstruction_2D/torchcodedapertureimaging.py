"""
@Title: Coded Aperture Imaging for Sky Image Analysis with PyTorch
@Author: Edoardo Giancarli
@Date: 23/12/24
@Content:
    - sky_image_simulation(): Simulates the sky image given the sources flux.
    - cai_simulation(): Simulates the reconstruction of the sky for a coded mask camera.
"""

import collections.abc as c
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


def cai_simulation(pattern_type: str,
                   rank: int,
                   sources_flux: c.Sequence[int],
                   mask_padding: bool = False,
                   sky_background_rate: None | int = None,
                   sources_pos: None | c.Sequence[tuple[int, int]] = None,
                   detector_background_rate: None | float = None,
                   print_info: bool = False,
                   ) -> tuple[torch.tensor, torch.tensor, dict]:
    """Simulates the reconstruction of the sky for a coded mask camera."""
    
    # mask initialization
    cai = CodedMaskInterface(pattern_type, rank, mask_padding)
    sky_img_shape = cai.basic_pattern_shape

    # sky image simulation
    if sources_pos is None:
        sources_pos = [(torch.randint(0, sky_img_shape[0], (1,)).item(), torch.randint(0, sky_img_shape[1], (1,)).item())
                       for _ in range(len(sources_flux))]

    sky_image, sky_background = sky_image_simulation(sky_img_shape, sources_flux,
                                                     sources_pos, sky_background_rate)

    # sky image reconstruction
    detector_image = cai.encode(sky_image, detector_background_rate)
    sky_reconstruction = cai.decode()

    # source info
    source_info = {
        'sources_pos': sources_pos,
        'sources_transmitted_flux': cai.open_fraction*sources_flux,
        'sky_image_fluxes': _wrap_sources(sky_image, sources_pos),
        'sky_image_shape': cai.sky_image_shape,
        'sky_image': sky_image,
        'sky_background': sky_background,
        'reconstructed_fluxes': _wrap_sources(sky_reconstruction, sources_pos),
        'coded_mask_interface': cai,
        'sky_reconstruction_SNR': cai.snr(),
        'mask_pattern': cai.mask,
        'mask_decoder': cai.decoder,
        'mask_PSF': cai.psf()}
    
    # print mask info
    if print_info: _print_info(cai)

    return detector_image, sky_reconstruction, source_info


def _print_info(obj) -> None:
    print(f"Mask pattern type: {obj.mask_type.pattern_type}\n"
          f"Basic pattern shape: {obj.basic_pattern_shape}\n"
          f"Mask shape: {obj.mask_shape}\n"
          f"Decoder shape: {obj.decoder_shape}\n"
          f"Detector image shape: {obj.detector_image_shape}\n"
          f"Sky reconstruction image shape: {obj.sky_reconstruction_shape}")

def _wrap_sources(sky, sources_pos) -> torch.tensor:
    sources = torch.tensor([sky[x[0], x[1]] for x in sources_pos])
    return sources


# end