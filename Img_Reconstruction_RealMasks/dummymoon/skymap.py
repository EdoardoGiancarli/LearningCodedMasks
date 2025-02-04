"""
Sky Image simulation...
"""

import warnings
import collections.abc as c
import numpy as np


def sky_image_simulation(sky_image_shape: tuple[int, int],
                         sources_flux: c.Sequence[int],
                         sources_pos: None | c.Sequence[tuple[int, int]] = None,
                         sky_background_rate: None | int = None,
                         ) -> tuple[c.Sequence, c.Sequence, c.Sequence]:
    """Simulates the sky image given the sources flux."""

    # generate sky
    if sky_background_rate is None:
        sky_image = np.zeros(sky_image_shape)
    else:
        sky_image = np.random.poisson(sky_background_rate, sky_image_shape)
    
    sky_background = sky_image.copy()

    if sources_pos is None:
        sources_pos = [(np.random.randint(0, sky_image_shape[0]), np.random.randint(0, sky_image_shape[1]))
                       for _ in range(len(sources_flux))]

    # assign fluxes to point-like sources
    for i, pos in enumerate(sources_pos):
        sky_image[pos[0], pos[1]] = sources_flux[i]
    
    return sky_image, sky_background, sources_pos



def sky_significance(sky_image: np.array,
                     sky_background_rate: float,
                     ) -> np.array:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = np.sqrt(2*(sky_image*np.log(sky_image/sky_background_rate) - (sky_image - sky_background_rate)))

    return np.nan_to_num(s)


# end