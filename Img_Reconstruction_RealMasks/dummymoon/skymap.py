"""
Sky Image simulation...
"""

import warnings
import collections.abc as c
import numpy as np

from bisect import bisect
from .skyrec import sky_encoding


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


def skymap_simulation(
        exposure: int | float,
        sources_pos: None | c.Sequence[tuple[float, float]],
        sources_rates: c.Sequence[float],
        skybg_rate: None | float,
        cam: object,
        ) -> tuple[np.array, np.array, np.array]:
    """
    Sky simulation given the exposure and the sources.
    --------------------------------------------------------
    Notes:
        - the exposure is in [s]
        - the sources positions are expressed as the angles
          wrt the camera on-axis in the interval [-45°, 45°]
        - the sources are initialized through the "observed"
          photons emission rate (point-like sources)
    """

    x_max = cam.specs["mask_maxx"] + cam.specs["detector_maxx"]
    y_max = cam.specs["mask_maxy"] + cam.specs["detector_maxy"]
    thetax_max = np.arctan(x_max/cam.specs["mask_detector_distance"])
    thetay_max = np.arctan(y_max/cam.specs["mask_detector_distance"])

    def _random_pos() -> c.Sequence[tuple[float, float]]:
        random_pos = [
            (np.random.uniform(-0.99*thetay_max, thetay_max), np.random.uniform(-0.99*thetax_max, thetax_max))
            for _ in range(len(sources_rates))
        ]
        return random_pos

    def _check_angles() -> None:
        rad_coords = [(np.deg2rad(y), np.deg2rad(x)) for (y, x) in sources_pos]
        for idx, pos in enumerate(rad_coords):
            if not (np.abs(pos[0]) < thetay_max and np.abs(pos[1]) < thetax_max):
                raise ValueError(f"Invalid coords values ({pos[0]}, {pos[1]}) at idx {idx}.")
        return rad_coords

    def _angles2pxs(theta_y: float,
                    theta_x: float,
                    ) -> tuple[int, int]:
        shift_y = cam.specs["mask_detector_distance"]*np.tan(theta_y)
        shift_x = cam.specs["mask_detector_distance"]*np.tan(theta_x)
        row = bisect(cam.bins_sky.y, shift_y) - 1
        col = bisect(cam.bins_sky.x, shift_x) - 1
        return row, col
    
    def source_counts(rate: int | float) -> float:
        return rate*exposure

    def sources_shadowgram(sky: np.array) -> np.array:
        return sky_encoding(sky, cam)

    def skybg_shadowgram(rate: int | float) -> np.array:
        bg = np.random.poisson(rate, cam.sky_shape)
        return sky_encoding(bg, cam)

    # generate background counts
    if skybg_rate is None:
        bg_det = np.zeros(cam.detector_shape)
    else:
        bg_det = skybg_shadowgram(skybg_rate)
    
    # random pos (already in [rad]) or check pos
    if sources_pos is not None:
        sources_pos = _check_angles()
    else:
        sources_pos = _random_pos()
    
    # assign fluxes to point-like sources
    sky = np.zeros(cam.sky_shape)
    sources_pxspos = [_angles2pxs(*p) for p in sources_pos]
    for i, pos in enumerate(sources_pxspos):
        sky[*pos] = source_counts(sources_rates[i])
    
    detector = (bg_det + sources_shadowgram(sky))*cam.specs["real_open_fraction"]
    
    return detector, sources_pxspos, sky


# end