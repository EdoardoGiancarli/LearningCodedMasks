"""
Module for camera calibration when performing thesky reconstruction wrt source position.
"""

import numpy as np
from tqdm import tqdm

from .skymap import sky_image_simulation   # to update
from .skyrec import sky_encoding, sky_reconstruction
from .skyrec import sky_snr, skyrec_norm


def skyrec_efficiency(depth: int,
                      cam: object,
                      transmit: bool = False,
                      ) -> tuple[np.array, np.array]:
    
    #TODO: optimize: insert mean percentage/snr values for the whole box
    #      -> for now it only computes the rec. counts percentage and SNR on the grid intersections
    #      -> Problems: time consuming since each point of the sky will be reconstructed
    

    n, m = [np.linspace(10, s - 10, depth*s//max(cam.sky_shape), dtype=int) for s in cam.sky_shape]
    y, x = map(len, (n, m))
    counts_map, snr_map = np.zeros((y, x)), np.zeros((y, x))
    print(f"Map resolution (px box dim): {cam.sky_shape[0]//y} x {cam.sky_shape[1]//x}")

    for col in tqdm(range(x - 1)):
        for row in range(y - 1):

            sky_image, *_ = sky_image_simulation(cam.sky_shape, [1e4], [(n[row], m[col])], 1)
            transmitted_photons = sky_image*cam.specs['real_open_fraction'] if transmit else sky_image
            detector = sky_encoding(transmitted_photons, cam)

            skyrec, skyvar = sky_reconstruction(detector, cam)
            skysnr = sky_snr(skyrec, skyvar)
            skyrec, _ = skyrec_norm(skyrec, skyvar, cam)

            counts_map[row, col] = skyrec[*(n[row], m[col])]*100/transmitted_photons[*(n[row], m[col])]
            snr_map[row, col] = skysnr[*(n[row], m[col])]
    
    return counts_map, snr_map


# end