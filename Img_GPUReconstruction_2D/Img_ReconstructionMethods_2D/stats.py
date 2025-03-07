"""
@Title: Stats methods
@Author: Edoardo Giancarli
@Date: 31/12/24
@Content:
    - variance: computes the variance of the reconstructed sky
    - snr: computes the SNR of the reconstructed sky
    - significance: computes the significance of the counts in the reconstructed sky
    - enhance_skyrec_slices: shows the x-slice and y-slice of the reconstructed sources
@References:
    [1] A. Goldwurm and A. Gros, "Coded Mask Instruments for Gamma-Ray Astronomy", 2023
        (in "Handbook of X-ray and Gamma-ray Astrophysics", Springer 2023)
    [2] Notebook by Peppe (url: https://github.com/peppedilillo/masks/blob/main/notebooks/11_pcfov.ipynb)
"""


import collections.abc as c
import warnings

import numpy as np
from scipy.signal import correlate
import plot_module as plot


def variance(decoder, detector_image):
    return correlate(decoder**2, detector_image)



def snr(sky_reconstruction, sky_reconstruction_var):
    return sky_reconstruction/np.sqrt(sky_reconstruction_var)



def significance(sky_image,
                 sky_background_rate):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = np.sqrt(2*(sky_image*np.log(sky_image/sky_background_rate) - (sky_image - sky_background_rate)))

    return np.nan_to_num(s)



def enhance_skyrec_slices(sky_reconstruction, sources_pos):
    u, v = sky_reconstruction.shape
    center = (u//2, v//2)
    pos_wrt_center = [(pos[0] - center[0], pos[1] - center[1]) for _, pos in enumerate(sources_pos)]
    
    n, m = (u + 2)//3, (v + 2)//3    # FCFOV shape

    for idx, pos in enumerate(sources_pos):
        S_hat_slicex = sky_reconstruction[pos[0], :]
        S_hat_slicey = sky_reconstruction[:, pos[1]]

        if (np.abs(pos_wrt_center[idx][0]) < n//2) and (np.abs(pos_wrt_center[idx][1]) < m//2):
            zone = " (FCFOV)"
        else:
            zone = " (PCFOV)"

        plot.sequence_plot([S_hat_slicex, S_hat_slicey],
                           [f"$\\hat{{S}}_{idx}$ x-axis slice" + zone, f"$\\hat{{S}}_{idx}$ y-axis slice" + zone],
                           style=["bar"]*2,
                           simulated_sources=[(pos[1], *pos, -S_hat_slicex[pos[1]]//7),
                                              (pos[0], *pos, -S_hat_slicey[pos[0]]//7)])


# end