"""
Test for IROS performance.
"""

import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from mbloodmoon import simulation_files, codedmask, simulation, iros
from mbloodmoon.images import compose, upscale

from dummymoon import image_plot

plt.ion()


root_path = "/mnt/d/PhD_AASS/Coding/Images_fits/"

def perform_IROS(simul_data: str,
                 mask_file: str,
                 camera_a: str,
                 camera_b: str,
                 dataset: str,
                 max_iterations: int = 10,
                 snr_threshold: int | float = 5,
                 ) -> dict:
    
    def update_reconstruction_catalog(sources: tuple,
                                      residuals: tuple,
                                      ) -> None:
        for idx, key in enumerate(iros_reconstruction_log.keys()):
            iros_reconstruction_log[key]["sources_log"].append(sources[idx])
            iros_reconstruction_log[key]["residuals_log"].append(residuals[idx])

    filepaths = simulation_files(root_path + simul_data)

    # load mask and data
    wfm = codedmask(root_path + mask_file, upscale_x=5)
    sdl1a = simulation(filepaths[camera_a][dataset])
    sdl1b = simulation(filepaths[camera_b][dataset])

    # init
    iros_reconstruction_log = {
        camera_a: {"sources_log": [],
                   "residuals_log": []},
        camera_b: {"sources_log": [],
                   "residuals_log": []},
    }

    # IROS
    loop = iros(
        camera=wfm,
        sdl_cam1a=sdl1a,
        sdl_cam1b=sdl1b,
        max_iterations=max_iterations,
        snr_threshold=snr_threshold,
        dataset=dataset,
        )

    for idx, (sources, residuals) in enumerate(tqdm(loop)):
        print(f"Iteration {idx}: " + "ok\n" if sources else "no sources detected\n")
        update_reconstruction_catalog(sources, residuals)
    
    return iros_reconstruction_log



if __name__ == '__main__':

    # paths for mask and data .fits
    mask_file = "wfm_mask.fits"
    simul_data = "iros_simulation_GC_LMC/galctr_rxte_sax_2-30keV_1ks_sources_cxb/"  #lmc_rxte_sax_2-30keV_10ks_sources_cxb/

    # select data
    cam_a = "cam1a"
    cam_b = "cam1b"
    dataset = "reconstructed"

    max_iterations = 7
    snr_threshold = 5

    iros_log = perform_IROS(
        simul_data, mask_file,
        cam_a, cam_b, dataset,
        max_iterations, snr_threshold
        )
    
    s, r = "sources_log", "residuals_log"
    last_res_a = iros_log[cam_a][r][-1]
    last_res_b = iros_log[cam_b][r][-1]

    q = 1 - 2e-4
    a = last_res_a.copy(); a[a < 0] = 0; a[a > np.quantile(last_res_a, q)] = 0
    b = last_res_b.copy(); b[b < 0] = 0; b[b > np.quantile(last_res_b, q)] = 0
    image_plot([a, b], [cam_a.upper(), cam_b.upper()])

    composed, _ = compose(
        upscale(a, upscale_y=8),
        upscale(b, upscale_y=8),
        strict=False,
        )
    
    image_plot([composed], [f"Composed {cam_a.upper()}-{cam_b.upper()}"])


# end