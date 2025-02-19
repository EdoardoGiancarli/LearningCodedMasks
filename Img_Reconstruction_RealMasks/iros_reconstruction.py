"""
Module for testing IROS reconstruction.
"""

from copy import deepcopy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
import pickle

from mbloodmoon.images import argmax, compose, upscale
from mbloodmoon.coords import shift2equatorial
import mbloodmoon as bm

matplotlib.use('agg')
root_path = "/mnt/d/PhD_AASS/Coding/Images_fits/"

def plot_cameras(skyrecs, name) -> None:
    sky_a, sky_b = skyrecs
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    plt.tight_layout()
    for ax, b, bmax, title in zip(
            axs,
            [sky_a, sky_b],
            [argmax(sky_a), argmax(sky_b)],
            ["SkyRec CamA", "SkyRec CamB"],
    ):
        ax.imshow(b, vmin=0, vmax=-b.min())
        ax.scatter(bmax[1], bmax[0], facecolors='none', edgecolors='white', alpha=0.5)
        ax.set_title(title, fontsize=14, pad=8, fontweight='bold')
    plt.savefig(root_path + name + '.png')
    plt.close()

def plot_skyrec(skyrecs, title, source_indices=None, source_names=None, dpi=200, upsc_y=8):
    composed, _ = compose(*skyrecs, strict=False)
    fig, ax = plt.subplots(1, 1, figsize=(8, 10), dpi=dpi)
    if source_indices is not None and source_names is not None:
        for ((i, j), name) in zip(source_indices, source_names):
            ax.scatter(j, i * upsc_y + 53, s=30, facecolors="none", edgecolors="white", alpha=1., linewidth=.5)
            ax.text(j + 50 , i * upsc_y + 100, name, color="white", fontsize=4)
    im = ax.imshow(composed, vmax=np.quantile(composed, 0.9995), vmin=0., cmap="viridis")
    plt.colorbar(im, ax=ax, label='SNR', fraction=0.025, aspect=35, pad=0.02, shrink=0.33, location="bottom")
    ax.set_title(title, fontsize=12, pad=8, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(root_path + title.replace(' ', '_').lower() + ".png")
    plt.close()

def perform_IROS():
    pass





if __name__ == '__main__':

    ## paths for mask and data .fits
    mask_file = root_path + "wfm_mask.fits"
    simul_data = root_path + "iros_simulation_GC_LMC/20241011_galctr_rxte_sax_2-30keV_1ks_2cams_sources_cxb/"  #lmc_rxte_sax_2-30keV_10ks_sources_cxb/
    # galctr_rxte_sax_2-30keV_1ks_sources_cxb/

    ## select data
    cam_a = "cam1a"
    cam_b = "cam1b"
    dataset = "reconstructed"
    # filepaths = bm.simulation_files(simul_data)
    # wfm = bm.codedmask(mask_file, upscale_x=5)
    # sdl_1a = bm.simulation(filepaths["cam1a"][dataset])
    # sdl_1b = bm.simulation(filepaths["cam1b"][dataset])

    ## save data and catalog comparison
    save_file = False
    compare_w_catalog = False

    save_to = root_path + "iros_performance_test.fits" if save_file else None
    catalog = bm.simulation_files(simul_data)[cam_a]['sources'] if compare_w_catalog else None    # TODO

    ## perform IROS
    max_iterations = 5
    snr_threshold = 5

    iros_log = perform_IROS(
        simul_data=simul_data,
        mask_file=mask_file,
        cam_a=cam_a,
        cam_b=cam_b,
        dataset=dataset,
        max_iterations=max_iterations,
        snr_threshold=snr_threshold,
    )

    ## comparing IROS reconstruction with catalog
    #source_indices, source_names = compare_w_catalog(iros_log, catalog)

    ## saving IROS reconstruction
    ...

    ## plot
    # upscaled_cams = [upscale(cam, upscale_y=8) for cam in ...]
    # upscaled_snrs = [upscale(snr, upscale_y=8) for snr in ...]
 
    # print("Trying to plot damn cameras big images...")
    # plot_cameras(upscaled_cams, "cams")
    # print("Cameras done, now the SNRs...")
    # plot_cameras(upscaled_snrs, "snrs_cams")
    # print("Trying to plot damn composed big images...")
    # plot_skyrec(upscaled_cams, "Galactic Center IROS rec")
 
    # print("Sky done, now the SNR...")
    # title = "SNR CAM composition"
    # plot_skyrec(upscaled_snrs, source_indices, source_names, title)


# end