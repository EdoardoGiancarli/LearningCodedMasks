"""
Module for testing IROS output.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
import pickle

from mbloodmoon.images import compose, upscale, argmax
import mbloodmoon as bm

matplotlib.use('agg')
root_path = "/mnt/d/PhD_AASS/Coding/Images_fits/"

def plot_skyrec(skyrecs, source_indices, source_names, title, upsc_y=8, dpi=300):
    composed, _ = compose(*skyrecs, strict=False)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12.8), dpi=dpi)
    for ((i, j), name) in zip(source_indices, source_names):
        ax.scatter(j, i * upsc_y + 53, s=30, facecolors="none", edgecolors="white", alpha=1., linewidth=.5)
        ax.text(j + 50 , i * upsc_y + 100, name, color="white", fontsize=4)
    im = ax.imshow(composed, vmax=np.quantile(composed, 0.9995), vmin=0., cmap="viridis")
    plt.colorbar(im, ax=ax, label='SNR', fraction=0.025, aspect=35, pad=0.02, shrink=0.33, location="bottom")
    plt.axis("off")
    plt.tight_layout()
    ax.set_title(title, fontsize=14, pad=8, fontweight='bold')
    plt.savefig(root_path + title.replace(' ', '_').lower() + ".png")
    plt.close()

def plot_cameras(skyrecs,
                 name) -> None:
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

def plot_composed_cam(skyrecs,
                      title,
                      dpi=150,
                      ) -> None:
    composed, _ = compose(*skyrecs, strict=False)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=dpi)
    ax.imshow(composed, vmax=np.quantile(composed, 0.9995), vmin=0.)
    plt.tight_layout()
    ax.set_title(title, fontsize=14, pad=8, fontweight='bold')
    plt.savefig(root_path + title.replace(' ', '_').lower() + ".png")
    plt.close()




if __name__ == '__main__':

    only_plot = True

    if not only_plot:
        # paths for mask and data .fits
        mask_file = root_path + "wfm_mask.fits"
        simul_data = root_path + "iros_simulation_GC_LMC/20241011_galctr_rxte_sax_2-30keV_1ks_2cams_sources_cxb/"  #lmc_rxte_sax_2-30keV_10ks_sources_cxb/
        # galctr_rxte_sax_2-30keV_1ks_sources_cxb/

        # select data
        filepaths = bm.simulation_files(simul_data)   # fixed bm.io: >>> search for *detected*.fits
        wfm = bm.codedmask(mask_file, upscale_x=5)
        sdl_1a = bm.simulation(filepaths["cam1a"]["reconstructed"])
        sdl_1b = bm.simulation(filepaths["cam1b"]["reconstructed"])

        # make imgs
        canvas = [np.zeros(wfm.sky_shape), np.zeros(wfm.sky_shape)]
        canvas_snr = [np.zeros(wfm.sky_shape), np.zeros(wfm.sky_shape)]
        detectors = [bm.count(wfm, sdl_1a.data)[0], bm.count(wfm, sdl_1b.data)[0]]
        skys = [bm.decode(wfm, d) for d in detectors]
        variances = [bm.variance(wfm, d) for d in detectors]
        snrs = [bm.snratio(s, v) for s, v in zip(skys, variances)]

        rec_sources = [[], []]
        loop = bm.iros(wfm, sdl_1a, sdl_1b, max_iterations=20)

        for idx, (sources, residuals) in enumerate(tqdm(loop)):
            print(f"## Iteration {idx}")
            for c, _ in enumerate(sources):
                *shift, flux = sources[c]
                print(f"# Source {idx}: pos {shift}, counts {flux}")
                (i_min, i_max, j_min, j_max), _ = bm.strip(wfm, bm.shift2pos(wfm, *shift))
                (i_min, i_max, j_min, j_max) = (i_min - 18, i_max + 18, j_min - 3, j_max + 3)
                canvas[c][i_min:i_max, j_min:j_max] += skys[c][i_min:i_max, j_min:j_max]
                canvas_snr[c][i_min:i_max, j_min:j_max] += snrs[c][i_min:i_max, j_min:j_max]
    
                skys[c] = residuals[c]
                snrs[c] = bm.snratio(skys[c], variances[c])
                rec_sources[c].append(sources[c])
        
        upscaled_cams = [upscale(cam, upscale_y=8) for cam in canvas]
        upscaled_snrs = [upscale(snr, upscale_y=8) for snr in canvas_snr]

        for var, name in zip([skys, snrs, rec_sources, upscaled_cams, upscaled_snrs],
                             ["sky_residues", "snr_residues", "rec_sources", "upscaled_cams", "upscaled_snrs"]):
            with open(root_path + name + ".pickle", "wb") as f:
                pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Trying plotting damn big images...")
        #plot_cameras(upscaled_cams, "cams")
        #plot_cameras(upscaled_snrs, "snrs_cams")
        #plot_composed_cam(upscaled_cams, "Galactic Center IROS rec")
        #plot_composed_cam(upscaled_snrs, "SNR CAM composition")
    
    else:
        with open(root_path + 'upscaled_cams.pickle', 'rb') as handle:
            upscaled_cams = pickle.load(handle)
        with open(root_path + 'upscaled_snrs.pickle', 'rb') as handle:
            upscaled_snrs = pickle.load(handle)
        with open(root_path + 'snr_residues.pickle', 'rb') as handle:
            snr_residues = pickle.load(handle)
        
        upscaled_res = [upscale(r, upscale_y=8) for r in snr_residues]
        total_snr = [upscaled_res[idx] + upscaled_snrs[idx] for idx in range(2)]
        plot_composed_cam(total_snr, "Galactic Center IROS SNR")
        
        #print("Trying plotting cams SNR...")
        #plot_cameras(upscaled_snrs, "snrs_cams")
        #print("Trying plotting composed cams...")
        #plot_composed_cam(upscaled_cams, "Galactic Center IROS rec")
        #print("Trying plotting composed SNRS...")
        #plot_composed_cam(upscaled_snrs, "SNR CAM composition")


# end