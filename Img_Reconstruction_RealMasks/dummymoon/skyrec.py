"""
Sky Reconstruction through cross-correlation...
"""

import warnings
import numpy as np
from scipy.signal import correlate
from scipy.stats import norm
from skymap import sky_image_simulation
from display import crop, image_plot
from tqdm import tqdm
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap as lc


def transmitted_sky_image(sky: np.array,
                          cam: object,
                          ) -> np.array:
    
    return sky*cam.specs['real_open_fraction']


def sky_encoding(sky: np.array,
                 cam: object,
                 ) -> np.array:
    
    return correlate(cam.mask, sky, mode='valid')*cam.bulk


def sky_reconstruction(detector: np.array,
                       cam: object,
                       ) -> tuple[np.array, np.array]:
    
    sum_det, sum_bulk = detector.sum(), cam.bulk.sum()

    sky = correlate(cam.decoder, detector)
    bal_sky = sky - cam.balancing*sum_det/sum_bulk

    var = correlate(np.square(cam.decoder), detector)
    bal_var = var + np.square(cam.balancing)*sum_det/np.square(sum_bulk) - 2*sky*cam.balancing/sum_bulk
    #bal_var = var - correlate(np.square(decoder), bulk)*sum_det/sum_bulk
    bal_var[bal_var <= 0] = np.inf

    return bal_sky, bal_var


def skyrec_norm(skyrec: np.array,
                skyvar: np.array,
                cam: object,
                ) -> np.array:
    
    # TODO: insert counts reconstruction correction for pos in img
    
    aperture = cam.mask.sum()
    det_eff_response = cam.bulk.sum()/cam.mask.size
    norm = 1/(aperture*det_eff_response)

    return norm*skyrec, np.square(norm)*skyvar


def sky_snr(sky: np.array,
            var: np.array,
            ) -> np.array:
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = np.nan_to_num(sky/np.sqrt(var))
    
    return s


def sky_snr_peaks(skysnr:np.array,
                  threshold: int | float,
                  skysignificance:np.array,
                  sources_pos: list[tuple[int, int]],
                  ) -> None:
    
    loc = np.argwhere(skysnr > threshold).T
    snr_pos = np.dstack((loc[0], loc[1]))[0]

    image_plot([skysignificance, skysnr],
               ["Sky Significance", f"SkyRec SNR > {threshold}"],
               cbarlabel=["significance[$\sigma$]", "SNR[$\sigma$]"],
               cbarlimits=[(None, None), (None, None)],
               cbarscinot=[True]*2,
               cbarcmap=["viridis"]*2,
               simulated_sources=[sources_pos, snr_pos])


def show_snr_distr(snr: np.array,
                   _title: str = None) -> None:
    
    title = "SNR Statistics"
    if _title: title + _title
    
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    fig.tight_layout()
    ax.hist(snr.reshape(-1), bins=50, density= True,
            color='SkyBlue', edgecolor='b', alpha=0.7)
    ax.plot(x := np.linspace(-5, 5, 1000), norm.pdf(x),
            color="OrangeRed", label="Normal distr.")
    ax.set_xlabel("SNR", fontsize=12, fontweight='bold')
    ax.set_ylabel("density", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, pad=8, fontweight='bold')
    ax.grid(visible=True, color="lightgray", linestyle="-", linewidth=0.3)
    ax.legend(loc='best')
    ax.tick_params(which='both', direction='in', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.show()


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


def print_skyrec_info(sky_image: np.array,
                      skyrec: np.array,
                      skyvar: np.array,
                      sources_pos: list[tuple[int, int]],
                      show_sources: bool = True,
                      ) -> None:
    
    for idx, pos in enumerate(sources_pos):
        print(
            f"Simulated Source [{idx}] transmitted counts: {sky_image[*pos]:.0f} +/- {np.sqrt(sky_image[*pos]):.0f}\n"
            f"Reconstructed Source [{idx}] counts: {skyrec[*pos]:.0f} +/- {np.sqrt(skyvar[*pos]):.0f}\n"
            f"Source [{idx}] reconstructed counts wrt simulated: {skyrec[*pos]*100/sky_image[*pos]:.2f}%\n"
        )
        
        if show_sources:
            crp = 40
            try:
                c_sky = crop(sky_image, pos, (crp, crp))
                c_skyrec = crop(skyrec, pos, (crp, crp))
                print(f"True pos: {np.argwhere(c_skyrec == c_skyrec.max())[0].T == np.argwhere(c_sky == c_sky.max())[0].T}\n")
            except ValueError:
                print("Source difficult to crop...\n")


def print_snr_info(sky_image: np.array,
                   skysnr: np.array,
                   sources_pos: list[tuple[int, int]],
                   show_sources: bool = True,
                   ) -> None:
    
    for idx, pos in enumerate(sources_pos):
        print(f"SNR Source [{idx}] value: {skysnr[*pos]:.0f}")
        
        if show_sources:
            crp = 40
            try:
                c_sky = crop(sky_image, pos, (crp, crp))
                c_skysnr = crop(skysnr, pos, (crp, crp))
                print(f"True pos: {np.argwhere(c_skysnr == c_skysnr.max())[0].T == np.argwhere(c_sky == c_sky.max())[0].T}\n")
            except ValueError:
                print("Source difficult to crop...\n")


# end