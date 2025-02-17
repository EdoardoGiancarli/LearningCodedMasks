"""
Module for testing IROS reconstruction.
"""

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

def plot_composed_cam(skyrecs, title) -> None:
    composed, _ = compose(*skyrecs, strict=False)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)
    ax.imshow(composed, vmax=np.quantile(composed, 0.9995), vmin=0.)
    plt.tight_layout()
    ax.set_title(title, fontsize=14, pad=8, fontweight='bold')
    plt.savefig(root_path + title.replace(' ', '_').lower() + ".png")
    plt.close()

def plot_skyrec(skyrecs, source_indices, source_names, title, upsc_y=8):
    composed, _ = compose(*skyrecs, strict=False)
    fig, ax = plt.subplots(1, 1, figsize=(8, 10), dpi=250)
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



def perform_IROS(sdlA: object,
                 sdlB: object,
                 cam: object,
                 dataset: str = "reconstructed",
                 save_to: str = None,
                 max_iterations: int = 10,
                 snr_threshold: int | float = 5,
                 catalog_name: str = None,
                 ):

    def _init_log():
        print("## Initializing...")
        canvas = [np.zeros(cam.sky_shape), np.zeros(cam.sky_shape)]
        canvas_snr = [np.zeros(cam.sky_shape), np.zeros(cam.sky_shape)]
        detectors = [bm.count(cam, sdlA.data)[0], bm.count(cam, sdlB.data)[0]]
        skys = [bm.decode(cam, d) for d in detectors]
        variances = [bm.variance(cam, d) for d in detectors]
        snrs = [bm.snratio(s, v) for s, v in zip(skys, variances)]
        rec_sources = [[], []]
        return canvas, canvas_snr, skys, variances, snrs, rec_sources
    
    def save_output(iros_sources) -> None:
        # TODO:
        #   - to save:
        #   - coords in [px, px], [theta_x, theta_y], [RA, DEC] (from output sky shifts, see shift2pos in mask.py)
        #   - fluence at the peak for source choice
        #   - effective counts subtracted from the detector
        #   - estimated counts from the source (before the mask)
        #   - fluence in flux [ph/cm^2/s] and rate [ph/s]
        #   - SNR, value for the goodness of the fit for pos/fluence
        #
        #   - save BINTABLE with comparison between IROS and catalog
        print("## Saving...")

        def gen_bintab_column():
            pass

        def save_to_pickle(var) -> None:
            with open(root_path + save_to + str(var) + ".pickle", "wb") as handle:
                pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)

        def compare_w_catalog():
        # TODO:
        #   - create list of tuple with (source_name [str], match [bool])
        #     with comparison between IROS and catalog sources at that position
        #   - maybe it's useful to insert the sources not detected by IROS but
        #     still present in the catalog with something like (source_name [str], None | "NOT DETECTED")
            print("## Comparing with Catalog...")
            catalog = fits.getdata(catalog_name)
            catalog = catalog[catalog["AVG_FLUX"] > 0.1]
            source_names = []
            sources_radec_A = [
                bm.shift2equatorial(sdlA, cam, shift_x, shift_y)
                for shift_x, shift_y, _ in iros_sources[0]
                ]
            source_indices = [
                bm.shift2pos(wfm, iros_sources[0][i][0], iros_sources[1][i][0])
                for i in range(len(iros_sources[0]))
                ]
            for source in sources_radec_A:
                argsource = np.argmin(
                    np.square(catalog["RA"] - source.ra) + np.square(catalog["DEC"] - source.dec)
                )
                name = catalog["NAME"][argsource]
                flux = catalog["AVG_FLUX"][argsource]
                source_names.append(name)
            return source_indices, source_names
        
        # creating HDU list and Primary Header
        hdu_list = fits.HDUList([])
        primary_header = fits.getheader(root_path + mask_file, ext=2)     ###TODO!!!
        primary_header['EXTNAME'] = 'PRIMARY'
        primary_hdu = fits.PrimaryHDU(header=primary_header)
        hdu_list.append(primary_hdu)

        # creating binary table                                           ###TODO!!!
        for key in []:
            cols = [gen_bintab_column()]
            table_hdu = fits.BinTableHDU.from_columns(*cols)
            table_hdu.header['EXTNAME'] = f"{key.upper()}"
            hdu_list.append(table_hdu)
        
        hdu_list.writeto(root_path + save_to)

        # saving canvas (just in case, for now)
        for var in [canvas, canvas_snr]:
            save_to_pickle(var)

        # catalog comparison
        source_indices, source_names = compare_w_catalog() if catalog_name else (None, None)

        return source_indices, source_names



    # init log
    canvas, canvas_snr, skys, variances, snrs, rec_sources = _init_log()

    # IROS
    print("## Looping around the FOV...")
    loop = bm.iros(
        camera=cam,
        sdl_cam1a=sdlA,
        sdl_cam1b=sdlB,
        max_iterations=max_iterations,
        snr_threshold=snr_threshold,
        dataset=dataset,
        )

    for idx, (sources, residuals) in enumerate(tqdm(loop)):
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

    # save data as .fits/.pickle files
    if save_to is not None:
        source_indices, source_names = save_output(rec_sources)
    else:
        source_indices, source_names = None, None
    
    iros_log = {
        "rec_sources": rec_sources,
        "sky_canvas": canvas,
        "snr_canvas": canvas_snr,
        "source_indices": source_indices,
        "source_names": source_names,
    }
    
    return iros_log





if __name__ == '__main__':

    # paths for mask and data .fits
    mask_file = root_path + "wfm_mask.fits"
    simul_data = root_path + "iros_simulation_GC_LMC/20241011_galctr_rxte_sax_2-30keV_1ks_2cams_sources_cxb/"  #lmc_rxte_sax_2-30keV_10ks_sources_cxb/
    # galctr_rxte_sax_2-30keV_1ks_sources_cxb/

    # select data
    filepaths = bm.simulation_files(simul_data)   # fixed bm.io: >>> search for *detected*.fits
    wfm = bm.codedmask(mask_file, upscale_x=5)
    dataset = "reconstructed"
    sdl_1a = bm.simulation(filepaths["cam1a"][dataset])
    sdl_1b = bm.simulation(filepaths["cam1b"][dataset])

    # save data and catalog comparison
    save_file = False
    compare_w_catalog = False

    save_to = root_path + "iros_performance_test.fits" if save_file else None
    catalog = root_path + "catalog.fits" if compare_w_catalog else None                 ### TODO!!!

    # perform IROS
    max_iterations = 5
    snr_threshold = 5

    iros_log = perform_IROS(
        sdlA=sdl_1a,
        sdlB=sdl_1b,
        cam=wfm,
        dataset=dataset,
        save_to=save_to,
        max_iterations=max_iterations,
        snr_threshold=snr_threshold,
        catalog_name=catalog,
    )

    # plot
    upscaled_cams = [upscale(cam, upscale_y=8) for cam in iros_log["sky_canvas"]]
    upscaled_snrs = [upscale(snr, upscale_y=8) for snr in iros_log["snr_canvas"]]

    print("Trying to plot damn cameras big images...")
    plot_cameras(upscaled_cams, "cams")
    print("Cameras done, now the SNRs...")
    plot_cameras(upscaled_snrs, "snrs_cams")
    print("Trying to plot damn composed big images...")
    plot_composed_cam(upscaled_cams, "Galactic Center IROS rec")

    print("Sky done, now the SNR...")
    source_indices, source_names = iros_log["source_indices"], iros_log["source_names"]
    title = "SNR CAM composition"
    if source_indices and source_names:
        plot_skyrec(upscaled_snrs, source_indices, source_names, title)
    else:
        plot_composed_cam(upscaled_snrs, title)


# end