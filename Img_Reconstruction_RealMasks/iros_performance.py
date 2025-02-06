"""
Test for IROS performance.
"""

import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm

from mbloodmoon import simulation_files, codedmask, simulation, iros
from mbloodmoon.images import compose, upscale

from dummymoon import image_plot

plt.ion()


def perform_IROS(simul_data: str,
                 mask_file: str,
                 camera_a: str,
                 camera_b: str,
                 dataset: str,
                 compare_w_catalog: bool = True,
                 save_to: str = None,
                 max_iterations: int = 10,
                 snr_threshold: int | float = 5,
                 ) -> dict:
    
    root_path = "/mnt/d/PhD_AASS/Coding/Images_fits/"

    def update_reconstruction_catalog(sources: tuple,
                                      residuals: tuple,
                                      ) -> None:
        for idx, key in enumerate(iros_reconstruction_log.keys()):
            iros_reconstruction_log[key]["sources_log"].append(sources[idx])
            iros_reconstruction_log[key]["residuals_log"].append(residuals[idx])
    
    def compare_catalog():
        # TODO:
        #   - create list of tuple with (source_name [str], match [bool])
        #     with comparison between IROS and catalog sources at that position
        #   - maybe it's useful to insert the sources not detected by IROS but
        #     still present in the catalog with something like (source_name [str], None | "NOT DETECTED")
        pass
    
    def save_output() -> None:
        # TODO:
        #   - save coord in RA, DEC (from output sky shifts)
        #   - convert fluence in flux [ph/cm^2/s] and rate [ph/s]
        #   - save BINTABLE with comparison between IROS and catalog
        #   - save recovered source SNR? 

        hdu_list = fits.HDUList([])

        primary_header = fits.getheader(root_path + mask_file, ext=2)
        primary_header['EXTNAME'] = 'PRIMARY'
        primary_hdu = fits.PrimaryHDU(header=primary_header)
        hdu_list.append(primary_hdu)

        for key in iros_reconstruction_log.keys():
            table_hdu = fits.BinTableHDU.from_columns([
                fits.Column(name='SKYSHIFT_X', array=[s[0] for s in iros_reconstruction_log[key]["sources_log"]], format='F', unit="mm"),
                fits.Column(name='SKYSHIFT_Y', array=[s[1] for s in iros_reconstruction_log[key]["sources_log"]], format='F', unit="mm"),
                fits.Column(name='FLUENCE', array=[s[2] for s in iros_reconstruction_log[key]["sources_log"]], format='F', unit=""),
            ])
            table_hdu.header['EXTNAME'] = f"{key.upper()}"
            hdu_list.append(table_hdu)
        
        if compare_w_catalog:
            pass
        
        hdu_list.writeto(root_path + save_to)

    filepaths = simulation_files(root_path + simul_data)

    # load mask and data
    wfm = codedmask(root_path + mask_file, upscale_x=5)
    sdl_a = simulation(filepaths[camera_a][dataset])
    sdl_b = simulation(filepaths[camera_b][dataset])

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
        sdl_cam1a=sdl_a,
        sdl_cam1b=sdl_b,
        max_iterations=max_iterations,
        snr_threshold=snr_threshold,
        dataset=dataset,
        )

    for idx, (sources, residuals) in enumerate(tqdm(loop)):
        print(f"Iteration {idx}: " + "ok\n" if sources else "no sources detected\n")
        update_reconstruction_catalog(sources, residuals)
    
    # compare with sources catalog
    if compare_w_catalog:
        matches = compare_catalog()
    
    # save data as .fits file
    if save_to is not None:
        save_output()
    
    return iros_reconstruction_log


def post_process(a: np.array,
                 max_value: float,
                 ) -> np.array:
    post_a = a.copy()
    post_a[a < 0] = 0
    post_a[a > max_value] = 0
    return upscale(post_a, upscale_y=8)



if __name__ == '__main__':

    # paths for mask and data .fits
    mask_file = "wfm_mask.fits"
    simul_data = "iros_simulation_GC_LMC/galctr_rxte_sax_2-30keV_1ks_sources_cxb/"  #lmc_rxte_sax_2-30keV_10ks_sources_cxb/

    # select data
    cam_a = "cam1a"
    cam_b = "cam1b"
    dataset = "reconstructed"

    save_file = True
    output_name = "iros_performance_test.fits" if save_file else None

    max_iterations = 5
    snr_threshold = 5

    iros_log = perform_IROS(
        simul_data=simul_data,
        mask_file=mask_file,
        camera_a=cam_a,
        camera_b=cam_b,
        dataset=dataset,
        save_to=output_name,
        max_iterations=max_iterations,
        snr_threshold=snr_threshold,
        )
    
    s, r = "sources_log", "residuals_log"
    last_res_a = iros_log[cam_a][r][-1]
    last_res_b = iros_log[cam_b][r][-1]

    q = 1 - 2e-4
    post_res_a = post_process(last_res_a, np.quantile(last_res_a, q))
    post_res_b = post_process(last_res_b, np.quantile(last_res_b, q))
    composed, _ = compose(post_res_a, post_res_b, strict=False)
    
    # image_plot([post_res_a, post_res_b], [cam_a.upper(), cam_b.upper()])
    # image_plot([composed], [f"Composed {cam_a.upper()}-{cam_b.upper()}"])


# end