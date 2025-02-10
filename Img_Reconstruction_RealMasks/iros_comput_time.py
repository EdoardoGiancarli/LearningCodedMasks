"""
Module for testing IROS computational time performance.
"""

from tqdm import tqdm
from mbloodmoon import simulation_files, codedmask, simulation, iros

def run_IROS(simul_data: str,
             mask_file: str,
             camera_a: str,
             camera_b: str,
             dataset: str,
             max_iterations: int = 30,
             snr_threshold: int | float = 5,
             ) -> None:
    
    # load mask and data
    filepaths = simulation_files(root_path + simul_data)
    wfm = codedmask(root_path + mask_file, upscale_x=5)
    sdl_a = simulation(filepaths[camera_a][dataset])
    sdl_b = simulation(filepaths[camera_b][dataset])

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
    


if __name__ == '__main__':

    # paths for mask and data .fits
    root_path = "/mnt/d/PhD_AASS/Coding/Images_fits/"
    mask_file = "wfm_mask.fits"
    simul_data = "iros_simulation_GC_LMC/lmc_rxte_sax_2-30keV_10ks_sources_cxb/"  # galctr_rxte_sax_2-30keV_1ks_sources_cxb/

    # select data
    cam_a = "cam1a"
    cam_b = "cam1b"
    dataset = "reconstructed"

    print("### Beginning IROS...")
    run_IROS(
        simul_data=simul_data,
        mask_file=mask_file,
        camera_a=cam_a,
        camera_b=cam_b,
        dataset=dataset,
    )
    print("### End IROS...")


# end