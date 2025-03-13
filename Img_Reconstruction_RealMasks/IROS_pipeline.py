"""
Pipeline for running IROS
"""

import numpy as np

import mbloodmoon.iros_management as iros

from mbloodmoon.io import simulation_files, simulation
from mbloodmoon.mask import codedmask, decode, count, variance, snratio
from mbloodmoon.images import upscale, compose


# TODO [1:-1, :] because of problems with `upscale()` and wfm upscaling
def _upscale(arr, upsy):
    ups = upscale(arr, upscale_y=upsy)[1:-1, :]
    return ups


N_TEST = 2


"""
#### IROS SETUP.
"""
print("#### IROS Setup...\n")
root_path = "/mnt/d/PhD_AASS/Coding/Images_fits/"                                                           # directory with files
mask_file = root_path + "wfm_mask.fits"                                                                     # WFM mask
simul_data = root_path + "iros_simulation_GC_LMC/20241011_galctr_rxte_sax_2-30keV_1ks_2cams_sources_cxb/"   # Simulated photons

cam_a = "cam1a"
cam_b = "cam1b"
dataset = "reconstructed"

upsx_0, upsy_0 = 3, 1
wfm = codedmask(mask_file, upscale_x=upsx_0, upscale_y=upsy_0)     # for IROS the skies are upscaled only along the x-dim

filepaths = simulation_files(simul_data)
sdlA = simulation(filepaths[cam_a][dataset])
sdlB = simulation(filepaths[cam_b][dataset])

max_iterations = 15
snr_threshold = 5

sdls = (sdlA, sdlB)
detectors = tuple(count(wfm, sdl.data)[0] for sdl in sdls)
variances = tuple(variance(wfm, d) for d in detectors)

wfm_WCS = codedmask(mask_file, upscale_x=3, upscale_y=5)           # WCS fit (here the camera is upscaled with the final upscaling)
wcs_fit = tuple(iros.handle.fit_WCS(wfm_WCS, sdl) for sdl in sdls)


"""
#### SAVING SIMULATED SKIES.
"""
print("#### Saving simulated skies...\n")
names = tuple(root_path + f"sky_SIMUL_{cam.upper()}_TEST{N_TEST}.fits" for cam in (cam_a, cam_b))
comp_name = root_path + f"composed_sky_SIMUL_{cam_a.upper()}_{cam_b.upper()}_TEST{N_TEST}.fits"

skies = tuple(decode(wfm, d) for d in detectors)
snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skies, variances))

ups_skies = tuple(_upscale(sky, upsy=5) for sky in skies)
ups_snrs = tuple(_upscale(snr, upsy=5) for snr in snrs)

for res, snr, sdl, name, wcs in zip(ups_skies, ups_snrs, sdls, names, wcs_fit):
    iros.save_sky(res, snr, sdl, name, wcs)

iros.save_sky(
    sky=compose(*ups_skies, strict=False)[0],
    snr=compose(*ups_snrs, strict=False)[0],
    sdl=sdlA,
    save_to=comp_name,
    wcs=wcs_fit[0],
)


"""
#### RUN IROS AND SAVE OUTPUT + RESIDUES.
"""
print("#### Running IROS...\n")
iros_output, skies = iros.perform_iros(
    camerasID=(cam_a, cam_b),
    camera=wfm,
    sdl_camA=sdlA,
    sdl_camB=sdlB,
    max_iterations=15,
    snr_threshold=snr_threshold,
    dataset=dataset,
)

iros.save_iros_output(iros_output, mask_file, root_path + f"IROS_output_TEST{N_TEST}.fits")

names = tuple(root_path + f"skyRES_IROS_{cam.upper()}_TEST{N_TEST}.fits" for cam in (cam_a, cam_b))
comp_name = root_path + f"composed_skyRES_IROS_{cam_a.upper()}_{cam_b.upper()}_TEST{N_TEST}.fits"

snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skies, variances))

ups_skies = tuple(_upscale(sky, upsy=5) for sky in skies)
ups_snrs = tuple(_upscale(snr, upsy=5) for snr in snrs)

for res, snr, sdl, name, wcs in zip(ups_skies, ups_snrs, sdls, names, wcs_fit):
    iros.save_sky(res, snr, sdl, name)

iros.save_sky(
    sky=compose(*ups_skies, strict=False)[0],
    snr=compose(*ups_snrs, strict=False)[0],
    sdl=sdlA,
    save_to=comp_name,
    wcs=wcs_fit[0],
)


"""
#### COMPUTE SOURCES PARAMS WITH IROS OUTPUT.
"""
print("#### Computing sources parameters...\n")
log = iros.gen_params_log((cam_a, cam_b))

iros_data = iros.compute_params(
    iros_output=iros_output,
    camera=wfm,
    sdl_camA=sdlA,
    sdl_camB=sdlB,
    log=log,
)

# WARNING: the px position in this DB might not match with upscaled skies
iros.save_iros_data(
    data=iros_data,
    mask_file=mask_file,
    sdls=(sdlA, sdlB),
    save_to=root_path + f"iros_data_TEST{N_TEST}.fits",
)


"""
#### CATALOG COMPARISON AND DATABASE UPDATE.
"""
print("#### Performing catalog comparison...\n")
# WARNING: source assignment relies only on catalog sources
dataset = iros.compare_w_catalog(
    data=iros_data,
    catalogA=filepaths[cam_a]["sources"],
    catalogB=filepaths[cam_b]["sources"],
    camerasID=(cam_a, cam_b),
    min_flux=0.1,
)

iros.save_iros_data(
    data=dataset,
    mask_file=mask_file,
    sdls=(sdlA, sdlB),
    save_to=root_path + f"IROS_sources_database_TEST{N_TEST}.fits",
)


"""
#### GENERATING SKIES FROM IROS OUTPUT + RESIDUES.
"""
print("#### Generating and saving IROS output skies...\n")
names = tuple(root_path + f"OUTsky_IROS_{cam.upper()}_TEST{N_TEST}.fits" for cam in (cam_a, cam_b))
comp_name = root_path + f"composed_OUTsky_IROS_{cam_a.upper()}_{cam_b.upper()}_TEST{N_TEST}.fits"

skies = tuple(iros.make_sky(dataset, camID, wfm) for camID in (cam_a, cam_b))
snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skies, variances))

ups_skies = tuple(_upscale(sky, upsy=5) + res for sky, res in zip(skies, ups_skies))
ups_snrs = tuple(_upscale(snr, upsy=5) + res for snr, res in zip(snrs, ups_snrs))

for res, snr, sdl, name, wcs in zip(ups_skies, ups_snrs, sdls, names, wcs_fit):
    iros.save_sky(res, snr, sdl, name)

iros.save_sky(
    sky=compose(*ups_skies, strict=False)[0],
    snr=compose(*ups_snrs, strict=False)[0],
    sdl=sdlA,
    save_to=comp_name,
    wcs=wcs_fit[0],
)


# end