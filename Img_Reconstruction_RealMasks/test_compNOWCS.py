import numpy as np
from astropy.io import fits
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd

import mbloodmoon.iros_management as iros
from mbloodmoon.io import simulation_files, simulation
from mbloodmoon.mask import codedmask, decode, count, variance, snratio
from mbloodmoon.images import upscale


# TODO [1:-1, :] because of problems with `upscale()` and wfm upscaling
def _upscale(arr, upsy):
    ups = upscale(arr, upscale_y=upsy)[1:-1, :]
    return ups


N_TEST = 3333
UPSX_0, UPSY_FINAL = 3, 5


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

wfm = codedmask(mask_file, upscale_x=UPSX_0, upscale_y=1)     # for IROS the skies are upscaled only along the x-dim

filepaths = simulation_files(simul_data)
sdlA = simulation(filepaths[cam_a][dataset])
sdlB = simulation(filepaths[cam_b][dataset])

sdls = (sdlA, sdlB)
detectors = tuple(count(wfm, sdl.data)[0] for sdl in sdls)
skies = tuple(_upscale(decode(wfm, d), upsy=UPSY_FINAL) for d in detectors)


names = tuple(root_path + f"sky_SIMUL_{cam.upper()}_TEST{N_TEST}.fits" for cam in (cam_a, cam_b))
comp_name = root_path + f"COMPOSED_sky_SIMUL_{cam_a.upper()}_{cam_b.upper()}_TEST{N_TEST}.fits"

for res, sdl, name in zip(skies, sdls, names):
    iros.save_sky(res, np.zeros((10, 10)), sdl, name)




with fits.open(names[0]) as hduA:
    sky_hduA = hduA[1]
with fits.open(names[1]) as hduB:
    sky_hduB = hduB[1]

hdus = (sky_hduA, sky_hduB)
wcs_out, shape_out = find_optimal_celestial_wcs(input_data=hdus)
array, _ = reproject_and_coadd(
    input_data=hdus,
    output_projection=wcs_out,
    shape_out=shape_out,
    reproject_function=reproject_interp,
    combine_function="sum",
)

#Updating the header
#hduA[1].header.update(wcs_out.to_header()) 
fits.writeto(
    filename=root_path + "sky_SIMUL_COMPOSED.fits",
    data=array,
    header=wcs_out.to_header()
)


wcs_out, shape_out = find_optimal_celestial_wcs(input_data=hdus)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "...", line 141, in find_optimal_celestial_wcs
raise TypeError("WCS does not have celestial components")
