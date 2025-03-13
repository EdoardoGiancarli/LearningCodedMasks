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
    ups = upscale(arr, upscale_y=upsy)[2:-1, :]
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

upsx_0, upsy_0 = 5, 1
wfm = codedmask(mask_file, upscale_x=upsx_0, upscale_y=upsy_0)     # for IROS the skies are upscaled only along the x-dim

filepaths = simulation_files(simul_data)
sdlA = simulation(filepaths[cam_a][dataset])
sdlB = simulation(filepaths[cam_b][dataset])

max_iterations = 15
snr_threshold = 5

sdls = (sdlA, sdlB)
detectors = tuple(count(wfm, sdl.data)[0] for sdl in sdls)
variances = tuple(variance(wfm, d) for d in detectors)

wfm_WCS = codedmask(mask_file, upscale_x=5, upscale_y=8)           # WCS fit (here the camera is upscaled with the final upscaling)
wcs_fit = tuple(iros.handle.fit_WCS(wfm_WCS, sdl) for sdl in sdls)


"""
#### SAVING SIMULATED SKIES.
"""
print("#### Saving simulated skies...\n")
names = tuple(root_path + f"sky_SIMUL_{cam.upper()}_TEST{N_TEST}.fits" for cam in (cam_a, cam_b))
comp_name = root_path + f"composed_sky_SIMUL_{cam_a.upper()}_{cam_b.upper()}_TEST{N_TEST}.fits"

skies = tuple(decode(wfm, d) for d in detectors)
snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skies, variances))

ups_skies = tuple(_upscale(sky, upsy=8) for sky in skies)
ups_snrs = tuple(_upscale(snr, upsy=8) for snr in snrs)

for res, snr, sdl, name, wcs in zip(ups_skies, ups_snrs, sdls, names, wcs_fit):
    iros.save_sky(res, snr, sdl, name, wcs)

iros.save_sky(
    sky=compose(*ups_skies, strict=False)[0],
    snr=compose(*ups_snrs, strict=False)[0],
    sdl=sdlA,
    save_to=comp_name,
    wcs=wcs_fit[0],
)


from mbloodmoon.coords import pos2equatorial

def test_WCSfit(wcs, sdl, camera, tol):
    n, m = camera.sky_shape
    l = 50
    res = [
        int(np.all(np.abs(wcs.all_pix2world(np.array([(l*x, l*y)]), 0, ra_dec_order=True)[0] - pos2equatorial(sdl, camera, l*y, l*x)) < tol))
        for y in range(0, n//l) for x in range(0, m//l)
    ]
    print(f"Fit accuracy at tolerance {tol}: {sum(res)*100/len(res)}%")

    res2 = [
        np.square((wcs.all_pix2world(np.array([(l*x, l*y)]), 0, ra_dec_order=True)[0] - pos2equatorial(sdl, camera, l*y, l*x))/tol)
        for y in range(0, n//l) for x in range(0, m//l)
    ]
    dof = len(res2) - 2
    print(f"{sum(res2) / dof}")

tol = 1e-7
test_WCSfit(wcs_fit[0], sdlA, wfm_WCS, tol)
test_WCSfit(wcs_fit[1], sdlB, wfm_WCS, tol)


# end