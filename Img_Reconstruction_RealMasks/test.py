import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import dummymoon as dm

#plt.ion()
matplotlib.use('agg')

fits_path = "../Images_fits/wfm_mask.fits"
wfm = dm.import_mask(fits_path, True, True)

n_sources = 5
sources_flux = np.random.randint(int(5e2), int(2e3), size=n_sources)
sources_pos = None
sky_background_rate = int(5e1)

sky_image, sky_background, sources_pos = dm.sky_image_simulation(wfm.sky_shape, sources_flux, sources_pos, sky_background_rate)

dm.image_plot([sky_image, sky_background],
                ["Simulated Sky Image", "Simulated Sky Background"],
                cbarlabel=["counts", "counts"],
                cbarcmap=["viridis"]*2,
                simulated_sources=[sources_pos, None])

