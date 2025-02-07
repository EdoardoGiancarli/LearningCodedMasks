# iros.py
 
import pickle
 
import matplotlib.pyplot as plt
import numpy as np
 
from bloodmoon import simulation_files
import bloodmoon as bm
from bloodmoon.images import compose
from bloodmoon.images import upscale
 
PIC_COUNTER = [0]
 
 
def plot(skys_t, dpi=150):
    composed, _ = compose(
        *[upscale(sky, upscale_x=1, upscale_y=8) for sky in skys_t],
        strict=False,
    )
 
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=dpi)
    ax.imshow(composed, vmax=np.quantile(composed, 0.9995), vmin=0.)
    plt.tight_layout()
    plt.savefig(f"pic{PIC_COUNTER[0]}.png")
    PIC_COUNTER[0] += 1
    return
 
 
if __name__ == '__main__':
    wfm = bm.codedmask("../../../simulations/wfm_mask.fits", upscale_x=5)
    files = simulation_files("../../../simulations/galcenter")
    ds_cam1a = bm.simulation(files["cam1a"]["reconstructed"])
    ds_cam1b = bm.simulation(files["cam1b"]["reconstructed"])
 
    canvas = [np.zeros(wfm.sky_shape), np.zeros(wfm.sky_shape)]
    canvas_snr = [np.zeros(wfm.sky_shape), np.zeros(wfm.sky_shape)]
    detectors = [bm.count(wfm, ds_cam1a.data)[0], bm.count(wfm, ds_cam1b.data)[0]]
    skys = [bm.decode(wfm, d) for d in detectors]
    variances = [bm.variance(wfm, d) for d in detectors]
    snrs = [bm.snratio(s, v) for s, v in zip(skys, variances)]
 
    subtracted = [[], []]
    iros = bm.iros(wfm, ds_cam1a, ds_cam1b, max_iterations=20)
    for i, (sources, residuals) in enumerate(iros):
        print(f"on iteration {i}")
        for c, _ in enumerate(sources):
            *shift, flux = sources[c]
            print(shift, flux)
            (i_min, i_max, j_min, j_max), _ = bm.strip(wfm, bm.shift2pos(wfm, *shift))
            (i_min, i_max, j_min, j_max) = (i_min - 18, i_max + 18, j_min - 3, j_max + 3)
            canvas[c][i_min:i_max, j_min:j_max] += skys[c][i_min:i_max, j_min:j_max]
            canvas_snr[c][i_min:i_max, j_min:j_max] += snrs[c][i_min:i_max, j_min:j_max]
 
            skys[c] = residuals[c]
            snrs[c] = bm.snratio(skys[c], variances[c])
            subtracted[c].append(sources[c])
 
    with open("sources_new.pickle", "wb") as f:
        pickle.dump(subtracted, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open("canvas.pickle", "wb") as f:
    #     pickle.dump(canvas, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open("canvas_snr.pickle", "wb") as f:
    #     pickle.dump(canvas_snr, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open("bkg.pickle", "wb") as f:
    #     pickle.dump(skys, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open("bkg_snr.pickle", "wb") as f:
    #     pickle.dump(snrs, f, protocol=pickle.HIGHEST_PROTOCOL)
 
    plot(canvas)
    plot(canvas_snr)
 
 
# reconstruct.py
 
import pickle
 
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
 
from bloodmoon import shift2pos
import bloodmoon as bm
from bloodmoon.coords import shift2equatorial
from bloodmoon.images import compose
from bloodmoon.images import upscale
 
 
def plot(sky_t, source_indices, source_names, dpi=300):
    composed, _ = compose(
        *[upscale(sky, upscale_x=1, upscale_y=8) for sky in sky_t],
        strict=False,
    )
    fig, ax = plt.subplots(1, 1, figsize=(12, 12.8), dpi=dpi)
    for ((i, j), name) in zip(source_indices, source_names):
        ax.scatter(j, i * 8 + 53, s=30, facecolors="none", edgecolors="white", alpha=1., linewidth=.5)
        ax.text(j + 50 , i * 8 + 100, name, color="white", fontsize=4)
    im = ax.imshow(composed, vmax=np.quantile(composed, 0.9995), vmin=0., cmap="viridis")
    plt.colorbar(im, ax=ax, label='SNR', fraction=0.025, aspect=35, pad=0.02, shrink=0.33, location="bottom")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"annotated_new.png")
    return
 
 
if __name__ == '__main__':
    with open('sources.pickle', 'rb') as handle:
        sources = pickle.load(handle)
    with open('canvas.pickle', 'rb') as handle:
        canvas = pickle.load(handle)
    with open('canvas_snr.pickle', 'rb') as handle:
        canvas_snr = pickle.load(handle)
    with open('bkg.pickle', 'rb') as handle:
        bkg = pickle.load(handle)
    with open('bkg_snr.pickle', 'rb') as handle:
        bkg_snr = pickle.load(handle)
 
    catalog = fits.getdata("catalog.fits")
 
    wfm = bm.codedmask("../../../simulations/wfm_mask.fits", upscale_x=5)
    simfiles = bm.simulation_files("../../../simulations/galcenter")
    sdl_1a = bm.simulation(simfiles["cam1a"]["reconstructed"])
    sdl_1b = bm.simulation(simfiles["cam1b"]["reconstructed"])
 
    sources_radec_1a = [
        shift2equatorial(
            sdl_1a,
            wfm,
            shift_x,
            shift_y,
        ) for shift_x, shift_y, _ in sources["cam1a"]
    ]
 
    sources_indices_1a = [
        shift2pos(
            wfm,
            shift_x,
            shift_y,
        ) for shift_x, shift_y, _ in sources["cam1a"]
    ]
    sources_indices_1b = [
        shift2pos(
            wfm,
            shift_x,
            shift_y,
        ) for shift_x, shift_y, _ in sources["cam1b"]
    ]
 
    sources_indices = [
        shift2pos(
            wfm,
            sources["cam1a"][i][0],
            sources["cam1b"][i][0],
        ) for i in range(len(sources["cam1a"]))
    ]
    print(sources_indices_1a)
    print(sources_indices)
 
    catalog = catalog[catalog["AVG_FLUX"] > 0.1]
    source_names = []
    for source in sources_radec_1a:
        argsource = np.argmin(
            np.square(catalog["RA"] - source.ra) + np.square(catalog["DEC"] - source.dec)
        )
        name = catalog["NAME"][argsource]
        flux = catalog["AVG_FLUX"][argsource]
        source_names.append(name)
    print(source_names)
 
    CONTRAST = 1.0
    canvas_sum = [np.zeros_like(c) for c in canvas_snr]
    for c, csum in enumerate(canvas_sum):
        csum += canvas_snr[c]
        mask = canvas_snr[c] > 0
        csum[~mask] = bkg_snr[c][~mask] * CONTRAST
    plot(canvas_sum, sources_indices, source_names)