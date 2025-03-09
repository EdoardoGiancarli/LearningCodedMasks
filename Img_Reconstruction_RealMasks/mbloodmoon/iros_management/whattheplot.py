"""
Plotting...
"""

import numpy as np
import collections.abc as c

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib.ticker as ticker
#from matplotlib.gridspec import GridSpec as gridspec
#from matplotlib.colors import Normalize
from mbloodmoon.images import argmax
import mbloodmoon as bm

labelsize = 12
params = {'font.family': 'sans-serif',
          'font.weight': 'bold',
          'xtick.labelsize': labelsize,
          'ytick.labelsize': labelsize}

mpl.rcParams.update(params)


def crop(
    img: np.array,
    pos: tuple[int, int],
    cropping: tuple[int, int],
) -> np.array:
    y1, y2 = pos[0] - cropping[0], pos[0] + cropping[0]
    x1, x2 = pos[1] - cropping[1], pos[1] + cropping[1]
    return img[y1 : y2, x1 : x2]


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
    plt.savefig(name + '.png')
    plt.close()


def plot_skyrec(skyrecs, title, source_indices=None, source_names=None, dpi=200, upsc_y=8):
    composed, _ = bm.compose(*skyrecs, strict=False)
    fig, ax = plt.subplots(1, 1, figsize=(8, 10), dpi=dpi)
    if source_indices is not None and source_names is not None:
        for ((i, j), name) in zip(source_indices, source_names):
            ax.scatter(j, i * upsc_y + 53, s=30, facecolors="none", edgecolors="white", alpha=1., linewidth=.5)
            ax.text(j + 50 , i * upsc_y + 100, name, color="white", fontsize=4)
    im = ax.imshow(composed, vmax=np.quantile(composed, 0.9995), vmin=0., cmap="viridis")
    plt.colorbar(im, ax=ax, label='SNR', fraction=0.025, aspect=35, pad=0.02, shrink=0.33, location="bottom")
    ax.set_title(title, fontsize=12, pad=8, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(title.replace(' ', '_').lower() + ".png")
    plt.close()



# end