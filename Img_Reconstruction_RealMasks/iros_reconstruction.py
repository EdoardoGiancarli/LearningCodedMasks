"""
Module for testing IROS reconstruction.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm

from mbloodmoon import simulation_files, codedmask, simulation, iros
from mbloodmoon.images import compose, upscale

matplotlib.use('agg')
root_path = "/mnt/d/PhD_AASS/Coding/Images_fits/"


def plot(skyrecs, source_indices, source_names, title, upsc_y=8, dpi=300):
    composed, _ = compose(*skyrecs, strict=False)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12.8), dpi=dpi)
    for ((i, j), name) in zip(source_indices, source_names):
        ax.scatter(j, i * upsc_y + 53, s=30, facecolors="none", edgecolors="white", alpha=1., linewidth=.5)
        ax.text(j + 50 , i * upsc_y + 100, name, color="white", fontsize=4)
    im = ax.imshow(composed, vmax=np.quantile(composed, 0.9995), vmin=0., cmap="viridis")
    plt.colorbar(im, ax=ax, label='SNR', fraction=0.025, aspect=35, pad=0.02, shrink=0.33, location="bottom")
    plt.axis("off")
    plt.tight_layout()
    ax.set_title(title, fontsize=14, pad=8, fontweight='bold')
    plt.savefig(root_path + title.replace(' ', '').lower() + ".png")
    plt.close()



if __name__ == '__main__':
    pass






# end