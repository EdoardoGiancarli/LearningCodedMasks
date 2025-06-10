"""
IROS output plotting.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from mbloodmoon.mask import CodedMaskCamera
from mbloodmoon.mask import model_sky

__all__ = [
    "crop", "plot_distr", "plot_sequence",
    "enhance_slices", "make_sky", "plot_sky",
]

RCPARAMS = {
    'font.family': 'sans-serif',
    'font.weight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
}
mpl.rcParams.update(RCPARAMS)



def crop(
    image: np.array,
    pos: tuple[int, int],
    cropping: tuple[int, int],
    strict: bool = True,
) -> np.array:
    """
    Crops 2D array at given position and with given cropping.

    Args:
        image (np.array):
            2D array to crop.
        pos (tuple[int, int]):
            Center position for cropping.
        cropping (tuple[int, int]):
            Size of the cropping along (y, x).
        strict (bool, optional (default=True)):
            If `False` allows for the cropping to be adapted
            wrt the array edges when they are exceeded.
    
    Returns:
        output (np.array): Cropped 2D array (shape twice the `cropping`).
    
    Raises:
        ValueError: If cropping is not a positive int tuple.
        IndexError: If cropping wrt indexes exceeds 2D array edges
                    (only if `strict` is `True`).
    
    Notes:
        - Negative indexes are allowed.
    """
    n, m = image.shape
    y, x = pos
    cy, cx = cropping

    if cy <= 0 or cx <= 0:
        raise ValueError("Cropping must be a tuple of positive integers.")

    flagx = (((0 <= x - cx) and (x + cx < m)) or ((cx - x <= m) and (x + cx < 0)))
    flagy = (((0 <= y - cy) and (y + cy < n)) or ((cy - y <= n) and (y + cy < 0)))
    
    if not (flagx and flagy):
        if not strict:
            if not flagx:
                cx = min(x - 1, m - x - 2) if x > 0 else min(x + m + 1, -x - 1)
            if not flagy:
                cy = min(y - 1, n - y - 2) if y > 0 else min(y + n + 1, -y - 1)
            print(f"Cropping {cropping} at pos {pos} exceeds array edges, new cropping: {cy, cx}")
        else:
            raise IndexError(f"Cropping {cropping} at pos {pos} exceeds array edges.")
    
    y1, y2 = y - cy, y + cy
    x1, x2 = x - cx, x + cx
    return image[y1 : y2, x1 : x2]


def plot_distr(
    arr: np.array,
    title: str,
    bins: int | Sequence = 50,
    xlabel: str = None,
    pdf_distr: tuple[np.array, str] = None,
    cut: tuple[float] = None,
) -> None:
    """
    Plots the histogram of the values inside the input array.
    Array can be N-dim, it will be flattened for the plot.

    Args:
        arr (np.array):
            Input array, could be N-dim.
        title (str):
            Plot title.
        bins (int, Sequence, optional (default=50)):
            Histogram bins.
        xlabel (str):
            Label for x-axis and for legend.
        pdf_distr (np.array, tuple[np.array, str], optional (default=None)):
            Input PDF distribution for comparison.
        cut (float | tuple[float], optional (default=None)):
            Edge values for the distribution.

    Notes:
        - `pdf_distr` could be used as:
        >>> pdf_distr = array                # y
        >>> pdf_distr = (array, array)       # (x, y)
        >>> pdf_distr = (array, array, str)  # (x, y, distr. name)

        - for specific info, refer to matplotlib.pyplot docs
    """
    if np.ndim(arr) > 1: arr = arr.reshape(-1)

    if cut is None: cut = (min(arr), max(arr))
    elif isinstance(cut, float): cut = (-cut, cut)
    arr = np.delete(arr=arr, obj=np.argwhere(arr < cut[0]))
    arr = np.delete(arr=arr, obj=np.argwhere(arr > cut[1]))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.tight_layout()
    ax.hist(
        arr, bins=bins, density=True, color='SkyBlue', edgecolor='b',
        alpha=0.7, label=f"{xlabel} distr." if xlabel else None,
    )
    if pdf_distr is not None:
        x = pdf_distr[0] if isinstance(pdf_distr, tuple) else np.arange(len(pdf_distr))
        y = pdf_distr[1] if isinstance(pdf_distr, tuple) else pdf_distr
        distr = pdf_distr[2] if pdf_distr[2] else "th."
        ax.plot(x, y, color="OrangeRed", label=f"{distr} distr.")
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel("density", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, pad=8, fontweight='bold')
    ax.grid(visible=True, color="lightgray", linestyle="-", linewidth=0.3)
    ax.tick_params(which='both', direction='in', width=2, length=7 if 'major' else 4)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    if pdf_distr is not None: ax.legend(loc='best')
    plt.show()


def plot_sequence(
    input_sequence: list[Sequence],
    title: list[str],
    x: list[Sequence] = None,
    xlabel: list[str] = None,
    ylabel: list[str] = None,
    color: list[tuple[str, str]] = None,
    style: list[str] = None,
    offsetx: list[tuple[float, float]] = None,
    offsety: list[tuple[float, float]] = None,
) -> None:
    """
    Plots multiple sequences as subplots.

    Args:
        input_sequence (list[Sequence]):
            List of sequences to plot.
        title (list[str]):
            Titles for each subplot.
        x (list[Sequence], optional (default=None)):
            X-axis values for each sequence.
        xlabel (list[str], optional (default=None)):
            Labels for x-axes.
        ylabel (list[str], optional (default=None)):
            Labels for y-axes.
        color (list[tuple[str, str]], optional (default=None)):
            Colors (fill, edge) for each plot.
        style (list[str], optional (default=None)):
            Plot styles ('bar' or 'scatter').
        offsetx (list[tuple[float, float]], optional (default=None)):
            X-axis offset adjustments.
        offsety (list[tuple[float, float]], optional (default=None)):
            Y-axis offset adjustments.
    
    Notes: TODO
        - It is possible to have multiple plots in the same subplots:
        >>> input_sequence = [(y11, y12, ...), (y21, y22), ...]
    """
    def _handle_subplots(nplots, spacing=0.27):
        """Creates and configures subplots."""
        size = 6
        fig_size = (size * nplots + 1, size) if nplots > 1 else (size, size)
        fig, axes = plt.subplots(1, nplots, figsize=fig_size)
        fig.tight_layout()
        fig.subplots_adjust(wspace=spacing * 5 / size)
        return fig, (axes if nplots > 1 else [axes])

    def _set_labels(ax, xlabel, ylabel, title):
        """Configures labels and titles."""
        ax.set_xlabel(xlabel or "", fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel or "", fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, pad=8, fontweight='bold')

    def _set_ticks(ax):
        """Configures grid and tick properties."""
        ax.grid(visible=True, color="lightgray", linestyle="-", linewidth=0.2, alpha=0.70)
        ax.tick_params(which='both', direction='in', width=2, length=7 if 'major' else 4)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

    nplots = len(input_sequence)
    x = x or [None] * nplots
    xlabel = xlabel or [None] * nplots
    ylabel = ylabel or [None] * nplots
    color = color or [('OrangeRed', 'r')] * nplots
    style = style or ['bar'] * nplots
    offsetx = offsetx or [(None, None)] * nplots
    offsety = offsety or [(None, None)] * nplots

    fig, axes = _handle_subplots(nplots)
    for i, ax in enumerate(axes):
        x_values = x[i] if x[i] is not None else np.arange(len(input_sequence[i]))
        fcolor, ecolor = color[i]
      
        if style[i] == 'scatter':
            ax.scatter(x_values, input_sequence[i], c=fcolor, edgecolors=ecolor, 
                       s=70, alpha=0.8, linewidths=2)
        else:
            ax.bar(x_values, input_sequence[i], width=1, facecolor=fcolor, 
                   edgecolor=ecolor, linewidth=1, alpha=0.70)

        ax.set_xlim(offsetx[i][0], offsetx[i][1])
        ax.set_ylim(offsety[i][0], offsety[i][1])
        _set_labels(ax, xlabel[i], ylabel[i], title[i])
        _set_ticks(ax)
    plt.show()


def enhance_slices(
    sky: np.array,
    pos: tuple[int, int],
    crp: tuple[int, int] = (40, 40),
    source: str = None,
    cameraID: str = None,
) -> None:
    """
    Displays the counts distribution along the y-dim and the x-dim
    for a given sky and position.

    Args:
        sky (np.array): 2D array of the sky.
        pos (tuple[int, int]): Indexes of the source.
        crp (tuple[int, int], optional (default=(30, 30))): Half-size of the cropping.
        source (str, optional (default=None)): Source name.
        cameraID (str, optional (default=None)): WFM camera name (e.g. 'CAM1A').
    
    Notes:
        - `sky` is cropped to enhance the source counts distribution.
    """
    cropped = crop(sky, pos, crp)
    xslice, yslice = cropped[crp[0], :], cropped[:, crp[1]]
    plot_sequence(
        input_sequence=[xslice, yslice],
        title=[
            f"{source.upper() if source else "Source"} {ax}-axis Slice - {cameraID.upper() if cameraID else ""}"
            for ax in ("X", "Y")
        ],
        x=[np.arange(len(s)) - len(s) // 2 for s in (xslice, yslice)],
        xlabel=["x", "y"],
        ylabel=["counts", "counts"],
        color=[('white', 'r'), ('white', 'r')],
        offsety=[(np.min(s) - 10, np.max(s) + 10) for s in (xslice, yslice)]
    )



"""
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⢛⣛⣛⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⢿⠿⣿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⣛⡩⣔⠶⣚⡵⢫⠒⡤⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠲⡌⠛⣮⠵⣋⢮⠽⣹⢻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⠿⠿⠿⣿⠿⢋⡴⡺⢥⣛⡬⢏⡗⢪⢅⠫⡔⣳⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢳⢌⡓⡈⢯⡝⢮⢏⡳⣝⡺⣜⡻⢿⠟⣋⣭⣥⢶⢶⣞⣷⣻⣞⡷⡇⢸⢧⢳⡝⣣⢧⡛⠭⣌⠣⣌⢓⡰⢧⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣏⠦⡱⢱⡀⢫⢏⡾⣱⢣⢟⡴⢋⣰⣾⣻⢾⣭⣟⣯⣞⡷⣽⠾⣽⡇⣹⢎⣗⡺⢵⢣⡙⠲⣄⠳⡐⢎⡔⣣⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣯⠖⣡⠣⡜⣀⢻⢲⡭⣫⠎⣠⣾⣻⢶⢯⣟⡾⣞⡵⣯⣻⣽⡻⣷⣳⢬⢳⢎⡵⣋⠦⣑⠣⡜⢢⡙⢢⠆⡳⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡽⢠⠓⡜⢤⠘⣧⠳⠃⣰⣟⡾⣳⢯⣟⡾⣽⣫⣟⡷⣻⢶⣻⣗⡿⡬⢏⣞⡱⡏⠴⡡⢓⠬⡱⢌⠣⡜⣱⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣟⢧⡙⢜⢢⠂⡽⠁⣼⣻⢾⡽⢯⠿⠾⠙⠓⠛⠚⡙⠛⠋⠷⠺⠽⠿⣍⢶⡹⣜⢡⠣⢍⠲⣑⠪⡱⣜⣣⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⢎⡜⢢⠑⡃⢘⣉⢥⣡⠴⢦⡖⡶⢫⣟⡹⣏⡝⣏⢟⡺⢵⢳⡺⣜⣣⠟⣬⣓⠯⣎⠷⣤⠳⣝⣲⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠟⢈⡐⣤⢖⠯⣎⢷⣩⢞⢧⡝⣞⡳⣬⢳⢎⡽⢪⡝⡞⣭⠶⣹⢲⢭⡞⡵⢎⡻⣜⢳⢎⡟⣬⣓⢮⣛⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⢋⣤⣒⢮⢏⡽⣚⢮⣛⡼⢣⡞⡭⣞⢼⢣⡳⣭⢞⣭⢞⣧⣫⣝⡶⣋⢧⣛⣦⣛⡼⣫⠵⣋⢾⣩⢞⡱⣎⡗⢮⣓⢮⡝⣻⢿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⣉⡴⣺⠝⣦⡝⣺⢎⣵⣫⢶⣽⢺⣗⡿⣽⣼⣺⡌⢟⣹⢛⣾⡈⢷⡛⣞⣱⡷⠿⠶⣯⣟⡷⢹⣻⣽⣳⣞⣮⢷⣱⣞⢧⣝⣲⡹⢵⠺⣜⣛⠿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⣉⡴⡺⢵⣙⣶⢻⣶⣻⢰⣟⡾⣽⢫⣜⢣⣾⠟⡁⠤⣈⠻⣌⢲⣋⣾⢡⠚⣵⡾⢋⡔⠠⢃⠤⡙⣿⢨⢳⢬⠳⡜⣽⣞⡷⢯⣟⡾⣳⣟⣯⣟⡶⣭⣟⣼⣹
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⣡⢶⣹⢶⣻⢯⣟⣾⡻⠞⣧⠸⢦⣽⠰⣏⡜⣯⡟⠠⢾⡃⠀⠢⢹⡶⣈⢻⢬⠳⣬⠁⢾⣿⠀⡐⠂⠡⠘⢨⢧⢫⡝⣱⣟⡮⣿⣿⣾⣽⣷⣿⣶⣯⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣯⣾⣽⣯⣯⣿⣭⣿⣮⣷⣶⣷⣶⠸⣓⣾⡇⢺⡜⡷⡏⠀⢹⣟⠀⠡⢀⡿⣉⢎⠦⣙⠤⡁⠘⣿⡆⠀⠌⠐⡈⢼⢣⢏⡼⢳⡯⣝⡫⣝⢬⠳⣭⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⡹⡖⣿⣆⡙⡷⢡⡈⠈⣿⠀⣁⠎⡴⢡⢎⡱⢊⡜⡡⢆⠻⠇⣀⠌⣔⠂⡽⣚⣬⢳⡏⣶⣥⣷⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⣛⣫⡅⢿⢾⣽⣳⠡⢎⡱⢂⢖⡡⢞⡰⢃⢎⡴⣉⠖⡱⢊⠵⣉⠦⡙⠴⢈⡷⡱⢎⣳⣾⣔⠺⡬⣍⢟⡻⢿⣿⣿⣿⣿⣿⣿⣿⣿
⡃⢞⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣟⣰⣿⢿⣻⡿⡘⣯⢶⢯⣿⣔⢡⢋⠦⡑⢎⠴⡩⢞⡰⡙⢎⡱⣉⠖⡡⢎⡱⠃⣼⠲⣝⣣⣿⣰⢫⢿⣷⣾⣾⣵⣿⣿⣿⣿⣿⣿⣿⣿⣿
⡿⣌⢦⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⣴⣷⡘⣯⡟⢞⡽⣳⢎⣖⡩⢎⠲⡑⢎⠴⣉⠦⡱⢌⢎⣱⢪⣴⠃⣎⡗⢮⠭⣝⢳⣭⠶⣹⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣞⡽⢮⡗⣌⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣼⣿⣿⣷⣌⣷⠈⣖⠡⢎⣩⢙⣋⠳⡛⠼⢲⠵⣪⠵⡚⢎⠥⣃⠖⣸⠱⣊⠦⡹⢼⣿⣿⣷⣥⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣝⡞⣯⢾⡹⣞⡜⢻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⢜⣛⢦⡙⢎⢶⣡⡝⣌⣣⢚⡤⣣⠝⢮⡙⠦⡙⣆⠳⣌⠲⣍⡳⢭⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⡼⣝⡮⣗⢯⡳⢯⣗⢮⣙⡻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢿⠿⠋⣤⢏⡞⡬⡝⣎⠦⡑⠮⡕⣎⠧⡓⢥⢋⢦⡙⡶⡹⢌⠣⢆⡻⣬⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⢽⡺⣵⢫⣗⣻⢳⡞⣯⢽⣹⠶⣭⡭⣏⣟⣛⣛⢟⡻⠟⡟⣛⠻⣩⢋⡭⣡⢣⡖⣴⡜⣶⠖⡛⣴⢫⡜⡵⡹⢬⠳⣍⠲⣉⢆⡣⡝⢦⢫⠶⡙⠴⡑⢎⣱⡏⣽⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣯⢳⡭⢷⡺⣵⢻⡼⣣⠿⣜⢯⡳⣝⠾⣜⡧⢯⢯⣝⡻⡽⣭⡻⡵⢯⡽⣭⢗⡻⠮⡙⢄⡊⢴⢚⣯⣟⡶⣯⣭⡳⣍⡗⢮⣙⠶⡹⣍⢞⡲⣍⡶⣽⣞⣯⣗⢸⢯⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣼⢫⣞⢯⣳⢭⡷⣹⢧⡟⣽⢺⡝⣾⡹⢧⡻⣝⡞⣮⢗⡻⢶⡽⣹⡳⠝⡊⢍⣰⢡⠞⡦⢍⣸⠎⣷⢯⣽⣳⢯⡿⠙⠞⠷⠯⢾⡵⣞⣾⢳⣯⢟⡷⣞⣧⢿⢨⡷⣞⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⠞⢧⡻⣎⡷⣫⢞⣧⢻⡼⣳⢏⡾⣵⢫⣏⢷⣹⠾⡱⠏⢛⢃⡍⣥⢢⢳⡙⣎⠶⣩⢞⡹⢠⢯⡇⢿⣻⡼⣏⠋⣤⢛⠾⣱⢞⡲⢼⣙⢾⣛⡾⣏⡿⣽⣺⢿⠠⣟⡼⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣩⠔⣤⢡⢉⡙⢚⠚⠳⡙⢣⠛⡚⣑⢋⡌⣥⢢⠦⣕⠺⣍⠞⡜⢦⣋⠶⡹⣌⡳⡱⢎⠅⣾⣣⢏⢾⡳⠋⣤⠞⣥⠏⣻⠜⣮⣙⡞⡼⣊⢿⡽⣽⡽⣞⣳⢯⡃⣯⢳⣝⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣡⠻⣔⢫⢎⡵⣋⢞⡱⣍⢧⢫⠵⣩⠞⡜⢦⣋⠞⣬⢓⢮⡙⣎⠳⣌⡳⡱⢎⡵⣙⠎⣸⢧⣛⠎⣢⠴⣛⢦⢻⡜⢢⢏⡽⢆⣗⢺⡱⡝⣎⢿⣵⣻⣭⡟⣯⡕⣫⢟⡼⣹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣥⢛⡬⣓⢎⠶⣩⢎⡵⣊⢎⢧⡛⡴⣋⢞⡱⢎⡝⢦⣋⠶⡹⣌⠳⣥⠳⣙⣮⣶⠃⣴⢏⢋⡤⡞⣥⢻⢬⡓⡧⠞⣨⢳⢎⢯⡜⣣⡝⢮⠵⣎⢷⣳⢯⣞⡷⣇⢻⢮⣝⡳⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣌⠳⣜⡱⢎⡳⡱⢎⠶⣩⠞⢦⡙⢶⡩⢎⡵⢫⡜⣣⢎⡳⡱⢎⣳⣬⣿⣿⣿⠃⣜⣣⢎⣏⢶⡹⡬⢧⢳⡹⢼⠁⡾⡱⣎⣳⢚⡵⣚⡭⢞⣬⠳⣯⣛⡾⣽⣹⠌⣷⢺⡽⣹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⢎⡳⣌⠳⣍⠶⣙⢎⡳⢥⡛⢦⡹⢦⡙⢮⡜⣣⠞⡥⢎⣵⣽⣾⣿⣿⣿⣿⣿⡇⣼⠲⣍⡞⢦⡳⣙⢮⣣⠝⡇⢸⢣⢗⡱⣎⢳⢎⡵⣚⣭⢒⡟⡜⣯⢷⢯⣽⣣⢹⢧⡻⣵⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣫⢖⣩⠳⣌⠳⣍⢮⡱⢣⡝⢦⡙⢦⡝⢦⣹⣴⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⢛⡼⣜⢣⡗⡭⣖⢣⡻⢀⡯⣓⢮⠳⡼⣩⠞⡼⡱⢎⡽⣸⣱⢹⣞⣯⢶⣳⠌⣷⡹⣎⢯⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣡⠞⡴⢫⡜⡳⠜⣦⣙⣣⣼⣧⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡴⣩⠶⣙⢶⣩⢞⡕⣨⢳⡍⡮⡝⠶⣍⠾⣱⢭⣋⠶⣣⢎⡷⣞⣧⢿⣹⡎⢲⡽⣚⢧⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣥⣿⣶⣷⣾⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢋⣴⠿⡽⣞⣭⡟⣆⣉⠓⠺⠱⡽⡹⢬⣓⠧⡞⣬⣳⢽⣫⡽⣞⠾⣭⢷⣻⠄⣿⡹⣎⢷⣹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⢡⡾⣭⠿⣽⣹⢶⡻⣝⡯⣟⣷⡳⢶⣬⣥⢬⣳⡽⣞⡵⣯⣳⢯⡽⣻⣝⡾⣭⡗⢬⡳⣝⢮⡳⡽⣿⣿⣿⣿⣿⣿⣿⣿⣿
"""


def make_sky(
    data: dict,
    cameraID: str,
    camera: CodedMaskCamera,
    background: np.array = None,
    vignetting: bool = True,
    psfy: bool = True,
) -> np.array:
    """
    Generates a skymap with the info retrieved by IROS.

    Args:
        data (dict):
            Database with parameters computed from IROS.
        camerasID (tuple[str]):
            Cameras of the WFM being processed.
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        background (np.array, optional (default=None)):
            Background for the reconstructed sky.
        vignetting (bool, optional (default=True)):
            Simulates vignetting effects.
        psfy (bool, optional (default=True)):
            Simulates detector reconstruction effects.
    
    Returns:
        sky (np.array):
            Resulting sky from IROS image reconstruction.
    
    Raises:
        ValueError: If `background` has an invalid shape.
    
    Notes:
        - The input database must contain at least the sources (i) sky
          coords shifts in [mm] wrt the camera optical axis; (ii) their
          fluences [ph] and (iii) the px indexes.
        - The background is optional (e.g. could be a Poissonian distr.
          of photons decoded from the detector or the IROS residuals).
    """
    def valid_BG(bkg: np.array) -> bool:
        """Checks background shape."""
        if not (bkg.shape == camera.shape_sky):
            raise ValueError(f"Background must have same sky shape {camera.shape_sky}.")
        return True

    def make_source(
        shiftx: float,
        shifty: float,
        fluence: float,
        pos: tuple[int, int],
        cropping: tuple[int, int],
    ) -> np.array:
        """Generates a source shadowgram and returns a crop of the source."""
        model = model_sky(camera, shiftx, shifty, fluence, vignetting, psfy)
        return crop(model, pos, cropping, strict=False)

    if background is None:
        sky = np.zeros(camera.shape_sky)
    elif valid_BG(background):
        sky = background
    
    upx, upy = camera.upscale_f
    cropx, cropy = int(10 * (1 + upx / 3)), int(50 * (1 + upy))
    
    for shiftx, shifty, fluence, x, y in zip(
        data[cameraID]["shift_x"]["data"],
        data[cameraID]["shift_y"]["data"],
        data[cameraID]["fluence"]["data"],
        data[cameraID]["x"]["data"],
        data[cameraID]["y"]["data"],
    ):
        modeled = make_source(
            shiftx=shiftx,
            shifty=shifty,
            fluence=fluence,
            pos=(y, x),
            cropping=(cropy, cropx)
        )
        p, q = modeled.shape
        sky[y - p // 2 : y + p // 2, x - q // 2 : x + q // 2] = modeled
    
    return sky


PCAM = {
    "sourcename_fs": 3.8,
    "title_fs": 12,
    "title_pad": 8,
    "txt_fw": "bold",
    "txt_color": "white",
}


def plot_sky(
    sky: np.array,
    title: str,
    sources_ID: list[str] = None,
    sources_pos: list[tuple[int, int]] = None,
    highlight_pos: bool = True,
    save_to: str | Path = None,
) -> None:
    """
    Makes a plot for the given sky as seen by the WFM.

    Args:
        sky (np.array):
            2D array of the sky.
        title (str):
            Title of the plot.
        sources_ID (list[str], optional (default=None)):
            Names of the sources.
        sources_pos (list[tuple[int, int]], optional (default=None)):
            Position of the sources.
        highlight_pos (bool (default=True)):
            If True, the sources will be circled.
        save_to (str | Path (default=None)):
            Path to save the plot.

    Notes:
        - `sources_ID` should contains the names of the sources.
        - `sources_pos` should contains the px position (y, x) of the given sources.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7.5), dpi=200)
    
    if (sources_ID is not None) and (sources_pos is not None):
        for name, pos in zip(sources_ID, sources_pos):
            ax.text(
                pos[1] + 50, pos[0] + 150, name.upper(), color=PCAM["txt_color"],
                fontsize=PCAM["sourcename_fs"], fontweight=PCAM["txt_fw"],
            )
    if (sources_pos is not None) and highlight_pos:
        for pos in sources_pos:
            ax.scatter(
                pos[1], pos[0], s=30, facecolors="none",
                edgecolors="white", alpha=1, linewidth=0.5,
            )

    ax.imshow(sky, cmap='inferno', vmax=np.quantile(sky, 0.9995))
    ax.set_title(
        title, fontsize=PCAM["title_fs"], pad=PCAM["title_pad"], fontweight=PCAM["txt_fw"]
    )
    plt.axis("off")
    plt.tight_layout()
    if save_to is not None: plt.savefig(save_to)
    plt.show()


# end