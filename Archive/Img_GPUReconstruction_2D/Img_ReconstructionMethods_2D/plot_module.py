#### Script for plot ####

import numpy as np
import collections.abc as c

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib.ticker as ticker
#from matplotlib.gridspec import GridSpec as gridspec
#from matplotlib.colors import Normalize

labelsize = 12

params = {'font.family': 'sans-serif',
          'font.weight': 'bold',
          'xtick.labelsize': labelsize,
          'ytick.labelsize': labelsize}

mpl.rcParams.update(params)
#plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.turbo_r(np.linspace(0,1,5)))



def sequence_plot(input_sequence: list[c.Sequence],
                  title: list[str],
                  x: list[c.Sequence] = None,
                  xlabel: list[str] = None,
                  ylabel: list[str] = None,
                  color: list[(str, str)] = None,
                  style: list[str] = ["bar"],
                  simulated_sources: list[tuple[int, int, int, float]] = None,
                  ) -> None:
    """Plot(s) of the input 1D array(s)."""

    # number of plots
    n = len(input_sequence)

    # handle optional arguments
    x = x or [None]*n
    xlabel, ylabel = xlabel or [None]*n, ylabel or [None]*n
    color, style = color or [('OrangeRed', 'r')]*n, style or [None]*n
    simulated_sources = simulated_sources or [None]*n

    # create subplots
    fig, axes = _handle_subplots(n, 0.27)

    for i, ax in enumerate(axes):

        # check  plot_type
        if np.ndim(input_sequence[i]) != 1:
            raise ValueError(f"Invalid input_sequence for plot {i}. Must be a 1D array.")

        # plot
        if x[i] is not None:
            phase = x[i]
        else:
            phase = np.arange(len(input_sequence[i]))

        if color[i] is None:
            color[i] = ('OrangeRed', 'r')
            
        if style[i] == 'scatter':
            ax.scatter(phase, input_sequence[i], c=color[i][0],
                       linewidths=2, edgecolor=color[i][1], s=70, alpha=0.8)
        elif style[i] == 'bar':
            ax.bar(phase, input_sequence[i], width=1, color=color[i][0],
                   edgecolor=color[i][1], linewidth=3, alpha=0.70)
        
        if simulated_sources[i] is not None:
            _show_sources_pos(ax, input_sequence[i], simulated_sources[i])
        
        offsetx = 1
        ax.set_xlim(phase[0] - offsetx, phase[-1] + offsetx)
        ax.set_ylim(np.min(input_sequence[i]) - 1, np.max(input_sequence[i]) + 1)

        # styling
        _handle_labels(ax, xlabel[i], ylabel[i], title[i])
        _handle_ticks(ax)

    plt.show()


def image_plot(input_image: list[c.Sequence],
               title: list[str],
               xlabel: list[str] = None,
               ylabel: list[str] = None,
               cbarlabel: list[str] = None,
               cbarvalues: list[c.Sequence] = None,
               cbarcmap: list = None,
               simulated_sources: list[c.Sequence[int, int]] = None,
               ) -> None:
    """Plot(s) of the input 2D array(s)."""

    # number of plots
    n = len(input_image)

    # handle optional arguments
    xlabel, ylabel = xlabel or [None]*n, ylabel or [None]*n
    cbarlabel, cbarvalues = cbarlabel or [None]*n, cbarvalues or [None]*n
    cbarcmap = cbarcmap or [ListedColormap(["DeepPink", "Orange"])]*n
    simulated_sources = simulated_sources or [None]*n

    # create subplots
    fig, axes = _handle_subplots(n, 0.35)

    for i, ax in enumerate(axes):

        # check  plot_type
        if np.ndim(input_image[i]) != 2:
            raise ValueError(f"Invalid input_image for plot {i}. Must be a 2D array.")

        # plot
        img = ax.imshow(input_image[i], cmap=cbarcmap[i], origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad=0.15)
        cbar = fig.colorbar(img, cax=cax, ax=ax, orientation='vertical',
                            ticks=cbarvalues[i] or None)
        
        if simulated_sources[i] is not None:
            for j, k in simulated_sources[i]:
                ax.scatter(k, j, marker='o', linewidths=1.75, facecolor='None',
                           edgecolor='white', s=35, alpha=0.8)

        if cbarlabel[i]:
            cbar.set_label(cbarlabel[i], fontsize=labelsize, fontweight='bold')
        cbar.ax.tick_params(labelsize=labelsize-1)

        ax.set_facecolor('lightgray')

        # styling
        _handle_labels(ax, xlabel[i], ylabel[i], title[i])
        _handle_ticks(ax)

    plt.show()




def _handle_subplots(n, w):
    """Subplots customization."""

    size = 4.5
    figsize = (size*n + 1, size) if n > 1 else (size, size)
    fig, axs = plt.subplots(1, n, figsize=figsize)
    fig.tight_layout()
    fig.subplots_adjust(wspace=w*5/size)
    axes = axs.flat if n > 1 else [axs]

    return fig, axes

def _handle_labels(ax, xlabel, ylabel, title):
    """Labels customization."""

    ax.set_xlabel(xlabel if xlabel else "", fontsize=labelsize, fontweight='bold')
    ax.set_ylabel(ylabel if ylabel else "", fontsize=labelsize, fontweight='bold')
    ax.set_title(title, fontsize=14, pad=8, fontweight='bold')

def _handle_ticks(ax):
    """Ticks customization."""
    
    ax.grid(visible=True, color="lightgray", linestyle="-", linewidth=0.3)
    ax.tick_params(which='both', direction='in', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

def _show_sources_pos(ax, reconstr_sources, simulated_sources):
    """Shows the initialized sources position."""

    idx, x, y, offset = simulated_sources

    ax.text(idx, reconstr_sources[idx] + offset,
            f"$\\hat{{S}}_{{{x}{y}}}$", fontsize=10,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.25'))

    ax.scatter(idx, reconstr_sources[idx], marker='s', c='DeepSkyBlue',
                linewidths=2, edgecolor='b', s=40, alpha=0.8)

def _test():
    """Tests plot functions."""

    l = 15

    y_seq = np.random.randint(0, 21, l)
    s_seq = [0]*l
    for i in [5, 7, 14]: s_seq[i] = y_seq[i]
    n_seq = 2
    sequence_plot([y_seq]*n_seq,
                  ["Title"]*n_seq,
                  x=[np.arange(len(y_seq))]*n_seq,
                  xlabel=["x values"]*n_seq,
                  ylabel=["y values"]*n_seq,
                  color=[("LawnGreen", "g")]*n_seq,
                  style=["bar"]*n_seq,
                  simulated_sources=[(s_seq, -2)]*n_seq)

    y_img = np.random.randint(0, 21, (l, l))
    n_img = 2
    image_plot([y_img]*n_img,
               ["Title"]*n_img,
               xlabel=["x values"]*n_img,
               ylabel=["y values"]*n_img,
               cbarlabel=["cbar values"]*n_img,
               cbarcmap=[ListedColormap(["DodgerBlue", "DeepSkyBlue"])]*n_img)


# end