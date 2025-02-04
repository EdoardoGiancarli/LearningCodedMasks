"""
Iterative Removal of Sources...
"""

import collections.abc as c

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from .skyrec import sky_encoding, sky_reconstruction, sky_snr
from .display import image_plot, sequence_plot


def argmax(x: np.array) -> tuple[int, int]:
    row, col = np.unravel_index(np.argmax(x), x.shape)
    return int(row), int(col)


def check_snr_distr(snr: np.array,
                    iteration: int,
                    ) -> None:
    
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    fig.tight_layout()
    ax.hist(snr.reshape(-1), bins=50, density= True,
            color='SkyBlue', edgecolor='b', alpha=0.7)
    ax.plot(x := np.linspace(-5, 5, 1000), norm.pdf(x),
            color="OrangeRed", label="Normal distr.")
    ax.set_xlabel("SNR", fontsize=12, fontweight='bold')
    ax.set_ylabel("density", fontsize=12, fontweight='bold')
    ax.set_title(f"SkyRec SNR Statistics, iter. {iteration + 1}",
                 fontsize=14, pad=8, fontweight='bold')
    ax.grid(visible=True, color="lightgray", linestyle="-", linewidth=0.3)
    ax.legend(loc='best')
    ax.tick_params(which='both', direction='in', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.show()


def show_snr_peaks(x: list,
                   y: list,
                   snr_threshold: int | float,
                   ) -> None:
    sequence_plot([y], [f"SkyRec Peaks over SNR $=$ {snr_threshold}"],
                  [x], ["iteration"], ["SNR peaks"], style=["scatter"])


def record_source(sources_log: dict,
                  pos: np.array,
                  counts: int | float,
                  snr_value: int | float,
                  ) -> dict:
    sources_log['sources_pos'].append(pos)
    sources_log['sources_counts'].append(counts)
    sources_log['sources_snrs'].append(snr_value)
    return sources_log

def _check_peaks(n: int) -> bool:
    check = n != 0
    return check

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

def IROS(n_iterations: int,
         skysnr: np.array,
         skymap: np.array,
         detector_img: np.array,
         snr_threshold: np.array,
         cam: object,
         show_snr_distr: bool = True,
         show_peaks_num: bool = True
         ) -> c.Iterable:
    
    f = cam.specs['real_open_fraction']
    
    def shadowgram(pos: tuple[int, int],
                   counts: int | float,
                   ) -> np.array:
        
        s = np.zeros(cam.sky_shape)
        s[*pos] = counts
        det = sky_encoding(cam.mask, s, cam.bulk)
        assert det.shape == cam.detector_shape

        return det

    def select_source(snr: np.array,
                      sky: np.array,
                      sources_dataset: dict,
                      iteration: int,
                      ) -> tuple[tuple[int, int], int]:
        
        peaks_pos = np.argwhere(snr > snr_threshold).T
        n_peaks = len(peaks_pos[0])

        print(
            f"Number of outliers with SNR(σ) over {snr_threshold} at iteration {iteration + 1}: {n_peaks}"
            )
        
        loc = argmax(sky)
        counts = sky[*loc]

        if n_peaks == 0:
            return loc, counts, n_peaks

        elif loc in sources_dataset['sources_pos']:
            print(f"Source pos ({loc[0]}, {loc[1]}) already recorded...")
            return loc, counts, n_peaks
        
        else:
            print(f"New source found at pos ({loc[0]}, {loc[1]})!")
            sources_dataset = record_source(sources_dataset, loc,
                                            counts/f, snr[*loc])
            return loc, counts, n_peaks
    
    def iros_iteration(detector: np.array,
                       source_shadowgram: np.array,
                       ) -> tuple[np.array, np.array, np.array]:
        
        new_detector = detector - source_shadowgram
        new_skyrec, new_skyvar = sky_reconstruction(cam.mask, cam.decoder,
                                                    new_detector, cam.bulk)
        new_skysnr = sky_snr(new_skyrec, new_skyvar)

        return new_detector, new_skyrec, new_skysnr
    
    # init log variables
    sources_dataset = {'sources_pos': [],
                       'sources_counts': [],
                       'sources_snrs': []}
    snr_peaks_list = [0]*n_iterations

    # IROS loop
    for i in range(n_iterations):

        if show_snr_distr:
            check_snr_distr(skysnr, i)
        
        loc, counts, n_peaks = select_source(skysnr, skymap, sources_dataset, i)

        if show_peaks_num:
            snr_peaks_list[i] = n_peaks
            show_snr_peaks(np.arange(n_iterations), snr_peaks_list, snr_threshold)
        
        if _check_peaks(n_peaks):
            
            source_shadowgram = shadowgram(loc, counts)
            detector_img, skymap, skysnr = iros_iteration(detector_img, source_shadowgram)
            
            yield sources_dataset, skymap, skysnr
        
        else:
            print(f"No sources detected with SNR over {snr_threshold}...")
            return sources_dataset, skymap, skysnr



def iros_skyrec(sky_image: np.array,
                sources_pos: list[tuple],
                sources_dataset: dict,
                cam: object) -> None:

    sky = np.zeros(cam.sky_shape)
    for idx, pos in enumerate(sources_dataset['sources_pos']):
        sky[*pos] = sources_dataset['sources_counts'][idx]

    image_plot([sky_image, sky],
               ["Simulated Sky", "IROS Sky Reconstruction"],
               cbarlabel=["counts"]*2, cbarcmap=["viridis"]*2,
               simulated_sources=[sources_pos, sources_dataset['sources_pos']])


# end