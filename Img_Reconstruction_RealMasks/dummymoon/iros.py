"""
Iterative Removal of Sources...
"""

import collections.abc as c

import numpy as np
from scipy.stats import norm

from .skyrec import sky_encoding, sky_reconstruction, skyrec_norm
from .skyrec import sky_snr, show_snr_distr
from .display import image_plot, sequence_plot


def argmax(x: np.array) -> tuple[int, int]:
    row, col = np.unravel_index(np.argmax(x), x.shape)
    return int(row), int(col)


def snr_over_threshold(y: list,
                       snr_threshold: int | float,
                       ) -> None:
    sequence_plot([y], [f"SkyRec Peaks over SNR $=$ {snr_threshold}"],
                  [list(range(len(y)))], ["iteration"], ["SNR peaks"], style=["scatter"])


def record_source(sources_log: dict,
                  pos: np.array,
                  counts: int | float,
                  var: int | float,
                  snr: int | float,
                  ) -> dict:
    sources_log['sources_pos'].append(pos)
    sources_log['sources_counts'].append(counts)
    sources_log['sources_stds'].append(np.sqrt(var))
    sources_log['sources_snrs'].append(snr)
    return sources_log


def get_shadowgram(pos: tuple[int, int],
                   counts: int | float,
                   cam: object,
                   ) -> np.array:
        
    s = np.zeros(cam.sky_shape)
    s[*pos] = counts
    det = sky_encoding(s, cam)
    assert det.shape == cam.detector_shape
    return det


def select_source(sky: np.array,
                  var: np.array,
                  snr: np.array,
                  threshold: int | float,
                  sources_dataset: dict,
                  ) -> tuple[tuple[int, int], int]:
    
    def _snr_over_thres_pos():
        snr_peaks = np.argwhere(snr > threshold).T
        return np.dstack((snr_peaks[0], snr_peaks[1]))[0]
    
    loc = argmax(sky)
    counts = sky[*loc]

    snr_pos = _snr_over_thres_pos()
    n_peaks = len(snr_pos)
    check_snr = tuple(loc) in snr_pos

    if not check_snr:
        return loc, counts, n_peaks

    elif loc in sources_dataset['sources_pos']:
        print(f"Source pos ({loc[0]}, {loc[1]}) already recorded...")
        return loc, counts, n_peaks
    
    else:
        print(f"New source found at pos ({loc[0]}, {loc[1]})!")
        sources_dataset = record_source(sources_dataset, loc, counts,
                                        var[*loc], snr[*loc])
        return loc, counts, n_peaks


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


class LoadData:
    def __init__(self,
                 detector: np.array,
                 norm_skyrec: np.array,
                 norm_skyvar: np.array,
                 skysnr: np.array):
        self.detector = detector
        self.skyrec = norm_skyrec
        self.skyvar = norm_skyvar
        self.skysnr = skysnr


def IROS(n_iterations: int,
         data: LoadData,
         snr_threshold: np.array,
         cam: object,
         snr_distr: bool | tuple[bool, int] = False,
         snr_peaks: bool | tuple[bool, int] = False,
         ) -> c.Iterable:
    
    #TODO: implement skyrec correction wrt pos
    
    def iteration(
            detector: np.array,
            shadowgram: np.array,
        ) -> tuple[np.array, np.array, np.array, np.array]:
        
        det = detector - shadowgram
        det[det < 0] = 0
        new_sky, new_var = sky_reconstruction(det, cam)
        new_snr = sky_snr(new_sky, new_var)
        new_sky, new_var = skyrec_norm(new_sky, new_var, cam)

        return det, new_sky, new_var, new_snr
    
    # sky reconstruction
    detector = data.detector.copy()
    skyrec = data.skyrec.copy()
    skyvar = data.skyvar.copy()
    skysnr = data.skysnr.copy()

    # init log variables
    snr_peaks_list = []
    sources_dataset = {'sources_pos': [],
                       'sources_counts': [],
                       'sources_stds': [],
                       'sources_snrs': []}
    
    # IROS loop
    if isinstance(snr_distr, bool): _every1 = 1
    else: snr_distr, _every1 = snr_distr[0], snr_distr[1]
    if isinstance(snr_peaks, bool): _every2 = 1
    else: snr_peaks, _every2 = snr_peaks[0], snr_peaks[1]

    for i in range(n_iterations):

        if snr_distr and (i % _every1 == 0 or i == n_iterations - 1):
            show_snr_distr(skysnr, f": iter {i + 1}")
        
        pos, counts, n_peaks = select_source(skyrec, skyvar, skysnr,
                                             snr_threshold, sources_dataset)
        print(
            f"Outliers with SNR(σ) over {snr_threshold} at iter {i + 1}: {n_peaks}"
            )
        
        if n_peaks > 0:
            snr_peaks_list.append(n_peaks)
            if snr_peaks and (i % _every2 == 0 or i == n_iterations - 1):
                snr_over_threshold(snr_peaks_list, snr_threshold)
        
            shadowgram = get_shadowgram(pos, counts, cam)
            detector, skyrec, skyvar, skysnr = iteration(detector, shadowgram)
            yield skyrec, skyvar, skysnr, sources_dataset
        
        else:
            print(f"No sources detected with SNR over {snr_threshold}...")
            return skyrec, skyvar, skysnr, sources_dataset

    return skyrec, skyvar, skysnr, sources_dataset


def iros_skyrec(sky_image: np.array,
                sources_pos: list[tuple],
                sources_dataset: dict,
                data: LoadData,
                cam: object,
                ) -> tuple[np.array, np.array]:

    print(f"#### IROS Sky Reconstruction Run ####\n"
          f" - simulated sources: {len(sources_pos)}\n"
          f" - IROS rec. sources: {len(sources_dataset['sources_pos'])}" + (
              " c:" if len(sources_pos) == len(sources_dataset['sources_pos']) else " :c"
          ) + "\n")

    sky = np.zeros(cam.sky_shape)

    for idx, pos in enumerate(sources_dataset['sources_pos']):
        counts = sources_dataset['sources_counts'][idx]
        sky[*pos] = counts
        std = sources_dataset['sources_stds'][idx]
        snr = sources_dataset['sources_snrs'][idx]

        print(
            f"Source [{idx}] Reconstruction:\n"
            f" - simulated source transmitted counts: {sky_image[*pos]:.0f} +/- {np.sqrt(sky_image[*pos]):.0f}\n\n"
            f" - rec. source counts: {data.skyrec[*pos]:.0f} +/- {np.sqrt(data.skyvar[*pos]):.0f}\n"
            f" - rec. source SNR: {data.skysnr[*pos]:.0f}\n"
            f" - source rec. counts wrt simulated: {data.skyrec[*pos]*100/sky_image[*pos]:.2f}%\n\n"
            f" - IROS rec. source counts: {counts:.0f} +/- {std:.0f}\n"
            f" - IROS rec. source SNR: {snr:.0f}\n"
            f" - source rec. counts wrt simulated (with IROS): {counts*100/sky_image[*pos]:.2f}%\n"
        )
    
    print("#### End IROS Sky Reconstruction Run ####")
    
    residues = sky_image - sky

    image_plot([sky_image, data.skyrec],
               ["Simulated Sky", "Sky Reconstruction (1st)"],
               cbarlabel=["counts"]*2, cbarcmap=["viridis"]*2,
               simulated_sources=[sources_pos, sources_pos])
    
    image_plot([sky, residues],
               ["IROS Sky Reconstruction", "Residues: Sky - IROS"],
               cbarlabel=["counts"]*2, cbarcmap=["viridis"]*2,
               simulated_sources=[sources_dataset['sources_pos'], sources_dataset['sources_pos']])
    
    return sky, residues


def iros_log(sources_log: dict) -> None:

    print(f"#### IROS Sky Reconstruction Log ####\n")

    for idx, pos in enumerate(sources_log['sources_pos']):
        print(
            f"Source [{idx}] Log:\n"
            f" - pos: {pos}\n"
            f" - counts: {sources_log['sources_counts'][idx]:.0f}\n"
            f" - std: {sources_log['sources_stds'][idx]:.0f}\n"
            f" - SNR: {sources_log['sources_snrs'][idx]:.2f}\n"
        )

    print("#### End IROS Sky Reconstruction Log ####")

# end