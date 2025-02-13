"""
Iterative Removal of Sources...
"""

import collections.abc as c

import numpy as np

from .skyrec import sky_encoding, sky_reconstruction, skyrec_norm
from .skyrec import sky_snr, show_snr_distr
from .display import image_plot, sequence_plot


def argmax(x: np.array) -> tuple[int, int]:
    row, col = np.unravel_index(np.argmax(x), x.shape)
    return int(row), int(col)


def get_shadowgram(pos: tuple[int, int],
                   counts: int | float,
                   cam: object,
                   ) -> np.array:
    
    def _source_calibration() -> float:
        f = 1e4
        shadow = np.zeros(cam.sky_shape)
        shadow[*pos] = f
        d = sky_encoding(shadow, cam)
        rec, var = sky_reconstruction(d, cam)
        rec, _ = skyrec_norm(rec, var, cam)
        return rec[*pos]/f
    
    def _source_calibration2() -> float:
        f = 1e4
        n, m = cam.sky_shape
        shadow_on = np.zeros((n, m))
        shadow_off = shadow_on.copy()
        shadow_on[(n - 1)//2, (m - 1)//2] = f
        shadow_off[*pos] = f
        d_on = sky_encoding(shadow_on, cam).sum()
        d_off = sky_encoding(shadow_off, cam).sum()
        return d_off/d_on
    
    s = np.zeros(cam.sky_shape)
    pos_calibr = _source_calibration2()
    s[*pos] = counts

    return sky_encoding(s, cam)/pos_calibr


def select_source(skyrec: np.array,
                  skyvar: np.array,
                  skysnr: np.array,
                  threshold: int | float,
                  sources_dataset: dict,
                  ) -> tuple[tuple[int, int], int]:
    
    def _n_snr_peaks():
        return len(np.argwhere(skysnr > threshold).T[0])

    def record_source(pos: np.array,
                      counts: int | float,
                      var: int | float,
                      snr: int | float,
                      ) -> dict:
        for key, value in zip(sources_dataset.keys(), [pos, counts, var, snr]):
            sources_dataset[key].append(value)
        return sources_dataset
    
    loc = argmax(skyrec)
    counts = skyrec[*loc]
    snr = skysnr[*loc]
    check_snr = snr > threshold
    n = _n_snr_peaks()

    if not check_snr:
        return loc, counts, n

    elif loc in sources_dataset['sources_pos']:
        print(f"Source pos ({loc[0]}, {loc[1]}) already recorded...")
        return loc, counts, n
    
    else:
        print(f"New source found at pos ({loc[0]}, {loc[1]})!")
        sources_dataset = record_source(loc, counts, skyvar[*loc], snr)
        return loc, counts, n


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


def IROS(data: np.array,
         snr_threshold: int | float,
         cam: object,
         max_iterations: int = 20,
         snr_distr: bool | tuple[bool, int] = False,
         snr_peaks: bool | tuple[bool, int] = False,
         ) -> c.Iterable:
    """
    Dummy IROS pipeline to test the algorithm.
    """

    def _check_bool(v: bool | tuple) -> tuple[bool, int]:
        if isinstance(v, bool):
            return v, 5
        return v
    
    def show_snr_peaks(y: list) -> None:
        sequence_plot([y], [f"SkyRec Peaks over SNR $=$ {snr_threshold}"],
                      [list(range(len(y)))], ["iteration"],
                      ["SNR peaks"], style=["scatter"])

    # initialize IROS and log variables
    detector = data.copy()
    snr_distr, _every1 = _check_bool(snr_distr)
    snr_peaks, _every2 = _check_bool(snr_peaks)
    snr_peaks_list = []
    sources_dataset = {'sources_pos': [],
                       'sources_counts': [],
                       'sources_stds': [],
                       'sources_snrs': []}

    # perform IROS
    for i in range(max_iterations):

        # sky reconstruction
        skyrec, skyvar = sky_reconstruction(detector, cam)
        skysnr = sky_snr(skyrec, skyvar)
        skyrec, skyvar = skyrec_norm(skyrec, skyvar, cam)

        if (snr_distr and i % _every1 == 0) or (i == max_iterations - 1):
            show_snr_distr(skysnr, f": iter {i + 1}")

        # check SNR map and select source 
        source_pos, source_fluence, n_snr_peaks = select_source(skyrec, skyvar, skysnr,
                                                                snr_threshold, sources_dataset)
        
        # source removal
        if n_snr_peaks > 0:
            
            snr_peaks_list.append(n_snr_peaks)
            if (snr_peaks and i % _every2 == 0) or (i == max_iterations - 1):
                show_snr_peaks(snr_peaks_list)

            shadowgram = get_shadowgram(source_pos, source_fluence, cam)
            detector = detector - shadowgram
            detector[detector < 0] = 0           # check for negative counts

            yield skyrec, skyvar, skysnr, sources_dataset
        
        else:
            print(f"No sources detected with SNR over {snr_threshold}...")
            show_snr_distr(skysnr, f": iter {i + 1}")
            show_snr_peaks(snr_peaks_list)
            return skyrec, skyvar, skysnr, sources_dataset


def iros_skyrec(sky_image: np.array,
                sources_pos: list[tuple],
                sources_dataset: dict,
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
            f"# Source [{idx}] Reconstruction:\n"
            f"  - simulated source transmitted counts: {sky_image[*pos]:.0f} +/- {np.sqrt(sky_image[*pos]):.0f}\n"
            f"  - IROS rec. source counts: {counts:.0f} +/- {std:.0f}\n"
            f"  - IROS rec. source SNR: {snr:.0f}\n"
            f"  - source rec. counts wrt simulated: {counts*100/sky_image[*pos]:.2f}%\n"
        )
    
    residues = sky_image - sky

    image_plot([sky_image, sky, residues],
               ["Simulated Sky", "IROS Sky Reconstruction", "Residues: Sky - IROS"],
               cbarlabel=["counts"]*3, cbarcmap=["viridis"]*3,
               simulated_sources=[sources_pos] + [sources_dataset['sources_pos']]*2)
    
    return sky, residues


def iros_log(sources_log: dict) -> None:

    print("### IROS Sky Reconstruction Log ###")

    for idx, pos in enumerate(sources_log['sources_pos']):
        counts = sources_log['sources_counts'][idx]
        std = sources_log['sources_stds'][idx]
        print(
            f"# Source [{idx}] Log:\n"
            f"  - pos: {pos}\n"
            f"  - counts: {counts:.0f} +/- {std:.0f}\n"
            f"  - SNR: {sources_log['sources_snrs'][idx]:.2f}\n"
        )


# end