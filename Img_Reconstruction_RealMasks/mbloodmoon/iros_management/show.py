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

RCPARAMS = {
    'font.family': 'sans-serif',
    'font.weight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
}
mpl.rcParams.update(RCPARAMS)



def crop(
    image: np.array,
    pos: tuple[int],
    cropping: tuple[int],
) -> np.array:
    """
    Crops 2D array at given position and with given cropping.

    Args:
        image (np.array): 2D array to crop.
        pos (tuple[int]): Center position for cropping.
        cropping (tuple[int]): Size of the cropping (positive int).
    
    Returns:
        output (np.array): Cropped 2D array (shape twice the `cropping`).
    
    Raises:
        ValueError: If cropping is not a positive int tuple.
        IndexError: If cropping wrt indexes exceeds 2D array edges.
    
    Notes:
        - Negative indexes are allowed.
    
    TODO: insert croppingy, croppingx for further freedom
    """
    n, m = image.shape
    y, x = pos
    cy, cx = cropping
    if cy <= 0 or cx <= 0:
        raise ValueError("Cropping must be a tuple of positive integers.")
    if not (
        (((0 <= x - cx) and (x + cx < m)) or (((cx - x <= m) and (x + cx < 0)))) and
        (((0 <= y - cy) and (y + cy < n)) or (((cy - y <= n) and (y + cy < 0))))
    ):
        raise IndexError(f"Cropping {cropping} at pos {pos} exceeds array edges.")
    y1, y2 = y - cy, y + cy
    x1, x2 = x - cx, x + cx
    return image[y1 : y2, x1 : x2]


def snrdistr():
    pass


def plot_sequence():
    pass


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
    def valid_BG() -> bool:
        """Checks background shape."""
        if not (background.shape == camera.sky_shape):
            raise ValueError(f"Background must have same sky shape {camera.sky_shape}.")
        return True

    def make_source(
        shiftx: float,
        shifty: float,
        fluence: float,
        x: int,
        y: int,
    ) -> np.array:
        model = model_sky(camera, shiftx, shifty, fluence)
        canva = np.zeros_like(model)
        cy, cx = 40, 18
        canva[y - cy: y + cy, x - cx: x + cx] = crop(model, (y, x), (cy, cx))
        return canva

    if background is None:
        sky = np.zeros(camera.sky_shape)
    elif valid_BG():
        sky = background
    
    for shiftx, shifty, fluence, x, y in zip(
        data[cameraID]["shift_x"]["data"],
        data[cameraID]["shift_y"]["data"],
        data[cameraID]["fluence"]["data"],
        data[cameraID]["x"]["data"],
        data[cameraID]["y"]["data"],
    ):
        sky = sky + make_source(shiftx, shifty, fluence, x, y)
    
    sky[sky < 0] = 0
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
    sources_pos: list[tuple[int]] = None,
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
        sources_pos (list[tuple[int]], optional (default=None)):
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