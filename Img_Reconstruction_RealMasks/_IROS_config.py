"""
Support methods for the IROS pipeline.
"""

from pathlib import Path

import numpy as np

from mbloodmoon.io import simulation_files, simulation
from mbloodmoon.mask import decode, count, variance, snratio #, codedmask
# from mbloodmoon.images import upscale #, downscale
from mbloodmoon.utils import timer
import mbloodmoon.iros_management as iros

from temp_camera import codedmask
print("___using temp_camera_binning___")


def run_pipeline():
    """Runs the IROS pipeline."""
    raise NotImplementedError


# end