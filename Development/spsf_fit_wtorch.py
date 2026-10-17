"""
Module for testing source PSF fit through torch, to be integrated into IROS souce shadowgram modelling step.
"""

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch.types import Tensor

from bloodmoon.mask import codedmask


# end