import numpy as np

from bloodmoon.mask import codedmask
from bloodmoon.coords import angle2shift
from bloodmoon.optim import model_shadowgram as bm_shadowgram

from maskpattern_shift import model_shadowgram


basepath = '/home/edoardo/Datadisk/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data/Simulations'
maskpath = f'{basepath}/wfm_mask_summer2021.fits'

UPX, UPY = 5, 1
wfm = codedmask(maskpath, UPX, UPY)
VIGNETTING = False
PSFY = False
shifty, shiftx = (
    angle2shift(wfm, 20),
    angle2shift(wfm, 20),
)

det = model_shadowgram(wfm, shiftx, shifty, VIGNETTING, PSFY)


# end