"""
Module for `model_shadowgram()` profiling (with pyinstrument).
"""

from timeit import timeit

from bloodmoon.coords import angle2shift
from bloodmoon.mask import codedmask
from bloodmoon.optim import model_shadowgram as bm_shadowgram

from fract_shift2 import model_shadowgram




if __name__ == '__main__':

    UPX, UPY = 5, 1
    mask_path = '/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data/Simulations/wfm_mask_NTHT_20250725.fits'
    #mask_path = '/mnt/d/PhD_AASS/Coding/Images_fits/wfm_mask_NTHT_20250725.fits'
    wfm = codedmask(mask_path, UPX, UPY)
    VIGNETTING = True
    PSFY = False

    thetax, thetay = 20.0, 20.0
    shiftx, shifty = map(
        lambda x: angle2shift(wfm, x),
        (thetax, thetay),
    )
    
    bm_detector = bm_shadowgram(
        camera=wfm,
        shift_x=shiftx,
        shift_y=shifty,
        vignetting=VIGNETTING,
        psfy=PSFY,
    )
    detector = model_shadowgram(
        camera=wfm,
        shift_x=shiftx,
        shift_y=shifty,
        vignetting=VIGNETTING,
        psfy=PSFY,
    )

#    REP = 5
#
#    print(f'Timing model_shadowgram...')
#    t1 = timeit('model_shadowgram(wfm, shiftx, shifty, VIGNETTING, PSFY)', globals=globals(), number=REP)
#    
#    print(f'Timing bm_shadowgram...')  
#    t2 = timeit('bm_shadowgram(wfm, shiftx, shifty, VIGNETTING, PSFY)', globals=globals(), number=REP)
#
#    print(
#        f'model_shadowgram: {t1 / REP}s',
#        f'bm_shadowgram: {t2 / REP}s',
#    )


# end