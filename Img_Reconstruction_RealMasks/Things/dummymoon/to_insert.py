"""
Script-holder for things to be inserted and/or finalized...
"""

import numpy as np
from astropy.io import fits


#### in .io
def fits_info(path: str) -> None:
    with fits.open(path) as hdu:
        hdu.info()



def open_fits() -> fits.FITS_rec:   ### general
    """
    Temporary method for opening fits.
    """
    # TODO:
    #   - skyrec efficiency (to insert in MaskDataLoader as specs?)
    #   - iros output (more stand-alone method)

    # _validate_fits() in bloodmoon.io
    pass



def skyrec_efficiency_output() -> fits:   ### for skyrec efficiency
    pass

def load_skyrec_pos_efficiency_array():

    def _adapt_to_sky():
        pass

    pass



def iros_output() -> fits:   ### for IROS output
    pass

def load_iros_output():
    pass



#### in .iros
def compare_catalog() -> None:
    pass


# end