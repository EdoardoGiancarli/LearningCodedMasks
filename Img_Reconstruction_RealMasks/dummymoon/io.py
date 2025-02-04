"""
Opening .fits files...
"""

from scipy.signal import correlate
from mbloodmoon import codedmask
from .display import image_plot


def import_mask(fits_path: str,
                show_info: bool = True,
                show_data: bool = True,
                ) -> object:

    wfm = codedmask(fits_path)

    if show_info:
        print("### Camera Parameters")
        for key, value in wfm.specs.items():
            print(f"{key}: {value}")
            
        print(f"\n### Shapes\n"
            f"Mask shape: {wfm.mask_shape}\n"
            f"Detector shape: {wfm.detector_shape}\n"
            f"Sky shape: {wfm.sky_shape}")

    if show_data:
        image_plot([wfm.mask, wfm.decoder],
                   ["Mask", "Decoder"],
                   cbarlabel=["aperture", "decoding value"],
                   cbarcmap=["viridis"]*2)
        
        psf = correlate(wfm.mask, wfm.decoder, mode="same")

        image_plot([wfm.bulk, psf],
                   ["Detector Bulk", "Mask PSF"],
                   cbarlabel=["detector response", "counts"],
                   cbarcmap=["inferno"]*2)

    return wfm


# end