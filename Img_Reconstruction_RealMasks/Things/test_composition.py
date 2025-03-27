from astropy.io import fits
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject.mosaicking import reproject_and_coadd
import matplotlib.pyplot as plt

root_path = "/mnt/d/PhD_AASS/Coding/Images_fits/"


#hduA = fits.open(root_path + "sky_SIMUL_CAM1A_TEST3_5000.fits")
#hduB = fits.open(root_path + "sky_SIMUL_CAM1B_TEST3_5000.fits")
#
#hdus = (hduA[1], hduB[1])
#wcs_out, shape_out = find_optimal_celestial_wcs(hdus)
#print(repr(wcs_out.to_header() ))
#
#array, footprint = reproject_and_coadd(hdus, wcs_out, shape_out=shape_out, reproject_function=reproject_interp, combine_function="sum")
#print(shape_out)
#
#
##Updating the header
##hduA[1].header.update(wcs_out.to_header()) 
#fits.writeto(root_path + "sky_SIMUL_COMPOSED_TEST3_5000.fits", array, wcs_out.to_header())
#
#plt.figure(figsize=(7, 7))
#plt.imshow(footprint, origin='lower')
#plt.show()
#
#hduA.close(); hduB.close()


from pathlib import Path

def camera_composition(
    skyA_path: str | Path,
    skyB_path: str | Path,
    save_to: str | Path,
) -> None:
    """
    Performs the composition of the WFM cameras skies and significances,
    including the reprojection of the World Coordinates System for RA/Dec.

    Specifically, it:
        - Opens the skies FITS file
        - Finds the optimal WCS fit and sky shape for the composition
        - Reprojects and sums the two skies making the composition
        - Reprojects and averages the two SNRs making the composition
        - Saves the composition FITS file

    Args:
        skyA_path (str, Path):
            File path for the camera A sky.
        skyB_path (str, Path):
            File path for the camera B sky.
        save_to (str, Path):
            File path or directory where the FITS image will be saved.

    Notes:
        - If the WCS fit keys are not present in the camera skies headers,
          a TypeError will be raised from `find_optimal_celestial_wcs()`:
        >>> TypeError: "WCS does not have celestial components."
    """
    with fits.open(skyA_path) as hduA, fits.open(skyB_path) as hduB:
        print("# Composing WFM skies...")
        skies, snrs = (hduA[1], hduB[1]), (hduA[2], hduB[2])
        wcs_out, shape_out = find_optimal_celestial_wcs(input_data=skies)

        sky_comp, _ = reproject_and_coadd(
            input_data=skies,
            output_projection=wcs_out,
            shape_out=shape_out,
            reproject_function=reproject_interp,
            combine_function="sum",
        )
        snr_comp, _ = reproject_and_coadd(
            input_data=snrs,
            output_projection=wcs_out,
            shape_out=shape_out,
            reproject_function=reproject_interp,
            combine_function="mean",
        )

        #hduA[1].header.update(wcs_out.to_header())  # updating the header of CAMERA A
        hdu_list = fits.HDUList([fits.PrimaryHDU()])

        for img, name in zip([sky_comp, snr_comp], ["sky", "snr"]):
            image_hdu = fits.ImageHDU(
                data=img,
                header=wcs_out.to_header(),
                name=name.upper(),
            )
            hdu_list.append(image_hdu)
        
        hdu_list.writeto(save_to, output_verify="fix+ignore")
        hdu_list.close()
        print("# WFM composition completed!")



hduA = root_path + "sky_SIMUL_CAM1A_TEST3_5000.fits"
hduB = root_path + "sky_SIMUL_CAM1B_TEST3_5000.fits"
comp_name = root_path + "sky_SIMUL_COMPOSED_TEST3_5000.fits"


camera_composition(
    skyA_path=hduA,
    skyB_path=hduB,
    save_to=comp_name,
)
