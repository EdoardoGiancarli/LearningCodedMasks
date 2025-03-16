from astropy.io import fits
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject.mosaicking import reproject_and_coadd
import matplotlib.pyplot as plt

root_path = "/mnt/d/PhD_AASS/Coding/Images_fits/"


hduA = fits.open(root_path + "sky_SIMUL_CAM1A_TEST2.fits")
hduB = fits.open(root_path + "sky_SIMUL_CAM1B_TEST2.fits")

hdus = (hduA[1], hduB[1])
wcs_out, shape_out = find_optimal_celestial_wcs(hdus)
print(repr(wcs_out.to_header() ))

array, footprint = reproject_and_coadd(hdus, wcs_out, shape_out=shape_out, reproject_function=reproject_interp, combine_function="sum")
print(shape_out)


#Updating the header
hduA[1].header.update(wcs_out.to_header()) 
fits.writeto(root_path + "sky_SIMUL_COMPOSED.fits", array, wcs_out.to_header())

plt.figure(figsize=(7, 7))
plt.imshow(footprint, origin='lower')
plt.show()

hduA.close(); hduB.close()


from pathlib import Path

def camera_composition(
    skyA_path: str | Path,
    skyB_path: str | Path,
    save_to: str | Path,
) -> None:
    """
    Performs the composition of the WFM cameras, including the
    reprojection of the World Coordinates System for RA/Dec.

    Specifically, it:
        - Opens the skies FITS file
        - Finds the optimal WCS fit and sky shape for the composition
        - Reprojects and sums the two skies making the composition

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
    with fits.open(skyA_path) as hduA:
        sky_hduA = hduA[1]
    with fits.open(skyB_path) as hduB:
        sky_hduB = hduB[1]

    hdus = (sky_hduA, sky_hduB)
    wcs_out, shape_out = find_optimal_celestial_wcs(input_data=hdus)
    array, _ = reproject_and_coadd(
        input_data=hdus,
        output_projection=wcs_out,
        shape_out=shape_out,
        reproject_function=reproject_interp,
        combine_function="sum",
    )

    # updating the header of CAMERA A
    #hduA[1].header.update(wcs_out.to_header()) 
    fits.writeto(
        filename=save_to,
        data=array,
        header=wcs_out.to_header()
    )
