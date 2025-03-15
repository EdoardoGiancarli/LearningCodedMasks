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
hduA.header.update(wcs_out.to_header()) 
fits.writeto(root_path + "sky_SIMUL_COMPOSED.fits", array, wcs_out.to_header())

plt.figure(figsize=(7, 7))
plt.imshow(footprint, origin='lower')
plt.show()

hduA.close(); hduB.close()