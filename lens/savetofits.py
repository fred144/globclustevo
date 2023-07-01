"""
using pyautolense. 
https://github.com/Jammy2211/PyAutoLens
Pythonized.
Can just input a fits image and model a lens on it using a variety of lensing cluster
mass distribution. 
ray traces from the source plane and interpolates to observer plane.
"""
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import autolens.plot as aplt
import autolens as al
from astropy.io import fits
from os import path
from scipy.interpolate import griddata


sys.path.append("../")
from modules.luminosity.lum_functions import lum_look_up_table


def lum_to_appmag_ab(lum, lum_dist, redshft):
    """
    Convert point luminosity to point absolute magnitude as detected
    need luminosity of individual star to be in eg/s/Angstrom
    and lumdistance in pc
    """
    abs_magab = -15.65 - 2.54 * np.log10(lum / 10**39)
    app_magab = abs_magab + 5 * np.log10(lum_dist / 100e9) + 50
    return app_magab


def ang_size(phys_size, lum_dist, redshift):
    """
    angsize in arcesconds
    """
    size_dist = lum_dist / (1 + redshift) ** 2
    return (phys_size / size_dist) * (2.06e5)


def zshifted_flux_jy(lum, lum_dis, wav_angs=1500):
    """
    need lum distance in parsecs
    """
    return 7.5e11 * (wav_angs / 1500) ** 2 * (lum / (4 * np.pi * (lum_dis * 3e18) ** 2))


# let's grab the star cluster data
particle_data = np.loadtxt(
    "../particle_data/pop_2_data/fs035_ms10/pos_00445_435_49_myr.txt"
)
d_lum = 117103e6  # pc
z = 10.9

x_viewed = ang_size(particle_data[:, 2], d_lum, z)
y_viewed = ang_size(particle_data[:, 3], d_lum, z)
flux = zshifted_flux_jy(
    lum_look_up_table(
        stellar_ages=particle_data[:, 1],
        table_link="../particle_data/luminosity_look_up_tables/l1500_inst_e.txt",
        column_idx=1,
        log=True,
    ),
    d_lum,
)


# construct what the galaxy would look like in the sky at a given redshift
gal_size = ang_size(200, d_lum, z)
bins = 1000
arsec_per_pixel = (2 * gal_size) / bins

print("200 pc is", gal_size, "arsec")
print(arsec_per_pixel, "arsec / pixel")

# axlim = 1  # arcsec
lums, xedges, yedges = np.histogram2d(
    x_viewed,
    y_viewed,
    bins=bins,
    weights=flux,
    normed=False,
    range=[[-gal_size, gal_size], [-gal_size, gal_size]],
)

hdu = fits.PrimaryHDU(lums.T)
hdu.writeto("./galaxy_image_lowsfe.fits", overwrite=True)
