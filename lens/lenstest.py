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
#%%
xcenters = (xedges[:-1] + xedges[1:]) / 2
ycenters = (yedges[:-1] + yedges[1:]) / 2
# plt.imshow()
fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=300)
a = ax[0].imshow(
    lums.T,
    extent=[-gal_size, gal_size, -gal_size, gal_size],
    cmap="bone",
    origin="lower",
)
fig.colorbar(a, ax=ax[0], label="Flux Jansky")
a = ax[1].imshow(
    np.log10(
        lums.T,
        out=np.full_like(lums.T, -9),
        where=(lums.T != 0),
    ),
    extent=[-gal_size, gal_size, -gal_size, gal_size],
    cmap="bone",
    origin="lower",
    vmin=-9,
    vmax=-6,
)
# ax.set(xlim=(-1, 1), ylim=(-1, 1))
fig.colorbar(a, ax=ax[1], label="log Flux Jansky")
ax[0].set(xlabel='arcsec ({:.4f} " / pixel)'.format(arsec_per_pixel), ylabel="arcsec")
ax[1].set(xlabel='arcsec ({:.4f} " / pixel)'.format(arsec_per_pixel), ylabel="arcsec")


#%% use pyAutolense

# read in the fits file
image_path = path.join("./galaxy_image_lowsfe.fits")
# make a grid of the source plane, without lensing
# techincally should be 0.0001
scale = 0.05  # arsec per pixel in the image
galaxy_image = al.Array2D.from_fits(file_path=image_path, pixel_scales=scale)
source_plane_grid_2d = al.Grid2D.uniform(
    shape_native=galaxy_image.shape_native,
    pixel_scales=galaxy_image.pixel_scales,
    origin=(0.0, 0.0),
)
# construct detector viewing plane
# we can downsample the resolution to show realistic image detection.
detector_arcsec_per_pxl = 0.01
reco_image_fov = 2  # arcsec on one side
num_pix = int(reco_image_fov / detector_arcsec_per_pxl)
image_plane_grid_2d = al.Grid2D.uniform(
    shape_native=(num_pix, num_pix),
    pixel_scales=detector_arcsec_per_pxl,
)
# galaxy doing the lensing
# available profiles, including DM pyautolens.readthedocs.io/en/latest/api/mass.html
# =============================================================================
# lens = al.Galaxy(
#     redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=20)
# )

lens = al.Galaxy(
    redshift=0.566,
    mass=al.mp.NFW(
        centre=(0.0, 0.0), ell_comps=(-15, 15), kappa_s=3e12, scale_radius=6
    ),
)
# =============================================================================
# source properties (the image loaded to be lensed)
source = al.Galaxy(redshift=10.9)
# run the ray tracing
tracer = al.Tracer.from_galaxies(galaxies=[lens, source])
traced_image_plane_grid_2d = tracer.traced_grid_2d_list_from(grid=image_plane_grid_2d)[
    -1
]
lensed_image = griddata(
    points=source_plane_grid_2d, values=galaxy_image, xi=traced_image_plane_grid_2d
)
lensed_image = al.Array2D.no_mask(
    values=lensed_image,
    shape_native=image_plane_grid_2d.shape_native,
    pixel_scales=image_plane_grid_2d.pixel_scales,
)
plt.figure()
array_2d_plotter = aplt.Array2DPlotter(array=lensed_image)
array_2d_plotter.figure_2d()
#%%
fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=300)
transform = np.reshape(
    lensed_image, (-1, int(reco_image_fov / detector_arcsec_per_pxl))
)
im = ax[0].imshow(
    transform,
    extent=[
        -reco_image_fov / 2,
        reco_image_fov / 2,
        -reco_image_fov / 2,
        reco_image_fov / 2,
    ],
    cmap="inferno",
)
fig.colorbar(im, ax=ax[0], label="Flux Jansky")

im = ax[1].imshow(
    np.log10(
        transform,
        where=(transform != 0),
        out=np.full_like(transform, -15),
    ),
    extent=[
        -reco_image_fov / 2,
        reco_image_fov / 2,
        -reco_image_fov / 2,
        reco_image_fov / 2,
    ],
    cmap="inferno",
    vmin=-6,
    vmax=-10,
)
fig.colorbar(im, ax=ax[1], label="log Flux Jansky")
ax[0].set(
    xlabel='arcsec ({:.2f} " / pixel)'.format(detector_arcsec_per_pxl), ylabel="arcsec"
)
ax[1].set(
    xlabel='arcsec ({:.2f} " / pixel)'.format(detector_arcsec_per_pxl), ylabel="arcsec"
)
