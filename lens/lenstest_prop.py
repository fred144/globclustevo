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
import os
from scipy.interpolate import griddata
import matplotlib.patches as patches

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "font.size": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
    }
)
plt.style.use("dark_background")

#%% use pyAutolense

# read in the fits file
image_path = path.join("./new2.fits")
# make a grid of the source plane, without lensing
# techincally should be 0.0001
# scale = 0.005  # arsec per pixel in the image
scale = 0.001  # arsec per pixel in the image
galaxy_image = al.Array2D.from_fits(file_path=image_path, pixel_scales=scale)
source_fov = galaxy_image.shape_native[0] * scale / 2.0

source_plane_grid_2d = al.Grid2D.uniform(
    shape_native=galaxy_image.shape_native,
    pixel_scales=galaxy_image.pixel_scales,
    origin=(5, 5),
)
# construct detector viewing plane
# we can downsample the resolution to show realistic image detection.
detector_arcsec_per_pxl = 0.005
reco_image_fov = 2  # arcsec on one side
num_pix = int(reco_image_fov / detector_arcsec_per_pxl)
image_plane_grid_2d = al.Grid2D.uniform(
    shape_native=(num_pix, num_pix),
    pixel_scales=detector_arcsec_per_pxl,
)
# galaxy doing the lensing
# available profiles, including DM pyautolens.readthedocs.io/en/latest/api/mass.html
# =============================================================================
lens = al.Galaxy(
    #   redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=7.0)
    redshift=0.5,
    # mass=al.mp.Isothermal(centre=(4.0, 4.0), ell_comps=(0.2, 0.0), einstein_radius=8.0),
    mass=al.mp.NFW(
        centre=(0.0, 0.0), ell_comps=(0.2, 0.0), scale_radius=8.0, kappa_s=1
    ),
)
# =============================================================================
# source properties (the image loaded to be lensed)
# source = al.Galaxy(redshift=10.25)
source = al.Galaxy(redshift=6.25)
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

#%%
fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), dpi=400)

transform = np.reshape(
    lensed_image, (-1, int(reco_image_fov / detector_arcsec_per_pxl))
)
low_scale = -2.5
high_scale = 2.5

im = ax[1].imshow(
    np.log10(
        transform,
        where=(transform != 0),
        out=np.full_like(transform, -15),
    )
    + 9,
    extent=[
        -reco_image_fov / 2,
        reco_image_fov / 2,
        -reco_image_fov / 2,
        reco_image_fov / 2,
    ],
    cmap="inferno",
    vmin=low_scale,
    vmax=high_scale,
)

ax[1].set_facecolor("black")


transform1 = np.reshape(galaxy_image, galaxy_image.shape_native)


im = ax[0].imshow(
    np.log10(
        transform1,
        where=(transform1 != 0),
        out=np.full_like(transform1, -15),
    )
    + 9,
    extent=[
        -source_fov / 2,
        source_fov / 2,
        -source_fov / 2,
        source_fov / 2,
    ],
    cmap="inferno",
    vmin=low_scale,
    vmax=high_scale,
)


ax[0].set_facecolor("black")

lum_cbar_ax = ax[0].inset_axes([0.08, 0.08, 0.5, 0.05])
lum_cbar = fig.colorbar(im, cax=lum_cbar_ax, pad=0, orientation="horizontal")
lum_cbar.ax.xaxis.set_tick_params(pad=3)
lum_cbar_ax.set_title(
    r"$\mathrm{\log_{10}\:Flux\:Pixel^{-1}\:(nJy)}$",
    fontsize=8,
    pad=5,
)
# add some scale bar
scalebar = patches.Rectangle(
    xy=(-0.2, 0.2),
    width=0.1,
    height=0.005,
    linewidth=0,
    edgecolor="white",
    facecolor="white",
)
ax[0].text(
    -0.15,
    0.22,
    r"$\mathrm{{0.1\:arcsec}}}$",
    ha="center",
    va="center",
    color="white",
    # fontproperties=leg_font,
)
ax[0].text(
    -0.15,
    0.18,
    r"$\mathrm{{{:.0f}\:pixels}}$".format(transform1.shape[0] * (0.1 / source_fov)),
    ha="center",
    va="center",
    color="white",
    # fontproperties=leg_font,
)
ax[0].add_patch(scalebar)

# lensed image scale bar
scalebar = patches.Rectangle(
    xy=(-0.75, 0.78),
    width=0.5,
    height=0.020,
    linewidth=0,
    edgecolor="white",
    facecolor="white",
)
ax[1].text(
    -0.5,
    0.85,
    r"$\mathrm{{0.5\:arcsec}}}$",
    ha="center",
    va="center",
    color="white",
)
ax[1].text(
    -0.5,
    0.72,
    r"$\mathrm{{{:.0f}\:pixels}}$".format(transform.shape[0] * (0.5 / reco_image_fov)),
    ha="center",
    va="center",
    color="white",
)
ax[1].add_patch(scalebar)
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[0].set_xticklabels([])
ax[0].set_yticklabels([])
fig.subplots_adjust(wspace=-0.01, hspace=0)
plt.savefig(
    os.path.expanduser("./lensed_and_source"),
    dpi=400,
    bbox_inches="tight",
    pad_inches=0.05,
)
