import sys


sys.path.append("../")


sys.path.insert(1, "/home/fgarcia4/.local/lib/python3.8/site-packages")
import numpy as np
import os
import glob
from modules.luminosity.lum_functions import lum_look_up_table
from modules.match_t_sims import find_matching_time, get_snapshots
from modules.macros import filter_snapshots
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib import colors

import yt
from modules.macros import filter_snapshots, ram_fields, t_myr_from_z, code_age_to_myr

# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib as mpl


yt.enable_parallelism()
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
    }
)
plt.style.use("dark_background")
main_plt_width = (200, "pc")
star_bins = 1000
gas_alpha = 0.5
pxl_size = (main_plt_width[0] / star_bins) ** 2

lum_range = (3e33, 3e36)
gas_range = (0.008, 0.5)

lum_alpha = 1
gas_alpha = 0.5
#%%
main_dir = os.path.relpath("../../cosm_test_data/fs035_ms10/")

cell_fields, epf = ram_fields()

pre_ds = yt.load(
    os.path.join(main_dir, "output_00560/info_00560.txt"),
    fields=cell_fields,
    extra_particle_fields=epf,
)
post_ds = yt.load(
    os.path.join(main_dir, "output_00561/info_00561.txt"),
    fields=cell_fields,
    extra_particle_fields=epf,
)

pre_ad = pre_ds.all_data()
post_ad = post_ds.all_data()


t_myr = float(post_ds.current_time.in_units("Myr"))
redshift = float(post_ds.current_redshift)
current_hubble = post_ds.hubble_constant

sfc_kazu_radii = np.abs(
    post_ds.arr(post_ad["SFC", "particle_metallicity"], "code_length").to("pc")
)

sfc_rel_ages = code_age_to_myr(
    post_ad["SFC", "particle_birth_epoch"], current_hubble, unique_age=False
)

x_pop2 = post_ad["star", "particle_position_x"]
y_pop2 = post_ad["star", "particle_position_y"]
z_pop2 = post_ad["star", "particle_position_z"]

youngest = np.argmax(sfc_rel_ages)
sfc_code_ctr = np.array(
    [
        post_ad["SFC", "particle_position_x"][youngest],
        post_ad["SFC", "particle_position_y"][youngest],
        post_ad["SFC", "particle_position_z"][youngest],
    ]
)

# center based on star position distribution
x_center = np.mean(x_pop2.to("pc"))
y_center = np.mean(y_pop2.to("pc"))
z_center = np.mean(z_pop2.to("pc"))
sfc_x = post_ds.arr(sfc_code_ctr[0], "code_length").to("pc")
sfc_y = post_ds.arr(sfc_code_ctr[1], "code_length").to("pc")
sfc_z = post_ds.arr(sfc_code_ctr[2], "code_length").to("pc")


# find CoM of the cloud before sfc,
sphere = pre_ds.sphere(sfc_code_ctr, (10, "pc"))
# return CoM in code units
pre_center = sphere.quantities.center_of_mass(use_gas=True).to("pc")


#%% draw full frame

gas = yt.ProjectionPlot(
    pre_ds, "z", ("gas", "density"), width=main_plt_width, center=pre_center
)
gas_frb = gas.data_source.to_frb(main_plt_width, star_bins)
gas_array = np.array(gas_frb["gas", "density"])

#
sf_gas = yt.ProjectionPlot(
    post_ds, "z", ("gas", "density"), width=main_plt_width, center=sfc_code_ctr
)
sf_gas_frb = sf_gas.data_source.to_frb(main_plt_width, star_bins)
sf_gas_array = np.array(sf_gas_frb["gas", "density"])


pre_star_dir = "../halo_data/fs035_ms10/fof_best/info_00560"
field_stars = np.loadtxt(os.path.join(pre_star_dir, "field_stars.txt"))
bound_stars = np.loadtxt(os.path.join(pre_star_dir, "bound_stars.txt"))
stars = np.vstack((field_stars, bound_stars))
star_lums = stars[:, 2]
# x = stars[:, 3] + float(x_center)
# y = stars[:, 4] + float(y_center)

# x = x - float(pre_center[0])
# y = y - float(pre_center[1])
lums, _, _ = np.histogram2d(
    (pre_ad["star", "particle_position_x"] - pre_center[0]).to("pc"),
    (pre_ad["star", "particle_position_y"] - pre_center[1]).to("pc"),
    bins=star_bins,
    weights=star_lums,
    normed=False,
    range=[
        [-main_plt_width[0] / 2, main_plt_width[0] / 2],
        [-main_plt_width[0] / 2, main_plt_width[0] / 2],
    ],
)
lums = lums.T
#%%

fig, ax = plt.subplots(figsize=(4, 4), dpi=400)
lum = ax.imshow(
    lums / pxl_size,
    cmap="inferno",
    origin="lower",
    extent=[
        -main_plt_width[0] / 2,
        main_plt_width[0] / 2,
        -main_plt_width[0] / 2,
        main_plt_width[0] / 2,
    ],
    norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
    alpha=lum_alpha,
)

gas_im = ax.imshow(
    gaussian_filter(gas_array, sigma=5),
    cmap="cubehelix",
    interpolation="gaussian",
    origin="lower",
    extent=[
        -main_plt_width[0] / 2,
        main_plt_width[0] / 2,
        -main_plt_width[0] / 2,
        main_plt_width[0] / 2,
    ],
    norm=LogNorm(gas_range[0], gas_range[1]),
    alpha=gas_alpha,
)

zoom_ax = ax.inset_axes([1, 0.5, 0.5, 0.5])
lum = zoom_ax.imshow(
    lums / pxl_size,
    cmap="inferno",
    # interpolation="gaussian",
    origin="lower",
    extent=[
        -main_plt_width[0] / 2,
        main_plt_width[0] / 2,
        -main_plt_width[0] / 2,
        main_plt_width[0] / 2,
    ],
    norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
    alpha=lum_alpha,
)
ax.text(
    0.05,
    0.95,
    (r"$\mathrm{{t = {:.1f} \: Myr}}$" "\n" r"$\mathrm{{z = {:.1f} }}$").format(
        float(pre_ds.current_time.in_units("Myr")), pre_ds.current_redshift
    ),
    ha="left",
    va="top",
    color="white",
    transform=ax.transAxes,
)


gas_im = zoom_ax.imshow(
    gas_array,
    cmap="cubehelix",
    # interpolation="gaussian",
    origin="lower",
    extent=[
        -main_plt_width[0] / 2,
        main_plt_width[0] / 2,
        -main_plt_width[0] / 2,
        main_plt_width[0] / 2,
    ],
    norm=LogNorm(gas_range[0], gas_range[1]),
    alpha=gas_alpha,
)
zoom_ax.set(xlim=(-10, 10), ylim=(-10, 10))
mark_inset(ax, zoom_ax, loc1=2, loc2=2, edgecolor="white", alpha=0.5, lw=0.8, ls="--")

sf_ax = zoom_ax.inset_axes([0, -1, 1, 1])
sf_im = sf_ax.imshow(
    sf_gas_array,
    cmap="cubehelix",
    # interpolation="gaussian",
    origin="lower",
    extent=[
        -main_plt_width[0] / 2,
        main_plt_width[0] / 2,
        -main_plt_width[0] / 2,
        main_plt_width[0] / 2,
    ],
    norm=LogNorm(gas_range[0], gas_range[1]),
    alpha=gas_alpha,
)
sf_ax.scatter(x_pop2.to("pc") - sfc_x, y_pop2.to("pc") - sfc_y, color="tab:cyan", s=0.5)
sf_ax.set(xlim=(-2.6, 2.6), ylim=(-2.6, 2.6))
scale = patches.Rectangle(
    xy=(-2.6, -3.3),
    width=5.2,
    height=0.25,
    linewidth=0,
    edgecolor="white",
    facecolor="white",
)
zoom_ax.text(
    0,
    -4.5,
    r"$\mathrm{{5.2 \:pc}}$",
    ha="center",
    va="center",
    color="white",
    # fontproperties=leg_font,
)
zoom_ax.add_patch(scale)

mark_inset(
    zoom_ax, sf_ax, loc1=1, loc2=2, edgecolor="white", alpha=0.5, lw=0.8, ls="--"
)

cir = plt.Circle((0, 0), float(sfc_kazu_radii[youngest]), color="white", fill=False)
sf_ax.set_aspect("equal", adjustable="datalim")
sf_ax.add_patch(cir)
# sf_ax.plot(
#     np.linspace(0, sfc_kazu_radii[youngest], 10), np.zeros(10), ls=":", color="white"
# )
sf_ax.arrow(
    -float(sfc_kazu_radii[youngest]),
    0,
    float(sfc_kazu_radii[youngest]),
    0,
    lw=0.5,
    color="white",
    length_includes_head=True,
    head_width=0.1,
)

sf_ax.arrow(
    0,
    0,
    -float(sfc_kazu_radii[youngest]),
    0,
    lw=0.5,
    color="white",
    length_includes_head=True,
    head_width=0.1,
)
sf_ax.text(
    0.05,
    0.95,
    (r"$\mathrm{{t = {:.1f} \: Myr}}$").format(float(t_myr)),
    ha="left",
    va="top",
    color="white",
    transform=sf_ax.transAxes,
)

sf_ax.text(
    0.28,
    0.48,
    (r"$\mathrm{{{:.1f} \: pc}}$").format(float(sfc_kazu_radii[youngest])),
    ha="center",
    va="top",
    color="white",
    transform=sf_ax.transAxes,
)

sf_ax.text(
    0.28,
    0.55,
    (r"$\mathrm{{R_{{MC}} }}$"),
    ha="center",
    va="center",
    color="white",
    transform=sf_ax.transAxes,
)
gas_cbar_ax = ax.inset_axes([0.05, 0.08, 0.45, 0.04])
gas_cbar = fig.colorbar(gas_im, cax=gas_cbar_ax, pad=0, orientation="horizontal")
gas_cbar.ax.tick_params(labelsize=7)
gas_cbar_ax.set_title(label=r"$\mathrm{Gas\:Density\:(g \: cm^{-2})}$", fontsize=8)

zoom_ax.set_xticklabels([])
zoom_ax.set_yticklabels([])
sf_ax.set_xticklabels([])
sf_ax.set_yticklabels([])
ax.xaxis.set_ticks_position("none")
ax.yaxis.set_ticks_position("none")
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.savefig(
    "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/lowres/sf_recipe.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.0,
)
