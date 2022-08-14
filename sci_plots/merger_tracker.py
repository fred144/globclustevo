"""
Merger tracking of BSCs.
"""

import sys

sys.path.append("../")
import numpy as np
import os
import glob
from modules.macros import (
    filter_snapshots,
    characterisitc_mass,
    sci_notation,
    ram_fields,
)
from modules.profiles.profile_functions import projected_surf_densities
from modules.profiles.profile_models import modified_king_model
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
from matplotlib import colors
from modules.match_t_sims import find_matching_time, get_snapshots


strt = 113
end = 1000

step = 1
sn_list = np.array([399, 402, 408, 413, 419, 421, 432])
bsc_list = np.array([94, 80, 52, 44, 39, 42, 47])

pop2_dir = r"../particle_data/pop_2_data/fs07_refine"
gas_dir = "../gas_data/fs07_refine"
halo_dir = r"../halo_data/fs07_refine/fof_best"

pop2 = filter_snapshots(r"../particle_data/pop_2_data/fs07_refine", strt, end, step)
dens = filter_snapshots(os.path.join(gas_dir, "gas_density"), strt, end)
halos = filter_snapshots(os.path.relpath(halo_dir), strt, end)

gas_dens = get_snapshots(dens, get_list=sn_list)
plt_p2 = get_snapshots(pop2, get_list=sn_list)
plt_halo = get_snapshots(halos, get_list=sn_list)

slice_axis = "z"
width = (400, "pc")


star_lum_bin = 5000
pxl_size = width[0] / star_lum_bin
proj_r = width[0] / 2
axlims = (-200, 200)
star_cmap = "rainbow_r"
dens_cmap = "cubehelix"
temp_cmap = "gist_heat"
star_map = cm.get_cmap(star_cmap)

star_t_range = (340, 470)
# .5 Myr intervals
evenly_spaced_times = np.arange(star_t_range[0], star_t_range[1], 0.1)
cmap = star_map(np.linspace(0, 1, evenly_spaced_times.size))
dense_norm = (0.007, 0.35)
temp_norm = (6e2, 6e4)
lum_range = (2e34, 2e36)  # (8e32, 2e37)
efficiency = 0.70

times = []
images = []
center_x = []
center_y = []
redshifts = []

for i, (ds) in enumerate(plt_halo):
    cata = np.loadtxt(glob.glob(os.path.join(ds, "catalogue*.txt"))[0])

    # get pre processed data from pop2 data sets
    t_myr, redshift = np.loadtxt(plt_p2[i], max_rows=2)[0:2, 6]

    code_ctr = np.loadtxt(plt_p2[i], max_rows=5)[2:5, 6]
    bsc_number = bsc_list[i]
    bsc_center_mask = cata[:, 0] == bsc_number
    bsc_center = cata[bsc_center_mask][0]
    bsc_x = bsc_center[1]
    bsc_y = bsc_center[2]

    field_stars = np.loadtxt(os.path.join(plt_halo[i], "field_stars.txt"))
    bound_stars = np.loadtxt(os.path.join(plt_halo[i], "bound_stars.txt"))
    stars = np.vstack((field_stars, bound_stars))
    star_lums = stars[:, 2]

    star_ages = stars[:, 1]  # Myr
    star_bes = t_myr - star_ages
    pos_pc = stars[:, 3:6]

    lums, _, _ = np.histogram2d(
        pos_pc[:, 0],
        pos_pc[:, 1],
        bins=star_lum_bin,
        weights=star_lums,
        normed=False,
        range=[[-proj_r, proj_r], [-proj_r, proj_r]],
    )
    lums = lums.T / pxl_size

    times.append(t_myr)
    images.append(lums)
    center_x.append(bsc_x)
    center_y.append(bsc_y)
    redshifts.append(redshift)

#%%
with plt.style.context("dark_background"):
    with plt.rc_context(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.size": 14,
        }
    ):
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            # gridspec_kw={"width_ratios": [1, , 1, 1, 1]},
            figsize=(7, 7),
            dpi=400,
            # sharex=True,
            # sharey=True,
            facecolor=cm.Greys_r(0),
        )

        master_image = ax.imshow(
            images[0],
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            cmap="inferno",
            norm=LogNorm(lum_range[0], lum_range[1]),
            origin="lower",
        )
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")

        inset_coords = [
            (0.05, 0.70),
            (0.05, 0.5),
            (0.05, 0.3),
            (0.05, 0.1),
            (0.25, 0.1),
            (0.45, 0.1),
            (0.65, 0.1),
        ]

        for i, coords in enumerate(inset_coords):
            inset = ax.inset_axes([coords[0], coords[1], 0.20, 0.20])
            inset_image = inset.imshow(
                images[i],
                extent=[-proj_r, proj_r, -proj_r, proj_r],
                cmap="inferno",
                norm=LogNorm(lum_range[0], lum_range[1]),
                origin="lower",
            )

            inset.set_xlim(center_x[i] - 10, center_x[i] + 10)
            inset.set_ylim(center_y[i] - 10, center_y[i] + 10)
            inset.set_xticklabels([])
            inset.set_yticklabels([])
            inset.xaxis.set_ticks_position("none")
            inset.yaxis.set_ticks_position("none")
            inset.text(
                0.5,
                0.20,
                "$\mathrm{{t}} = {:.0f} \: \mathrm{{Myr}}$".format(times[i]),
                fontsize=12,
                ha="center",
                va="top",
                color="white",
                transform=inset.transAxes,
            )

            if i == 0:
                inset.text(
                    0.40,
                    0.35,
                    "$\mathrm{{z}} = {:.2f} $".format(redshifts[i]),
                    fontsize=12,
                    ha="center",
                    va="top",
                    color="white",
                    transform=inset.transAxes,
                )
                from mpl_toolkits.axes_grid1.inset_locator import mark_inset

                mark_inset(ax, inset, loc1=1, loc2=4, edgecolor="white", alpha=0.8)

        # scale bar
        scale = patches.Rectangle(
            xy=(center_x[0] - 10, center_y[0] - 15),
            width=20,
            height=1,
            linewidth=0,
            edgecolor="white",
            facecolor="white",
        )
        ax.text(
            center_x[0],
            center_y[0] - 20,
            r"$\mathrm{20 \: pc}$",
            ha="center",
            va="center",
            color="white",
        )
        ax.add_patch(scale)

        # color bar
        # [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.46, 0.805, 0.40, 0.02])
        cbar = fig.colorbar(inset_image, cax=cbar_ax, pad=0, orientation="horizontal")
        cbar_units = r"$\mathrm{\left(erg\:\:s^{-1}\:\AA^{-1}\:pc^{-2}\right)}$"
        cbar_label = (
            r"$\mathrm{Surface \: Brightness}$" r"$, \mathrm{\lambda = 1500 \: \AA \:}$"
        )
        cbar.set_label(
            label=cbar_units,
            labelpad=2,
            # fontproperties=leg_font,
            fontsize=12,
        )
        cbar.ax.xaxis.set_tick_params(pad=2, labelsize=10)
        cbar_ax.set_title(cbar_label, fontsize=14)

        # efficiency
        props = dict(
            boxstyle="round",
            facecolor="black",
            alpha=1,
            linewidth=1,
            edgecolor="white",
        )

        ax.text(
            0.075,
            0.96,
            r"$f_* = {:.2f}$".format(efficiency),
            fontsize=14,
            ha="left",
            va="top",
            color="white",
            bbox=props,
            transform=ax.transAxes,
        )

        plt.savefig(
            os.path.expanduser(
                (
                    "~/g_drive/Research/AstrophysicsSimulation/sci_plots/final/"
                    "gc_merger.png"
                )
            ),
            dpi=500,
            bbox_inches="tight",
            pad_inches=0.05,
        )
