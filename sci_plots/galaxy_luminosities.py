"""
Projected  luminosity time series plots.
"""
import sys

sys.path.append("../")
import numpy as np
import os
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


f7_strt = 113
f7_end = 1110
f3_strt = 154
f3_end = 1316
step = 1


f7_pop2_dir = r"../particle_data/pop_2_data/fs07_refine"
f3_pop2_dir = r"../particle_data/pop_2_data/fs035_ms10"


f7_halo_dir = r"../halo_data/fs07_refine/fof_best"
f3_halo_dir = r"../halo_data/fs035_ms10/fof_best"

f7_pop2 = filter_snapshots(
    r"../particle_data/pop_2_data/fs07_refine", f7_strt, f7_end, step
)
# matched snapshots
f3_pop2 = filter_snapshots(
    r"../particle_data/pop_2_data/fs035_ms10", f3_strt, f3_end, step
)

# optional if you want to match them
# _, f3_matched_nums = find_matching_time(sequence=f7_pop2, look_up_sequence=f3_pop2)
# f3_pop2_matched = get_snapshots(f3_pop2, get_list=f3_matched_nums)


f7_halos = filter_snapshots(os.path.relpath(f7_halo_dir), f7_strt, f7_end)
f3_halos = filter_snapshots(os.path.relpath(f3_halo_dir), f3_strt, f3_end)


# dictate which snapshots will be plotted
# f7_sn_list = np.array([377, 502, 820, 1069])  # looks promising
# f3_sn_list = np.array([414, 492, 730, 1277])

f7_sn_list = np.array([377, 502])  # looks promising
f3_sn_list = np.array([414, 492])

f7_plt_p2 = get_snapshots(f7_pop2, get_list=f7_sn_list)
f3_plt_p2 = get_snapshots(f3_pop2, get_list=f3_sn_list)

f7_plt_halo = get_snapshots(f7_halos, get_list=f7_sn_list)
f3_plt_halo = get_snapshots(f3_halos, get_list=f3_sn_list)


width = 400  # pc
rows = f7_sn_list.size
cols = 2
star_lum_bin = 5000
pxl_size = (width / star_lum_bin) ** 2  # pc^2
proj_r = width / 2
row_lims = [(-60, 60), (-100, 100), (-100, 100), (-100, 100)]
pc2_to_cm2 = 9.52140614e36
star_lum_range = [
    (3e34 / pc2_to_cm2, 5e37 / pc2_to_cm2),
    (3e34 / pc2_to_cm2, 5e36 / pc2_to_cm2),
]
leg_font = font_manager.FontProperties(family="serif", math_fontfamily="cm", size=8)
#%%
with plt.style.context("dark_background"):
    with plt.rc_context(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "font.size": 7,
        }
    ):
        fig, ax = plt.subplots(
            nrows=rows,
            ncols=cols,
            figsize=(8, 8.2),
            dpi=400,
            # sharex="rows",
            # sharey="rows",
        )
        # ax[0, 0].set()
        # ax[1, 0].set(xticklabels=[], yticklabels=[])

        for i, (f7, f3) in enumerate(zip(f7_sn_list, f3_sn_list)):
            lum_range = star_lum_range[i]
            axlims = row_lims[i]
            # get pre processed data from pop2 data sets
            f7_t_myr, f7_redshift = np.loadtxt(f7_plt_p2[i], max_rows=2)[0:2, 6]
            f3_t_myr, f3_redshift = np.loadtxt(f3_plt_p2[i], max_rows=2)[0:2, 6]

            f7_code_ctr = np.loadtxt(f7_plt_p2[i], max_rows=5)[2:5, 6]
            f3_code_ctr = np.loadtxt(f3_plt_p2[i], max_rows=5)[2:5, 6]

            f7_field_stars = np.loadtxt(os.path.join(f7_plt_halo[i], "field_stars.txt"))
            f7_bound_stars = np.loadtxt(os.path.join(f7_plt_halo[i], "bound_stars.txt"))
            f7_stars = np.vstack((f7_field_stars, f7_bound_stars))

            f7_star_ages = f7_stars[:, 1]  # Myr
            # f7_star_bes = f7_t_myr - f7_star_ages
            f7_pos_pc = f7_stars[:, 3:6]
            f7_star_lums = f7_stars[:, 2]

            f3_field_stars = np.loadtxt(os.path.join(f3_plt_halo[i], "field_stars.txt"))
            f3_bound_stars = np.loadtxt(os.path.join(f3_plt_halo[i], "bound_stars.txt"))
            f3_stars = np.vstack((f3_field_stars, f3_bound_stars))

            f3_star_ages = f3_stars[:, 1]  # Myr
            # f3_star_bes = f3_t_myr - f3_star_ages
            f3_pos_pc = f3_stars[:, 3:6]
            f3_star_lums = f3_stars[:, 2]

            f7_xy_lums, _, _ = np.histogram2d(
                f7_pos_pc[:, 0],
                f7_pos_pc[:, 1],
                bins=star_lum_bin,
                weights=f7_star_lums,
                normed=False,
                range=[[-proj_r, proj_r], [-proj_r, proj_r]],
            )
            f7_xy_lums = f7_xy_lums.T
            f3_xy_lums, _, _ = np.histogram2d(
                f3_pos_pc[:, 0],
                f3_pos_pc[:, 1],
                bins=star_lum_bin,
                weights=f3_star_lums,
                normed=False,
                range=[[-proj_r, proj_r], [-proj_r, proj_r]],
            )
            f3_xy_lums = f3_xy_lums.T

            # show the projections
            f7_lums = ax[i, 0].imshow(
                f7_xy_lums / (pxl_size * pc2_to_cm2),
                extent=[-proj_r, proj_r, -proj_r, proj_r],
                cmap="inferno",
                norm=LogNorm(lum_range[0], lum_range[1]),
                origin="lower",
                interpolation="gaussian",
            )

            f3_lums = ax[i, 1].imshow(
                f3_xy_lums / (pxl_size * pc2_to_cm2),
                extent=[-proj_r, proj_r, -proj_r, proj_r],
                cmap="inferno",
                norm=LogNorm(lum_range[0], lum_range[1]),
                origin="lower",
                interpolation="gaussian",
            )

            ax[i, 0].set_xlim(axlims)
            ax[i, 0].set_ylim(axlims)
            ax[i, 0].set(xticklabels=[], yticklabels=[])
            ax[i, 0].xaxis.set_ticks_position("none")
            ax[i, 0].yaxis.set_ticks_position("none")
            ax[i, 1].set_xlim(axlims)
            ax[i, 1].set_ylim(axlims)
            ax[i, 1].set(xticklabels=[], yticklabels=[])
            ax[i, 1].xaxis.set_ticks_position("none")
            ax[i, 1].yaxis.set_ticks_position("none")
            # clean up edges
            ax[i, 0].set_facecolor(cm.Greys_r(0))
            ax[i, 1].set_facecolor(cm.Greys_r(0))
            # for t in range(cols):

            scale = patches.Rectangle(
                xy=(axlims[0] * 0.80, axlims[0] * 0.80),
                width=axlims[1] * 0.5,
                height=0.025 * axlims[1],
                linewidth=0,
                edgecolor="white",
                facecolor="white",
            )
            ax[i, 0].text(
                axlims[0] * 0.55,
                axlims[0] * 0.87,
                r"$\mathrm{{{:.0f} \: pc}}$".format(axlims[1] * 0.5),
                ha="center",
                va="center",
                color="white",
                fontproperties=leg_font,
            )
            ax[i, 0].add_patch(scale)

            # add time stamps
            t_stamps = [(f7_t_myr, f7_redshift), (f3_t_myr, f3_redshift)]
            col_idxs = [0, 1]
            for s, stamp in enumerate(t_stamps):
                ax[i, col_idxs[s]].text(
                    axlims[0] * 0.90,
                    axlims[1] * 0.90,
                    (
                        r"$\mathrm{{t = {:.1f} \: Myr}}$"
                        "\n"
                        r"$\mathrm{{z = {:.1f} }}$"
                    ).format(stamp[0], stamp[1]),
                    ha="left",
                    va="top",
                    color="white",
                    fontsize=9,
                    # bbox={
                    #     "boxstyle": "Square",
                    #     # have control over edge alpha and face alpha
                    #     "facecolor": colors.to_rgba("black")[:-1] + (0.5,),
                    #     "linewidth": 0.5,
                    #     "edgecolor": "white",
                    #     # "pad": 0.42,
                    # },
                )
            with plt.rc_context(
                {
                    "font.family": "serif",
                    "mathtext.fontset": "cm",
                    "xtick.labelsize": 7,
                    "ytick.labelsize": 7,
                    "font.size": 8,
                }
            ):
                cbar_ax = ax[i, 1].inset_axes([0.05, 0.08, 0.40, 0.04])
                cbar = fig.colorbar(
                    f7_lums, cax=cbar_ax, pad=0, orientation="horizontal"
                )
                cbar_label = (
                    # r"$\mathrm{\log\:\:Surface\:Brightness}, "
                    r"$\mathrm{\log_{10} \: \lambda = 1500 \: \AA \:}$"
                    "\n"
                    r"$\mathrm{\left(erg \:\: s^{-1} \: \AA^{-1} \: cm^{-2} \right)} $"
                )
                cbar.set_label(
                    label=cbar_label,
                    # fontsize=10,
                    size=8
                    # fontproperties=leg_font,
                )
                cbar.ax.xaxis.set_ticks_position("bottom")
                cbar.ax.xaxis.set_label_position("top")
                cbar_ax.tick_params(axis="both", direction="in", which="both")
                cbar.ax.xaxis.set_tick_params(pad=2)
                fig.canvas.draw()
                x_labels = [
                    i.get_text().replace("10^", "") for i in cbar_ax.get_xticklabels()
                ]
                cbar_ax.set_xticklabels(x_labels)

        # # add efficiency labels
        # fig.text(
        #     0.32,
        #     0.89,
        #     r"$f_{*} = 0.70$",
        #     ha="center",
        #     fontproperties=leg_font,
        # )
        # fig.text(
        #     0.70,
        #     0.89,
        #     r"$f_{*} = 0.35$",
        #     ha="center",
        #     fontproperties=leg_font,
        # )

        # add color bar
        # cbar_ax = fig.add_axes([0.138, 0.111, 0.75, 0.012])
        # cbar = fig.colorbar(f7_lums, cax=cbar_ax, pad=0, orientation="horizontal")

        # cbar_label = (
        #     r"$\mathrm{\log_{10}\:Surface\:Brightness}$"
        #     r"$, \mathrm{\lambda = 1500 \: \AA \:}$"
        #     # "\n"
        #     r"$\mathrm{\left(erg\:\:s^{-1}\:\AA^{-1}\:pc^{-2}\right)}$"
        # )
        # cbar.set_label(label=cbar_label, labelpad=2, fontproperties=leg_font, fontsize=7)
        # cbar.ax.xaxis.set_tick_params(pad=2, labelsize=6)

        # cbar_ax.set_title(
        #     cbar_label, fontsize=11, fontproperties=leg_font
        # )

        # fig.canvas.draw()
        # x_labels = [i.get_text().replace("10^", "") for i in cbar_ax.get_xticklabels()]
        # cbar_ax.set_xticklabels(x_labels)

        ax[0, 0].text(
            0.95,
            0.95,
            "$\mathrm{high-SFE}$",
            ha="right",
            va="top",
            color="white",
            transform=ax[0, 0].transAxes,
            fontsize=9,
            # fontproperties=leg_font,
            # bbox=props,
        )

        ax[0, 1].text(
            0.95,
            0.95,
            "$\mathrm{low-SFE}$",
            ha="right",
            va="top",
            color="white",
            transform=ax[0, 1].transAxes,
            fontsize=9,
            # fontproperties=leg_font,
            # bbox=props,
        )

ax[0, 0].spines["left"].set_visible(False)
ax[0, 0].spines["top"].set_visible(False)
ax[0, 1].spines["right"].set_visible(False)
ax[0, 1].spines["top"].set_visible(False)


ax[1, 0].spines["left"].set_visible(False)
ax[1, 0].spines["bottom"].set_visible(False)

ax[1, 1].spines["right"].set_visible(False)
ax[1, 1].spines["bottom"].set_visible(False)

plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig(
    os.path.expanduser(
        (
            "~/g_drive/Research/AstrophysicsSimulation/sci_plots/final/"
            "projected_lums.png"
        )
    ),
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.00,
)
