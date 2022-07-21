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
f7_end = 1000
f3_strt = 154
f3_end = 1177
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
f7_sn_list = np.array([377, 750, 978, 1000])  # looks promising
f3_sn_list = np.array([414, 688, 943, 1177])


f7_plt_p2 = get_snapshots(f7_pop2, get_list=f7_sn_list)
f3_plt_p2 = get_snapshots(f3_pop2, get_list=f3_sn_list)

f7_plt_halo = get_snapshots(f7_halos, get_list=f7_sn_list)
f3_plt_halo = get_snapshots(f3_halos, get_list=f3_sn_list)
#%%

width = 400
rows = f7_sn_list.size
cols = 2
star_lum_bin = 2000
pxl_size = (width / star_lum_bin) ** 2
proj_r = width / 2
row_lims = [(-100, 100), (-150, 150), (-150, 150), (-200, 200)]
star_lum_range = (8e31, 5e36)

leg_font = font_manager.FontProperties(family="serif", math_fontfamily="cm", size=8)

with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "font.size": 10,
    }
):
    fig, ax = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(3.5, 1.75 * rows),
        dpi=400,
        # sharex="rows",
        # sharey="rows",
    )
    for i, (f7, f3) in enumerate(zip(f7_sn_list, f3_sn_list)):
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
        f7_star_bes = f7_t_myr - f7_star_ages
        f7_pos_pc = f7_stars[:, 3:6]
        f7_star_lums = f7_stars[:, 2]

        f3_field_stars = np.loadtxt(os.path.join(f3_plt_halo[i], "field_stars.txt"))
        f3_bound_stars = np.loadtxt(os.path.join(f3_plt_halo[i], "bound_stars.txt"))
        f3_stars = np.vstack((f3_field_stars, f3_bound_stars))

        f3_star_ages = f3_stars[:, 1]  # Myr
        f3_star_bes = f3_t_myr - f3_star_ages
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
            f7_xy_lums / pxl_size,
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            cmap="inferno",
            norm=LogNorm(star_lum_range[0], star_lum_range[1]),
            origin="lower",
        )

        f3_lums = ax[i, 1].imshow(
            f3_xy_lums / pxl_size,
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            cmap="inferno",
            norm=LogNorm(star_lum_range[0], star_lum_range[1]),
            origin="lower",
        )

        ax[i, 0].set_xlim(axlims)
        ax[i, 0].set_ylim(axlims)

        ax[i, 1].set_xlim(axlims)
        ax[i, 1].set_ylim(axlims)

        # clean up edges
        ax[i, 0].set_facecolor(cm.Greys_r(0))
        ax[i, 1].set_facecolor(cm.Greys_r(0))
        for t in range(cols):
            ax[i, t].spines["top"].set_color("white")
            ax[i, t].spines["bottom"].set_color("white")
            # ax[i, t].spines["right"].set_color("white")
            ax[i, t].spines["left"].set_color("white")
            ax[i, t].tick_params(colors="white")
            ax[i, t].set_yticklabels([])
            ax[i, t].set_xticklabels([])
            ax[i, t].xaxis.set_ticks_position("none")
            ax[i, t].yaxis.set_ticks_position("none")

            scale = patches.Rectangle(
                xy=(axlims[0] * 0.75, axlims[0] * 0.75),
                width=axlims[1] * 0.5,
                height=0.03 * axlims[1],
                linewidth=0,
                edgecolor="white",
                facecolor="white",
            )
            ax[i, t].text(
                axlims[0] * 0.50,
                axlims[0] * 0.87,
                r"$\mathrm{{{:.0f} \: pc}}$".format(axlims[1] * 0.5),
                ha="center",
                va="center",
                color="white",
                fontproperties=leg_font,
                fontsize=7,
            )
            ax[i, t].add_patch(scale)

        # add time stamps
        t_stamps = [(f7_t_myr, f7_redshift), (f3_t_myr, f3_redshift)]
        col_idxs = [0, 1]
        for s, stamp in enumerate(t_stamps):
            ax[i, col_idxs[s]].text(
                axlims[0] * 0.90,
                axlims[1] * 0.90,
                (
                    r"$\mathrm{{t = {:.1f} \: Myr}}$" "\n" r"$\mathrm{{z = {:.1f} }}$"
                ).format(stamp[0], stamp[1]),
                ha="left",
                va="top",
                color="white",
                fontproperties=leg_font,
                fontsize=7,
                bbox={
                    "boxstyle": "Square",
                    # have control over edge alpha and face alpha
                    "facecolor": colors.to_rgba("black")[:-1] + (0.5,),
                    "linewidth": 0.5,
                    "edgecolor": "white",
                    # "pad": 0.42,
                },
            )

    # add efficiency labels
    fig.text(
        0.32,
        0.89,
        r"$f_{*} = 0.70$",
        ha="center",
        fontproperties=leg_font,
    )
    fig.text(
        0.70,
        0.89,
        r"$f_{*} = 0.35$",
        ha="center",
        fontproperties=leg_font,
    )

    # add color bar
    cbar_ax = fig.add_axes([0.138, 0.111, 0.75, 0.012])
    cbar = fig.colorbar(f7_lums, cax=cbar_ax, pad=0, orientation="horizontal")

    cbar_label = (
        r"$\mathrm{\log_{10}\:Projected\:Luminosity}$"
        r"$, \mathrm{\lambda = 1500 \: \AA \:}$"
        # "\n"
        r"$\mathrm{\left(erg\:\:s^{-1}\:\AA^{-1}\:pc^{-2}\right)}$"
    )
    cbar.set_label(label=cbar_label, labelpad=2, fontproperties=leg_font, fontsize=7)
    cbar.ax.xaxis.set_tick_params(pad=2, labelsize=6)

    # cbar_ax.set_title(
    #     cbar_label, fontsize=11, fontproperties=leg_font
    # )

    fig.canvas.draw()
    x_labels = [i.get_text().replace("10^", "") for i in cbar_ax.get_xticklabels()]
    cbar_ax.set_xticklabels(x_labels)

plt.subplots_adjust(hspace=0, wspace=-0.05)
plt.savefig(
    os.path.expanduser(
        (
            "~/g_drive/Research/AstrophysicsSimulation/sci_plots/final/"
            "projected_lums.png"
        )
    ),
    dpi=800,
    bbox_inches="tight",
    pad_inches=0.05,
    format="png",
)
