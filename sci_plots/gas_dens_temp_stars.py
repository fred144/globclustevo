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


f7_strt = 200
f7_end = 1000
f3_strt = 377
f3_end = 1177
step = 1


f7_pop2_dir = r"../particle_data/pop_2_data/fs07_refine"
f3_pop2_dir = r"../particle_data/pop_2_data/fs035_ms10"

f7_gas_dir = "../gas_data/fs07_refine"
f3_gas_dir = "../gas_data/fs035_ms10"

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

f7_dens = filter_snapshots(os.path.join(f7_gas_dir, "gas_density"), f7_strt, f7_end)
f3_dens = filter_snapshots(os.path.join(f3_gas_dir, "gas_density"), f3_strt, f3_end)

f7_temp = filter_snapshots(
    os.path.join(f7_gas_dir, "weighted_temperature"), f7_strt, f7_end
)
f3_temp = filter_snapshots(
    os.path.join(f3_gas_dir, "weighted_temperature"), f3_strt, f3_end
)

f7_halos = filter_snapshots(os.path.relpath(f7_halo_dir), f7_strt, f7_end)
f3_halos = filter_snapshots(os.path.relpath(f3_halo_dir), f3_strt, f3_end)


# dictate which snapshots will be plotted
# f7_sn_list = np.array([378, 489, 746, 1000])  # looks promising
# f3_sn_list = np.array([406, 569, 777, 1177])

f7_sn_list = np.array([375, 500, 750, 1000])  # looks promising
f3_sn_list = np.array([402, 577, 777, 1177])

f7_gas_dens = get_snapshots(f7_dens, get_list=f7_sn_list)
f3_gas_dens = get_snapshots(f3_dens, get_list=f3_sn_list)

f7_gas_temp = get_snapshots(f7_temp, get_list=f7_sn_list)
f3_gas_temp = get_snapshots(f3_temp, get_list=f3_sn_list)

f7_plt_p2 = get_snapshots(f7_pop2, get_list=f7_sn_list)
f3_plt_p2 = get_snapshots(f3_pop2, get_list=f3_sn_list)

f7_plt_halo = get_snapshots(f7_halos, get_list=f7_sn_list)
f3_plt_halo = get_snapshots(f3_halos, get_list=f3_sn_list)
#%%
slice_axis = "z"
width = (400, "pc")
rows = f7_sn_list.size
cols = 5
star_lum_bin = 1500
pxl_size = width[0] / star_lum_bin
proj_r = width[0] / 2
axlims = (-200, 200)
star_cmap = "rainbow_r"
dens_cmap = "cubehelix"
temp_cmap = "gist_heat"
star_map = cm.get_cmap(star_cmap)

star_t_range = (340, 595)
# .5 Myr intervals
evenly_spaced_times = np.arange(star_t_range[0], star_t_range[1], 0.5)
cmap = star_map(np.linspace(0, 1, evenly_spaced_times.size))
dense_norm = (0.007, 0.35)
temp_norm = (6e2, 6e4)


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
        gridspec_kw={"width_ratios": [1, 1, 0.1, 1, 1]},
        figsize=(7, 1.75 * rows),
        dpi=400,
        sharex=True,
        sharey=True,
    )

    for i, (f7, f3) in enumerate(zip(f7_sn_list, f3_sn_list)):

        ax[i, 2].set_visible(False)

        # get gas data
        f7_dens = np.loadtxt(f7_gas_dens[i])
        f3_dens = np.loadtxt(f3_gas_dens[i])

        f7_temp = np.loadtxt(f7_gas_temp[i])
        f3_temp = np.loadtxt(f3_gas_temp[i])

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

        f3_field_stars = np.loadtxt(os.path.join(f3_plt_halo[i], "field_stars.txt"))
        f3_bound_stars = np.loadtxt(os.path.join(f3_plt_halo[i], "bound_stars.txt"))
        f3_stars = np.vstack((f3_field_stars, f3_bound_stars))

        f3_star_ages = f3_stars[:, 1]  # Myr
        f3_star_bes = f3_t_myr - f3_star_ages
        f3_pos_pc = f3_stars[:, 3:6]

        # show projection plot
        dens_proj_f7 = ax[i, 0].imshow(
            f7_dens,
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            cmap=dens_cmap,
            norm=LogNorm(dense_norm[0], dense_norm[1]),
            origin="lower",
        )
        dens_proj_f3 = ax[i, 3].imshow(
            f3_dens,
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            cmap=dens_cmap,
            norm=LogNorm(dense_norm[0], dense_norm[1]),
            origin="lower",
        )

        temp_proj_f7 = ax[i, 1].imshow(
            f7_temp,
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            cmap=temp_cmap,
            norm=LogNorm(temp_norm[0], temp_norm[1]),
            origin="lower",
        )
        temp_proj_f3 = ax[i, 4].imshow(
            f3_temp,
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            cmap=temp_cmap,
            norm=LogNorm(temp_norm[0], temp_norm[1]),
            origin="lower",
        )

        # annotate with stars using a loop
        bes = [f7_star_bes, f3_star_bes]
        pos = [f7_pos_pc, f3_pos_pc]
        for eff_idx, (b, p) in enumerate(zip(bes, pos)):
            rounded_times = np.round_(b, 1)
            unique_birth_times = np.unique(rounded_times)
            # dicatate the colums based on the efficiencies
            if eff_idx == 0:  #%70
                col_idxs = [0, 1]
            elif eff_idx == 1:  #%35
                col_idxs = [3, 4]
            else:
                pass

            for col in col_idxs:
                for s, unique_age in enumerate(unique_birth_times):
                    mask = np.array(rounded_times) == unique_age
                    filtered_x = p[:, 0][mask]
                    filtered_y = p[:, 1][mask]
                    idx_of_nearest_c = np.argmin(
                        np.abs(evenly_spaced_times - unique_age)
                    )
                    color = cmap[idx_of_nearest_c]
                    color = color.reshape(1, -1)

                    ax[i, col].scatter(
                        filtered_x,
                        filtered_y,
                        marker=".",
                        c=color,
                        s=0.01,
                        edgecolors=None,
                        alpha=0.08,
                    )
        # clean up edges and add scale bars
        for t in range(cols):
            ax[i, t].spines["top"].set_color("white")
            ax[i, t].spines["bottom"].set_color("white")
            ax[i, t].spines["right"].set_color("white")
            ax[i, t].spines["left"].set_color("white")
            ax[i, t].tick_params(colors="white")
            ax[i, t].set_yticklabels([])
            ax[i, t].set_xticklabels([])
            ax[i, t].xaxis.set_ticks_position("none")
            ax[i, t].yaxis.set_ticks_position("none")

        ax[i, 0].set_xlim(axlims)
        ax[i, 0].set_ylim(axlims)

        # add time stamps
        t_stamps = [(f7_t_myr, f7_redshift), (f3_t_myr, f3_redshift)]
        col_idxs = [0, 3]
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
            scales_cols = [1, 4]
            # add scales
            for sc in scales_cols:
                scale = patches.Rectangle(
                    xy=(axlims[0] * 0.75, axlims[0] * 0.75),
                    width=axlims[1] * 0.5,
                    height=0.03 * axlims[1],
                    linewidth=0,
                    edgecolor="white",
                    facecolor="white",
                )
                ax[i, sc].text(
                    axlims[0] * 0.50,
                    axlims[0] * 0.87,
                    r"$\mathrm{{{:.0f} \: pc}}$".format(axlims[1] * 0.5),
                    ha="center",
                    va="center",
                    color="white",
                    fontproperties=leg_font,
                    fontsize=6,
                )
                ax[i, sc].add_patch(scale)

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

    # add color bars

    dens_cbar_ax = fig.add_axes([0.135, 0.108, 0.25, 0.012])
    dens_cbar = fig.colorbar(
        dens_proj_f7, cax=dens_cbar_ax, pad=0, orientation="horizontal"
    )
    dens_cbar.ax.xaxis.set_tick_params(labelsize=6)
    dens_cbar.set_label(
        label=(
            r"$\mathrm{\log_{10}\:Gas\:Density\:}$"
            # "\n"
            r"$\mathrm{\left(g\:cm^{-2}\right)}$"
        ),
        fontproperties=leg_font,
        labelpad=1,
        fontsize=7,
    )
    # tick label mod
    fig.canvas.draw()
    x_labels = [i.get_text().replace("10^", "") for i in dens_cbar_ax.get_xticklabels()]
    dens_cbar_ax.set_xticklabels(x_labels)

    temp_cbar_ax = fig.add_axes([0.389, 0.108, 0.25, 0.012])
    temp_cbar = fig.colorbar(
        temp_proj_f3, cax=temp_cbar_ax, pad=0, orientation="horizontal"
    )
    temp_cbar.ax.xaxis.set_tick_params(labelsize=6)
    temp_cbar.set_label(
        label=r"$\mathrm{\log_{10}\:Weighted\:Temperature \:}$"
        # "\n"
        r"$\mathrm{(K)}$",
        fontproperties=leg_font,
        labelpad=1,
        fontsize=7,
    )
    # tick label mod
    fig.canvas.draw()
    x_labels = [i.get_text().replace("10^", "") for i in temp_cbar_ax.get_xticklabels()]
    temp_cbar_ax.set_xticklabels(x_labels)

    star_cbar_ax = fig.add_axes([0.642, 0.108, 0.25, 0.012])
    star_cb = mpl.colorbar.ColorbarBase(
        star_cbar_ax,
        norm=mpl.colors.Normalize(star_t_range[0], star_t_range[1]),
        # ticks = [340,405,470],
        orientation="horizontal",
        cmap=star_cmap,
    )
    star_cb.ax.tick_params(labelsize=6)
    star_cb.set_label(
        (
            r"$\mathrm{Pop\:II\:Birth\:Time\:}$"
            # "\n"
            r"$\mathrm{(Myr)}$"
        ),
        fontproperties=leg_font,
        labelpad=1.5,
        fontsize=7,
    )

plt.subplots_adjust(hspace=0, wspace=-0.1)
plt.savefig(
    os.path.expanduser(
        (
            "~/g_drive/Research/AstrophysicsSimulation/sci_plots/final/"
            "projected_gas.png"
        )
    ),
    dpi=800,
    bbox_inches="tight",
    pad_inches=0.05,
    format="png",
)
