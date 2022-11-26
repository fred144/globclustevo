import sys

sys.path.append("..")
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
from modules.macros import filter_snapshots, sci_notation
import matplotlib.lines as mlines
from modules.match_t_sims import find_matching_time, get_snapshots
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from modules.profiles.profile_functions import surface_density
from modules.profiles.profile_models import modified_king_model
from modules.macros import filter_snapshots, characterisitc_mass
from matplotlib import colors
import matplotlib.patches as patches


def nav_fre_whi(r, rho_0, r_scale):
    rho = rho_0 / ((r / r_scale) * (1 + (r / r_scale)) ** 2)
    return rho


fs070 = filter_snapshots("../particle_data/pop_2_data/fs07_refine", 115, 1195, 1)
fs035 = filter_snapshots("../particle_data/pop_2_data/fs035_ms10", 154, 1364, 1)
# find matching fs = 0.35 snapshots in terms of time to fs = 0.70
# smaller goes fist
# in general, use the simulation with more snapshots as a lookup table and match
# the ones with less
_, f7_matched_nums = find_matching_time(sequence=fs035, look_up_sequence=fs070)

fs070_dat_dir = r"../dm/fs07_refine/dm_hop"
fs035_dat_dir = r"../dm/fs035_ms10/dm_hop"

samp_strt = 1
samp_end = 200
step = 10

fs035_matched = filter_snapshots(fs035_dat_dir, 154, 1364, 1)[samp_strt:samp_end:step]
fs070_matched = get_snapshots(
    snapshot_file_list=filter_snapshots(fs070_dat_dir, 115, 1195, 1),
    get_list=f7_matched_nums,
    verbose=False,
)[samp_strt:samp_end:step]
# 100
bins = 500
plt_rad = 2000  # pc
pxl_size = (plt_rad * 2 / bins) ** 2
profile_plot_bins = 25

f3_series = np.loadtxt("../dm/fs035_ms10/dm_hop/fs035_dm_halo_evo.txt")
f7_series = np.loadtxt("../dm/fs07_refine/dm_hop/fs070_dm_halo_evo.txt")
eff = [0.35, 0.70]


for i, (f3, f7) in enumerate(zip(fs035_matched, fs070_matched)):

    sims = [f3, f7]
    series = [f3_series, f7_series]

    with plt.style.context("dark_background"):
        with plt.rc_context(
            {
                "font.family": "serif",
                "mathtext.fontset": "cm",
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "font.size": 12,
            }
        ):
            fig, ax = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(12, 6.0),
                dpi=400,
                sharex=True,
                sharey=True,
                facecolor=cm.viridis(0),
            )
            plt.subplots_adjust(hspace=0, wspace=-0.1)

    for plt_idx, s in enumerate(sims):
        try:
            out_num = int(s.split("/")[-1].split("_")[-1])
        except:
            out_num = int(s.split("\\")[-1].split("_")[-1])

        if int(out_num) not in f3_series[:, 0] and plt_idx == 0:
            break
        if int(out_num) not in f7_series[:, 0] and plt_idx == 1:
            break

        dm_part_data = np.loadtxt(os.path.join(s, "dm_data.txt"))
        time = float(series[plt_idx][:, 1][series[plt_idx][:, 0] == out_num])
        vir_rad = float(series[plt_idx][:, 4][series[plt_idx][:, 0] == out_num])

        mass = dm_part_data[:, 0]
        x = dm_part_data[:, 1]
        y = dm_part_data[:, 2]
        z = dm_part_data[:, 3]

        recentered = np.average(np.vstack((x, y, z)).T, axis=0)

        # x = dm_part_data[:, 1] - recentered[0]
        # y = dm_part_data[:, 2] - recentered[1]
        # z = dm_part_data[:, 3] - recentered[2]

        xy_mass, _, _ = np.histogram2d(
            x,
            y,
            bins=bins,
            weights=mass,
            normed=False,
            range=[[-plt_rad, plt_rad], [-plt_rad, plt_rad]],
        )
        xy_mass = xy_mass.T

        with plt.style.context("dark_background"):
            with plt.rc_context(
                {
                    "font.family": "serif",
                    "mathtext.fontset": "cm",
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "font.size": 12,
                }
            ):

                xy = ax[plt_idx].imshow(
                    xy_mass,
                    cmap="viridis",
                    # interpolation="gaussian",
                    origin="lower",
                    extent=[-plt_rad, plt_rad, -plt_rad, plt_rad],
                    norm=LogNorm(),
                )
                ax[plt_idx].scatter([0], [0], marker="+", alpha=0.2, color="red", s=70)
                # ax[plt_idx].scatter(
                #     recentered[0],
                #     recentered[1],
                #     marker="x",
                #     alpha=0.8,
                #     color="orange",
                #     s=40,
                # )
                ax[plt_idx].set_facecolor(cm.viridis(0))
                ax[plt_idx].set_xticklabels([])
                ax[plt_idx].set_yticklabels([])
                ax[plt_idx].xaxis.set_ticks_position("none")
                ax[plt_idx].yaxis.set_ticks_position("none")

                ax[plt_idx].set_xlim(left=-2200, right=1100)
                ax[plt_idx].set_ylim(bottom=-2200, top=1100)

                # add scale
                master_scale = patches.Rectangle(
                    xy=(0, -500),
                    width=500,
                    height=10,
                    linewidth=0,
                    edgecolor="white",
                    facecolor="white",
                    # transform= ax[plt_idx].transAxes,
                )
                ax[plt_idx].text(
                    250,
                    -600,
                    r"$\mathrm{ 500 pc}$",
                    ha="center",
                    va="center",
                    color="white",
                    # rotation=270,
                    # fontproperties=leg_font,
                    # transform= ax[plt_idx].transAxes,
                )
                ax[plt_idx].add_patch(master_scale)

                # surface density profile
                with plt.rc_context(
                    {
                        "xtick.labelsize": 7,
                        "ytick.labelsize": 7,
                        "font.size": 10,
                    }
                ):
                    prof = ax[plt_idx].inset_axes([0.13, 0.13, 0.40, 0.40])

                #  dm efficiency
                mask = series[plt_idx][:, 1] <= time
                halo_eff = (series[plt_idx][:, 5] / series[plt_idx][:, 3])[mask][-1]

                r, rho, err = surface_density(
                    x_coord=x,
                    y_coord=y,
                    z_coord=z,
                    masses=mass,
                    radius=vir_rad,
                    num_bins=profile_plot_bins,
                )

                prof.errorbar(
                    r,
                    rho,
                    yerr=err,
                    fmt="o",
                    capsize=3,
                    capthick=1,
                    elinewidth=1,
                    ms=3,
                    alpha=1,
                    c="w",
                )
                concentration_param = 4
                rho_not = 1e-2
                r_s = vir_rad / concentration_param
                rho_not = 0.20 * np.sum(mass) / r_s**3
                nfw = nav_fre_whi(
                    np.geomspace(r.min(), r.max() * 1.5, 100),
                    rho_not,
                    vir_rad / concentration_param,
                )
                prof.plot(
                    np.geomspace(r.min(), r.max() * 1.5, 100),
                    nfw,
                    label=r"$ R_{{\mathrm{{vir}}}} \:/ \:r_{{\mathrm{{s}}}} = {:.1f} $"
                    "\n"
                    r"$\rho_0 = {:.3f}$".format(concentration_param, rho_not),
                )

                prof.set(
                    xscale="log",
                    yscale="log",
                    xlabel=r"$ \mathrm{R \:(pc)}$",
                    ylabel=r"$\mathrm{\rho \:\:(M_{\odot}\:pc^{-3})}$",
                    # ylim=(0.2, 300),
                    # xlim=(6, 3000),
                )

                prof_leg = prof.legend(fontsize=10, loc="lower left")
                # prof_leg.get_frame().set_edgecolor("grey")
                prof_leg.get_frame().set_alpha(0)

                ax[plt_idx].text(
                    0.05,
                    0.95,
                    "$f_* = {:.2f}$"
                    "\n"
                    "$\mathrm{{t = {:.2f} \: Myr}}$"
                    "\n"
                    r"$M_{{\mathrm{{vir}}}} = {} \: \mathrm{{M}}_{{\odot}}$"
                    "\n"
                    r"$R_{{\mathrm{{vir}}}} = {:.2f} \: \mathrm{{pc}}$"
                    "\n"
                    r"$\mathrm{{M_* / M_\mathrm{{DM}}}} = {:.5f} $".format(
                        eff[plt_idx],
                        time,
                        sci_notation(2, np.sum(mass)),
                        vir_rad,
                        halo_eff,
                    ),
                    ha="left",
                    va="top",
                    color="white",
                    transform=ax[plt_idx].transAxes,
                    bbox={
                        "boxstyle": "Square",
                        "facecolor": colors.to_rgba("black")[:-1] + (0.2,),
                        "linewidth": 0.8,
                        "edgecolor": "white",
                        "pad": 0.40,
                    },
                )
                prof.set_facecolor("black")
                prof.patch.set_alpha(0.2)

                if plt_idx == 0:
                    # color bar
                    cbar_ax = fig.add_axes([0.35, 0.15, 0.15, 0.02])
                    cbar = fig.colorbar(
                        xy, cax=cbar_ax, pad=0, orientation="horizontal"
                    )
                    cbar_ax.xaxis.set_tick_params(pad=2, labelsize=8)
                    cbar_ax.xaxis.set_ticks_position("top")
                    # fig.canvas.draw()
                    # x_labels = [
                    #     i.get_text().replace("10^", "") for i in cbar_ ax[plt_idx].get_xticklabels()
                    # ]
                    # cbar_ ax[plt_idx].set_xticklabels(x_labels)

                    cbar_ax.set_title(
                        "$\mathrm{DM \: Mass \: (M_{\odot})}$",
                        # labelpad=2,
                        # fontproperties=leg_font,
                        fontsize=14,
                    )
    # plt.show()
