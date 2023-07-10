"""
Master plot for bound unbound clusters with mass and luminosity,
Using data only from the FOF. Not filtered by fittability by running through the 
profiler.

"""

import sys

sys.path.append("../")
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from modules.macros import filter_snapshots, common_filter_snapshots
import os
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks, find_peaks_cwt, savgol_filter
from matplotlib.ticker import MaxNLocator
from modules.macros import t_myr_from_z, z_from_t_myr

# 70% efficiency run
fs70_ds = np.loadtxt("./../fof_time_series/fs07_refine_fof_best_113_1196.txt")[::1, :]
fail_mask = fs70_ds[:, 3] > 10

# all results are fit filtered
f7_t_sim_myr = fs70_ds[:, 0][fail_mask]
f7_redshift = fs70_ds[:, 1][fail_mask]

f7_total_mass = fs70_ds[:, 2][fail_mask] + fs70_ds[:, 3][fail_mask]
f7_mass_in_gc = fs70_ds[:, 2][fail_mask]

f7_total_lum = fs70_ds[:, 4][fail_mask] + fs70_ds[:, 5][fail_mask]
f7_lum_in_gc = fs70_ds[:, 4][fail_mask]


# 35% efficiency run
fs35_ds = np.loadtxt("./../fof_time_series/fs035_ms10_fof_best_154_1368.txt")[::1, :]
fail_mask = fs35_ds[:, 3] > 10

# all results are fit filtered
f3_t_sim_myr = fs35_ds[:, 0][fail_mask]
f3_redshift = fs35_ds[:, 1][fail_mask]

f3_total_mass = fs35_ds[:, 2][fail_mask] + fs35_ds[:, 3][fail_mask]
f3_mass_in_gc = fs35_ds[:, 2][fail_mask]

f3_total_lum = fs35_ds[:, 4][fail_mask] + fs35_ds[:, 5][fail_mask]
f3_lum_in_gc = fs35_ds[:, 4][fail_mask]

# derived quantitiest
f7_bound_total_mass_ratio = f7_mass_in_gc / f7_total_mass
f7_bound_total_light_ratio = f7_lum_in_gc / f7_total_lum

f3_bound_total_mass_ratio = f3_mass_in_gc / f3_total_mass
f3_bound_total_light_ratio = f3_lum_in_gc / f3_total_lum
#%%#
with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 10,
    }
):
    with plt.style.context("dark_background"):
        cmap = cm.get_cmap("Set2")
        cmap = cmap(np.linspace(0, 1, 8))
        fig, ax = plt.subplots(
            nrows=2,
            ncols=2,
            gridspec_kw={"height_ratios": [5, 3]},
            sharex="col",
            sharey="row",
            figsize=(4, 3),
            dpi=400,
        )

        ax[0, 0].plot(
            f7_t_sim_myr,
            f7_total_mass,
            label=r"$\mathrm{Total}$",
            color=cmap[0],
            lw=2,
            alpha=0.8,
        )
        ax[0, 0].plot(
            f7_t_sim_myr,
            f7_mass_in_gc,
            label="$\mathrm{BSC}$",
            color=cmap[1],
            lw=2,
            alpha=0.8,
        )
        ax[1, 0].plot(f7_t_sim_myr, f7_bound_total_mass_ratio, lw=2, c="grey")
        # ax[2, 0].plot(
        #     f7_t_sim_myr,
        #     f7_total_lum,
        #     label=r"$\mathrm{Total}$",
        #     lw=2,
        #     c=cmap[0],
        #     alpha=0.8,
        # )
        # ax[2, 0].plot(
        #     f7_t_sim_myr,
        #     f7_lum_in_gc,
        #     label=r"$\mathrm{BSC}$",
        #     lw=2,
        #     c=cmap[1],
        #     alpha=0.8,
        # )
        # ax[3, 0].plot(f7_t_sim_myr, f7_bound_total_light_ratio, lw=2, c="grey")

        ax[1, 0].axhline(y=0.5, ls="--", c="grey", alpha=0.8)
        # ax[3, 0].axhline(y=0.5, ls="--", c="grey", alpha=0.8)

        ax[0, 0].set_yscale("log")
        # ax[2, 0].set_yscale("log")

        ax[0, 0].set_ylabel(r"$\mathrm{M}_{} \: (\mathrm{M}_{\odot})$", labelpad=10)
        ax[1, 0].set_ylabel(r"$\mathrm{M_{BSC}} / \mathrm{M_{Total}}$", labelpad=5)
        # ax[2, 0].set_ylabel(
        #     (
        #         r"$\mathrm{L}_{\lambda = 1500 \: \mathrm{\AA}} \:$"
        #         r"$(\mathrm{erg} \:\mathrm{s}^{-1} \:\mathrm{\AA}^{-1})$"
        #     )
        # )
        # ax[3, 0].set_ylabel(r"$\mathrm{L_{BSC}} / \mathrm{L_{Total}}$", labelpad=5)
        left_panel_twin_ax = ax[0, 0].twiny()
        left_panel_twin_ax.plot(
            f7_t_sim_myr,
            f7_total_mass,
            label=r"$\mathrm{Total}$",
            color=cmap[0],
            lw=0,
            alpha=0.8,
        )
        left_panel_twin_ax.set_xticklabels(
            list(
                np.round(z_from_t_myr(left_panel_twin_ax.get_xticks()), 1).astype("str")
            )
        )
        left_panel_twin_ax.set_xlim(
            left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr)
        )
        left_panel_twin_ax.tick_params(axis="both", direction="in", which="both")

        # left_panel_twin_ax = ax[0, 0].twiny()
        # left_panel_twin_ax.invert_xaxis()

        # left_panel_twin_ax.set_xlim(left=f7_redshift.max(), right=f7_redshift.min())

        ax[1, 0].set_xlim(left=f7_t_sim_myr.min(), right=f7_t_sim_myr.max())
        ax[1, 0].set_ylim(0.01, 1.1)
        # ax[3, 0].set_ylim(0.01, 1.1)

        ax[0, 0].legend(title="$f_{*} = 0.70$", loc="lower right", fontsize=8)

        # =================================30%============================================

        ax[0, 1].plot(
            f3_t_sim_myr,
            f3_total_mass,
            label=r"$\mathrm{Total}$",
            color=cmap[2],
            lw=2,
            alpha=0.8,
        )
        ax[0, 1].plot(
            f3_t_sim_myr,
            f3_mass_in_gc,
            label="$\mathrm{BSC}$",
            color=cmap[3],
            lw=2,
            alpha=0.8,
        )
        ax[1, 1].plot(f3_t_sim_myr, f3_bound_total_mass_ratio, lw=2, c="grey")
        # ax[2, 1].plot(
        #     f3_t_sim_myr,
        #     f3_total_lum,
        #     label=r"$\mathrm{Total}$",
        #     lw=2,
        #     c=cmap[2],
        #     alpha=0.8,
        # )
        # ax[2, 1].plot(
        #     f3_t_sim_myr,
        #     f3_lum_in_gc,
        #     label=r"$\mathrm{BSC}$",
        #     lw=2,
        #     c=cmap[3],
        #     alpha=0.8,
        # )
        # ax[3, 1].plot(f3_t_sim_myr, f3_bound_total_light_ratio, lw=2, c="grey")

        ax[1, 1].axhline(y=0.5, ls="--", c="grey", alpha=0.8)
        # ax[3, 1].axhline(y=0.5, ls="--", c="grey", alpha=0.8)

        right_panel_twin_ax = ax[0, 1].twiny()
        right_panel_twin_ax.plot(f3_t_sim_myr, f3_total_mass, lw=0)
        right_panel_twin_ax.set_xticklabels(
            list(
                np.round(z_from_t_myr(right_panel_twin_ax.get_xticks()), 1).astype(
                    "str"
                )
            )
        )
        right_panel_twin_ax.set_xlim(
            left=np.min(f3_t_sim_myr), right=np.max(f3_t_sim_myr)
        )
        right_panel_twin_ax.tick_params(axis="both", direction="in", which="both")

        ax[1, 1].set_xlim(left=f3_t_sim_myr.min(), right=f3_t_sim_myr.max())
        # ax[1, 1].set_ylim(0.05, 1.05)
        # ax[3, 1].set_ylim(0.05, 1.05)

        ax[0, 1].legend(title="$f_{*} = 0.35$", loc="lower right", fontsize=8)

        # left_panel_twin_ax.xaxis.set_major_locator(MaxNLocator(5))
        # ax[3, 0].xaxis.set_major_locator(MaxNLocator(5))
        # right_panel_twin_ax.xaxis.set_major_locator(MaxNLocator(5))
        # ax[3, 1].xaxis.set_major_locator(MaxNLocator(4))

        ax[1, 0].yaxis.set_major_locator(MaxNLocator(5))
        # ax[3, 0].yaxis.set_major_locator(MaxNLocator(5))

        fig.text(0.5, 0.01, "$\mathrm{t } \:(\mathrm{Myr})$", ha="center")
        fig.text(0.5, 0.95, "$\mathrm{z}$", ha="center")
        plt.subplots_adjust(hspace=0, wspace=0)

        ax[0, 0].tick_params(axis="both", direction="in", which="both")
        ax[1, 0].tick_params(axis="both", direction="in", which="both")
        # ax[2, 0].tick_params(axis="both", direction="in", which="both")
        # ax[3, 0].tick_params(axis="both", direction="in", which="both")
        ax[0, 1].tick_params(axis="both", direction="in", which="both")
        ax[1, 1].tick_params(axis="both", direction="in", which="both")
        # ax[2, 1].tick_params(axis="both", direction="in", which="both")
        # ax[3, 1].tick_params(axis="both", direction="in", which="both")
        plt.savefig(
            os.path.expanduser(
                (
                    "../../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/darkmode/"
                    "mass_bound_unbound.png"
                )
            ),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )

    # plt.savefig(
    #     os.path.expanduser(
    #         (
    #             "~/g_drive/Research/AstrophysicsSimulation/sci_plots/final/"
    #             "mass_bound_unbound.png"
    #         )
    #     ),
    #     dpi=500,
    #     bbox_inches="tight",
    #     pad_inches=0.05,
    # )


#%%
# # another version

# with plt.rc_context(
#     {
#         "font.family": "serif",
#         "mathtext.fontset": "cm",
#         "xtick.labelsize": 10,
#         "ytick.labelsize": 10,
#         "font.size": 12,
#     }
# ):
#     cmap = cm.get_cmap("Set2")
#     cmap = cmap(np.linspace(0, 1, 8))
#     fig, ax = plt.subplots(
#         nrows=2,
#         ncols=2,
#         gridspec_kw={"height_ratios": [5, 3]},
#         sharex="col",
#         sharey="row",
#         figsize=(7.5, 4),
#         dpi=400,
#     )
#     f7_field_mass = f7_total_mass - f7_mass_in_gc
#     f7_bound_field = f7_mass_in_gc / f7_field_mass
#     mask = f7_field_mass > 10

#     ax[0, 0].plot(
#         f7_t_sim_myr[mask],
#         f7_total_mass[mask],
#         label=r"$\mathrm{Total}$",
#         color=cmap[0],
#         lw=2,
#     )
#     ax[0, 0].plot(
#         f7_t_sim_myr[mask],
#         f7_mass_in_gc[mask],
#         label="$\mathrm{Bound}$",
#         color=cmap[1],
#         lw=2,
#     )
#     ax[0, 0].plot(
#         f7_t_sim_myr[mask],
#         f7_field_mass[mask],
#         label=r"$\mathrm{Field}$",
#         lw=2,
#         c=cmap[4],
#     )

#     ax[0, 0].set_yscale("log")
#     ax[0, 0].set_ylabel((r"$\mathrm{M}_{*} \: (\mathrm{M}_{\odot})$"))
#     ax[0, 0].legend(
#         # title="Pop II Stars",
#         loc="lower right",
#         fontsize=12,
#     )
#     # ax[0, 0].set_xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))

#     # these are matplotlib.patch.Patch properties
#     props = dict(
#         boxstyle="round",
#         facecolor="white",
#         alpha=0.5,
#         lw=0.8,
#         edgecolor="gray",
#     )
#     textstr = "$f_{*} = 0.70$"
#     # place a text box in upper left in axes coords
#     ax[0, 0].text(
#         0.05,
#         0.90,
#         textstr,
#         transform=ax[0, 0].transAxes,
#         fontsize=14,
#         verticalalignment="top",
#         bbox=props,
#     )

#     # add a twin axis
#     ax1_twin = ax[0, 0].twiny()
#     ax1_twin.invert_xaxis()
#     ax1_twin.xaxis.set_major_locator(MaxNLocator(5))
#     ax1_twin.set_xlim(left=np.max(f7_redshift), right=np.min(f7_redshift))

#     ax[1, 0].plot(f7_t_sim_myr[mask], f7_bound_field[mask], lw=2, c="grey")

#     ax[1, 0].axhline(y=1, ls="--", c="grey", alpha=0.8)
#     ax[1, 0].set_yscale("log")

#     ax[1, 0].set_ylabel(r"$\mathrm{M_{*,bound}} / \mathrm{M_{*,field}}$")
#     ax[1, 0].xaxis.set_major_locator(MaxNLocator(5))
#     ax[1, 0].set_xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))

#     # =================================30%============================================
#     f3_field_mass = f3_total_mass - f3_mass_in_gc
#     f3_bound_field = f3_mass_in_gc / f3_field_mass
#     mask = f3_field_mass > 10

#     ax[0, 1].plot(
#         f3_t_sim_myr, f3_total_mass, label=r"$\mathrm{Total}$", color=cmap[2], lw=3
#     )
#     ax[0, 1].plot(
#         f3_t_sim_myr,
#         f3_mass_in_gc,
#         label="$\mathrm{Bound}$",
#         color=cmap[3],
#         lw=2,
#     )
#     ax[0, 1].plot(
#         f3_t_sim_myr[mask],
#         f3_field_mass[mask],
#         label=r"$\mathrm{Field}$",
#         lw=2,
#         c=cmap[5],
#     )
#     ax[0, 1].legend(
#         # title="Pop II Stars",
#         loc="lower right",
#         fontsize=12,
#     )

#     # these are matplotlib.patch.Patch properties
#     props = dict(
#         boxstyle="round",
#         facecolor="white",
#         alpha=0.5,
#         lw=0.8,
#         edgecolor="gray",
#     )
#     textstr = "$f_{*} = 0.35$"
#     # place a text box in upper left in axes coords
#     ax[0, 1].text(
#         0.05,
#         0.90,
#         textstr,
#         transform=ax[0, 1].transAxes,
#         fontsize=14,
#         verticalalignment="top",
#         bbox=props,
#     )

#     # add a twin axis

#     ax2_twin = ax[0, 1].twiny()
#     ax2_twin.invert_xaxis()
#     ax2_twin.xaxis.set_major_locator(MaxNLocator(5))
#     ax2_twin.set_xlim(left=np.max(f3_redshift), right=np.min(f3_redshift))

#     ax[1, 1].plot(f3_t_sim_myr[mask], f3_bound_field[mask], lw=2, c="grey")

#     ax[1, 1].axhline(y=1, ls="--", c="grey", alpha=0.8)

#     ax[1, 1].set_xlim(left=np.min(f3_t_sim_myr), right=np.max(f3_t_sim_myr))
#     ax[1, 1].xaxis.set_major_locator(MaxNLocator(5))

#     fig.text(0.5, 0.03, "$\mathrm{t } \:(\mathrm{Myr})$", ha="center")
#     fig.text(0.5, 0.93, "$\mathrm{z}$", ha="center")

#     plt.subplots_adjust(hspace=0, wspace=0)
