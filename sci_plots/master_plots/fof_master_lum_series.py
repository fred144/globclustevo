import sys

sys.path.append("../../")
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


# 70% efficiency run
fs70_ds = np.loadtxt("../fof_time_series/fs070_fof_series_results.txt")
fail_mask = fs70_ds[:, 3] > 10


# all results are fit filtered
f7_t_sim_myr = fs70_ds[:, 0][fail_mask]
f7_redshift = fs70_ds[:, 1][fail_mask]

# f7_total_mass = fs70_ds[:, 5][fail_mask]
# f7_mass_in_gc = fs70_ds[:, 6][fail_mask]

f7_total_lum = fs70_ds[:, 4][fail_mask] + fs70_ds[:, 5][fail_mask]
f7_lum_in_gc = fs70_ds[:, 4][fail_mask]

# f7_core_mass_in_gc =fs70_ds[:, 7][fail_mask]

# 35% efficiency run
fs35_ds = np.loadtxt("../fof_time_series/fs035_fof_series_results.txt")
fail_mask = fs35_ds[:, 3] > 10

# all results are fit filtered
f3_t_sim_myr = fs35_ds[:, 0][fail_mask]
f3_redshift = fs35_ds[:, 1][fail_mask]

# f3_total_mass = fs35_ds[:, 5][fail_mask]
# f3_mass_in_gc = fs35_ds[:, 6][fail_mask]

f3_total_lum = fs35_ds[:, 4][fail_mask] + fs35_ds[:, 5][fail_mask]
f3_lum_in_gc = fs35_ds[:, 4][fail_mask]

# f3_core_mass_in_gc =fs35_ds[:, 7][fail_mask]

#%%
with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.size": 14,
    }
):
    cmap = cm.get_cmap("Set2")
    cmap = cmap(np.linspace(0, 1, 8))
    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        gridspec_kw={"height_ratios": [5, 3]},
        sharex="col",
        sharey="row",
        figsize=(12, 6),
        dpi=300,
    )
    f7_bound_total_light = f7_lum_in_gc / f7_total_lum

    ax[0, 0].plot(f7_t_sim_myr, f7_total_lum, label=r"Total", linewidth=4, c=cmap[0])
    ax[0, 0].plot(
        f7_t_sim_myr, f7_lum_in_gc, label=r"Bound, FOF", linewidth=4, c=cmap[1]
    )
    # ax[0, 0].scatter(f7_t_sim_myr, f7_total_lum, label=r"Total", s =3, c=cmap[0])
    # ax[0, 0].scatter(
    #     f7_t_sim_myr, f7_lum_in_gc, label=r"Bound, FOF",s =3, c=cmap[1]
    # )
    # ax[0].plot(t_sim_myr, core_mass_in_gc, label=r"Core Mass", linewidth=4, c=cmap[4])

    ax[0, 0].set_yscale("log")
    ax[0, 0].set_ylabel(
        (
            r"$\mathrm{L}_{\lambda = 1500 \: \mathrm{\AA}} \:$"
            r"$(\mathrm{erg} \:\mathrm{s}^{-1} \:\mathrm{\AA}^{-1})$"
        )
    )
    ax[0, 0].legend(
        title="Pop II Stars",
        loc="lower right",
        fontsize=12,
    )
    ax[0, 0].set_xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))

    # these are matplotlib.patch.Patch properties
    props = dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.5,
        linewidth=0.8,
        edgecolor="gray",
    )
    textstr = "$f_{*} = 0.70$"
    # place a text box in upper left in axes coords
    ax[0, 0].text(
        0.05,
        0.90,
        textstr,
        transform=ax[0, 0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    ax1_twin = ax[0, 0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(f7_redshift), right=np.min(f7_redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    ax[1, 0].plot(f7_t_sim_myr, f7_bound_total_light, linewidth=4, c="grey")

    ax[1, 0].axhline(y=0.5, ls="--", c="grey", alpha=0.8)
    ax[1, 0].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1, 0].xaxis.set_major_locator(MaxNLocator(7))
    ax[1, 0].yaxis.set_major_locator(MaxNLocator(4))
    ax[1, 0].set_ylabel(r"$\mathrm{Bound} / \mathrm{Total}$")
    ax[1, 0].set_xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))

    # =================================30%============================================

    f3_bound_total_light = f3_lum_in_gc / f3_total_lum

    ax[0, 1].plot(f3_t_sim_myr, f3_total_lum, label=r"Total", color=cmap[2], lw=4)
    ax[0, 1].plot(f3_t_sim_myr, f3_lum_in_gc, label=r"Bound,  FOF", color=cmap[3], lw=4)
    ax[0, 1].legend(
        title="Pop II Stars",
        loc="lower right",
        fontsize=12,
    )
    ax[0, 1].set_xlim(left=np.min(f3_t_sim_myr), right=np.max(f3_t_sim_myr))
    # these are matplotlib.patch.Patch properties
    props = dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.5,
        linewidth=0.8,
        edgecolor="gray",
    )
    textstr = "$f_{*} = 0.35$"
    # place a text box in upper left in axes coords
    ax[0, 1].text(
        0.05,
        0.90,
        textstr,
        transform=ax[0, 1].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    # add a twin axis

    ax2_twin = ax[0, 1].twiny()
    ax2_twin.invert_xaxis()
    ax2_twin.set_xlim(left=np.max(f3_redshift), right=np.min(f3_redshift))
    ax2_twin.set(xlabel="$\mathrm{z}$")

    ax[1, 1].plot(f3_t_sim_myr, f3_bound_total_light, linewidth=4, c="grey")

    ax[1, 1].axhline(y=0.5, ls="--", c="grey", alpha=0.8)
    ax[1, 1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1, 1].set_xlim(left=np.min(f3_t_sim_myr), right=np.max(f3_t_sim_myr))
    ax[1, 1].xaxis.set_major_locator(MaxNLocator(7))
    plt.subplots_adjust(hspace=0, wspace=0)

#%%
# another version

with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.size": 14,
    }
):
    cmap = cm.get_cmap("Set2")
    cmap = cmap(np.linspace(0, 1, 8))
    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        gridspec_kw={"height_ratios": [5, 3]},
        sharex="col",
        sharey="row",
        figsize=(12, 6),
        dpi=300,
    )
    f7_field_lum = f7_total_lum - f7_lum_in_gc
    f7_bound_field = f7_lum_in_gc / f7_field_lum
    mask = f7_field_lum > 1e30

    ax[0, 0].plot(
        f7_t_sim_myr[mask], f7_total_lum[mask], label=r"Total", linewidth=4, c=cmap[0]
    )
    ax[0, 0].plot(
        f7_t_sim_myr[mask],
        f7_lum_in_gc[mask],
        label=r"Bound, FOF",
        linewidth=4,
        c=cmap[1],
    )
    ax[0, 0].plot(
        f7_t_sim_myr[mask], f7_field_lum[mask], label=r"Field", linewidth=4, c=cmap[4]
    )

    ax[0, 0].set_yscale("log")
    ax[0, 0].set_ylabel(
        (
            r"$\mathrm{L}_{\lambda = 1500 \: \mathrm{\AA}} \:$"
            r"$(\mathrm{erg} \:\mathrm{s}^{-1} \:\mathrm{\AA}^{-1})$"
        )
    )
    ax[0, 0].legend(
        title="Pop II Stars",
        loc="lower right",
        fontsize=12,
    )
    ax[0, 0].set_xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))

    # these are matplotlib.patch.Patch properties
    props = dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.5,
        linewidth=0.8,
        edgecolor="gray",
    )
    textstr = "$f_{*} = 0.70$"
    # place a text box in upper left in axes coords
    ax[0, 0].text(
        0.05,
        0.90,
        textstr,
        transform=ax[0, 0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    ax1_twin = ax[0, 0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(f7_redshift), right=np.min(f7_redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    ax[1, 0].plot(f7_t_sim_myr[mask], f7_bound_field[mask], linewidth=4, c="grey")

    ax[1, 0].axhline(y=0.5, ls="--", c="grey", alpha=0.8)

    ax[1, 0].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1, 0].set_xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))
    ax[1, 0].set_ylabel(r"$\mathrm{Bound} / \mathrm{Field}$")
    ax[1, 0].xaxis.set_major_locator(MaxNLocator(7))
    ax[1, 0].yaxis.set_major_locator(MaxNLocator(4))
    ax[1, 0].set_yscale("log")
    # =================================30%============================================

    f3_field_lum = f3_total_lum - f3_lum_in_gc
    f3_bound_field = f3_lum_in_gc / f3_field_lum
    mask = f3_field_lum > 1e30

    ax[0, 1].plot(f3_t_sim_myr, f3_total_lum, label=r"Total", color=cmap[2], lw=4)
    ax[0, 1].plot(
        f3_t_sim_myr[mask],
        f3_lum_in_gc[mask],
        label=r"Bound,  FOF",
        color=cmap[3],
        lw=4,
    )
    ax[0, 1].plot(
        f3_t_sim_myr[mask], f3_field_lum[mask], label=r"Field", linewidth=4, c=cmap[5]
    )

    ax[0, 1].legend(
        title="Pop II Stars",
        loc="lower right",
        fontsize=12,
    )
    ax[0, 1].set_xlim(left=np.min(f3_t_sim_myr), right=np.max(f3_t_sim_myr))
    # these are matplotlib.patch.Patch properties
    props = dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.5,
        linewidth=0.8,
        edgecolor="gray",
    )
    textstr = "$f_{*} = 0.35$"
    # place a text box in upper left in axes coords
    ax[0, 1].text(
        0.05,
        0.90,
        textstr,
        transform=ax[0, 1].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    # add a twin axis

    ax2_twin = ax[0, 1].twiny()
    ax2_twin.invert_xaxis()
    ax2_twin.set_xlim(left=np.max(f3_redshift), right=np.min(f3_redshift))
    ax2_twin.set(xlabel="$\mathrm{z}$")

    ax[1, 1].plot(f3_t_sim_myr[mask], f3_bound_field[mask], linewidth=4, c="grey")

    ax[1, 1].axhline(y=0.5, ls="--", c="grey", alpha=0.8)
    ax[1, 1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1, 1].set_xlim(left=np.min(f3_t_sim_myr), right=np.max(f3_t_sim_myr))
    ax[1, 1].xaxis.set_major_locator(MaxNLocator(7))
    plt.subplots_adjust(hspace=0, wspace=0)
