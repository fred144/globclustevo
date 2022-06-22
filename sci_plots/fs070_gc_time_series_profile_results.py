import sys

sys.path.append("..")
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
fs70_ds = np.loadtxt(
    "../gc_profiles/profile_runs/fs07_refine/time_series_run_stats.txt"
)[::1, :]
fail_mask = fs70_ds[:, 4] > 1

# all results are fit filtered
f7_t_sim_myr = fs70_ds[:, 1][fail_mask]
f7_redshift = fs70_ds[:, 2][fail_mask]

f7_total_mass = fs70_ds[:, 5][fail_mask]
f7_mass_in_gc = fs70_ds[:, 6][fail_mask]

f7_total_lum = fs70_ds[:, 8][fail_mask]
f7_lum_in_gc = fs70_ds[:, 9][fail_mask]

# f7_core_mass_in_gc =fs70_ds[:, 7][fail_mask]


# 35% efficiency run
fs35_ds = np.loadtxt(
    "../gc_profiles/profile_runs/fs035_ms10/hi_fidelity_time_series_run_stats.txt"
)[::1, :]
fail_mask = fs35_ds[:, 4] > 1

# all results are fit filtered
f3_t_sim_myr = fs35_ds[:, 1][fail_mask]
f3_redshift = fs35_ds[:, 2][fail_mask]

f3_total_mass = fs35_ds[:, 5][fail_mask]
f3_mass_in_gc = fs35_ds[:, 6][fail_mask]

f3_total_lum = fs35_ds[:, 8][fail_mask]
f3_lum_in_gc = fs35_ds[:, 9][fail_mask]

# f3_core_mass_in_gc =fs35_ds[:, 7][fail_mask]
#%%# fitted bound and unbound mass
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
        ncols=1,
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True,
        figsize=(8, 6),
        dpi=300,
    )

    ax[0].plot(f7_t_sim_myr, f7_total_mass, label=r"Total", color=cmap[2], lw=4)
    ax[0].plot(
        f7_t_sim_myr, f7_mass_in_gc, label=r"In GCs, fitted", color=cmap[3], lw=4
    )

    ax[0].set_yscale("log")
    ax[0].set_ylabel((r"$\mathrm{M}_{*} \: (\mathrm{M}_{\odot})$"))
    ax[0].legend(
        title="Pop II Stars",
        loc="lower right",
        fontsize=12,
    )
    ax[0].set_xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))
    # ax[0].set_ylim(4e2, 5e5)
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
    ax[0].text(
        0.05,
        0.90,
        textstr,
        transform=ax[0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    # the limits are controlled by 0.70 efficiency
    # add a twin axis
    plt.xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))
    ax1_twin = ax[0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(f7_redshift), right=np.min(f7_redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    f7_bound_total_ratio = f7_mass_in_gc / f7_total_mass
    ax[1].axhline(y=0.5, ls="--", c="grey", alpha=0.8)
    ax[1].plot(f7_t_sim_myr, f7_bound_total_ratio, linewidth=4, c=cmap[4])

    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1].set_ylabel(r"$\mathrm{In} \: \mathrm{GCs}  / \mathrm{Total}$")
    ax[1].set_ylim(0.05, 1.05)
    ax[1].yaxis.set_major_locator(MaxNLocator(4))
    plt.subplots_adjust(hspace=0)
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
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6), dpi=300)

    ax[0].plot(f7_t_sim_myr, f7_total_mass, label=r"Total", linewidth=4, c=cmap[2])
    ax[0].plot(
        f7_t_sim_myr, f7_mass_in_gc, label=r"In GCs, fitted", linewidth=4, c=cmap[3]
    )
    f7_field_mass = f7_total_mass - f7_mass_in_gc
    ax[0].plot(f7_t_sim_myr, f7_field_mass, label=r"Field", linewidth=4, c=cmap[1])

    ax[0].set_yscale("log")
    ax[0].set_ylabel((r"$\mathrm{M}_{*} \: (\mathrm{M}_{\odot})$"))
    ax[0].legend(
        title="Pop II Stars",
        loc="lower right",
        fontsize=12,
    )
    ax[0].set_xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))
    ax[0].set_ylim(4e2, 5e5)
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
    ax[0].text(
        0.05,
        0.90,
        textstr,
        transform=ax[0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    # the limits are controlled by 0.70 efficiency
    # add a twin axis
    plt.xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))
    ax1_twin = ax[0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(f7_redshift), right=np.min(f7_redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    f7_bound_field = f7_mass_in_gc / f7_field_mass
    ax[1].axhline(y=1, ls="--", c="grey", alpha=0.8)
    mask = f7_field_mass > 0
    ax[1].plot(f7_t_sim_myr[mask], f7_bound_field[mask], linewidth=4, c=cmap[7])

    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")

    ax[1].set_ylabel(r"$\mathrm{In} \: \mathrm{GCs}  / \mathrm{Field}$")
    ax[1].set_yscale("log")
    ax[1].set_ylim(0.05, 15)
    plt.subplots_adjust(hspace=0)

#%% fitted bound and unbound luminosity

with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.size": 14,
    }
):
    cmap = cm.get_cmap("Set3")
    cmap = cmap(np.linspace(0, 1, 11))
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6), dpi=300)

    ax[0].plot(f7_t_sim_myr, f7_total_lum, label=r"Total", linewidth=4, c=cmap[0])
    ax[0].plot(
        f7_t_sim_myr, f7_lum_in_gc, label=r"In GCs, fitted", linewidth=4, c=cmap[3]
    )
    # ax[0].plot(t_sim_myr, core_mass_in_gc, label=r"Core Mass", linewidth=4, c=cmap[4])

    ax[0].set_yscale("log")
    ax[0].set_ylabel(
        (
            r"$\mathrm{L}_{\lambda = 1500 \: \mathrm{\AA}} \:$"
            r"$(\mathrm{erg} \:\mathrm{s}^{-1} \:\mathrm{\AA}^{-1})$"
        )
    )
    ax[0].legend(
        title="Pop II Stars",
        loc="lower right",
        fontsize=12,
    )
    ax[0].set_xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))
    # ax[0].set_ylim(4e2, 5e5)
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
    ax[0].text(
        0.05,
        0.90,
        textstr,
        transform=ax[0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    # the limits are controlled by 0.70 efficiency add a twin axis

    plt.xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))
    ax1_twin = ax[0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(f7_redshift), right=np.min(f7_redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    f7_bound_total_light = f7_lum_in_gc / f7_total_lum
    ax[1].axhline(y=0.5, ls="--", c="grey", alpha=0.8)
    ax[1].plot(f7_t_sim_myr, f7_bound_total_light, linewidth=4, c=cmap[8])

    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    # ax[1].set_xscale("log")
    ax[1].set_ylabel(r"$\mathrm{In} \: \mathrm{GCs} / \mathrm{Total}$")
    # ax[1].set_ylim(0.05, 1.05)
    plt.subplots_adjust(hspace=0)

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
    cmap = cm.get_cmap("Set3")
    cmap = cmap(np.linspace(0, 1, 11))
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6), dpi=300)

    ax[0].plot(f7_t_sim_myr, f7_total_lum, label=r"Total", linewidth=4, c=cmap[0])
    ax[0].plot(
        f7_t_sim_myr, f7_lum_in_gc, label=r"In GCs, fitted", linewidth=4, c=cmap[3]
    )
    f7_field_lum = f7_total_lum - f7_lum_in_gc

    ax[0].plot(
        f7_t_sim_myr[mask], f7_field_lum[mask], label=r"Field", linewidth=4, c=cmap[4]
    )

    ax[0].set_yscale("log")
    ax[0].set_ylabel(
        (
            r"$\mathrm{L}_{\lambda = 1500 \: \mathrm{\AA}} \:$"
            r"$(\mathrm{erg} \:\mathrm{s}^{-1} \:\mathrm{\AA}^{-1})$"
        )
    )
    ax[0].legend(
        title="Pop II Stars",
        loc="lower right",
        fontsize=12,
    )
    ax[0].set_xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))
    # ax[0].set_ylim(4e2, 5e5)
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
    ax[0].text(
        0.05,
        0.90,
        textstr,
        transform=ax[0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    # the limits are controlled by 0.70 efficiency add a twin axis

    plt.xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))
    ax1_twin = ax[0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(f7_redshift), right=np.min(f7_redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    f7_bound_field = f7_lum_in_gc / f7_field_lum
    ax[1].axhline(y=1, ls="--", c="grey", alpha=0.8)
    ax[1].plot(f7_t_sim_myr[mask], f7_bound_field[mask], linewidth=4, c=cmap[8])

    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")

    ax[1].set_ylabel(r"$\mathrm{In} \: \mathrm{GCs} / \mathrm{Field}$")
    ax[1].set_yscale("log")
    # ax[1].set_ylim(0.05, 1.05)
    plt.subplots_adjust(hspace=0)
