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


def consecutive_filter(x, y, percent_thresh):
    """If, in a 1d array, the y value changes by percent thresh with regards to previous
    value, drop that value.
    """
    consecutive_diff = np.diff(y) / y[:-1] * 100
    mask = consecutive_diff > percent_thresh
    mask = np.insert(mask, 0, True)
    filtered_x = x[mask]
    filtered_y = y[mask]
    return filtered_x, filtered_y, mask


# 70% efficiency run
fs70_ds = np.loadtxt(
    "../../gc_profiles/profile_runs/fs07_refine/time_series_run_stats.txt"
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


threshold_list = [-1, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10]

for thresh in threshold_list:
    f7_t_sim_myr, f7_mass_in_gc, ma = consecutive_filter(
        f7_t_sim_myr, f7_mass_in_gc, thresh
    )
    f7_redshift = f7_redshift[ma]
    f7_total_mass = f7_total_mass[ma]
    f7_total_lum = f7_total_lum[ma]
    f7_lum_in_gc = f7_lum_in_gc[ma]

# 35% efficiency run
fs35_ds = np.loadtxt(
    "../../gc_profiles/profile_runs/fs035_ms10/hi_fidelity_time_series_run_stats.txt"
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

for thresh in threshold_list:
    f3_t_sim_myr, f3_mass_in_gc, ma = consecutive_filter(
        f3_t_sim_myr, f3_mass_in_gc, thresh
    )
    f3_redshift = f3_redshift[ma]
    f3_total_mass = f3_total_mass[ma]
    f3_total_lum = f3_total_lum[ma]
    f3_lum_in_gc = f3_lum_in_gc[ma]
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
        ncols=2,
        gridspec_kw={"height_ratios": [5, 3]},
        sharex="col",
        sharey="row",
        figsize=(12, 6),
        dpi=300,
    )

    ax[0, 0].plot(f7_t_sim_myr, f7_total_mass, label=r"Total", color=cmap[0], lw=4)
    ax[0, 0].plot(
        f7_t_sim_myr, f7_mass_in_gc, label=r"In GCs, profiled", color=cmap[1], lw=4
    )

    ax[0, 0].set_yscale("log")
    ax[0, 0].set_ylabel((r"$\mathrm{M}_{*} \: (\mathrm{M}_{\odot})$"))
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

    # add a twin axis

    ax1_twin = ax[0, 0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(f7_redshift), right=np.min(f7_redshift))
    ax1_twin.xaxis.set_major_locator(MaxNLocator(7))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    f7_bound_total_ratio = f7_mass_in_gc / f7_total_mass

    ax[1, 0].plot(f7_t_sim_myr, f7_bound_total_ratio, linewidth=4, c="grey")

    ax[1, 0].axhline(y=0.5, ls="--", c="grey", alpha=0.8)
    ax[1, 0].yaxis.set_major_locator(MaxNLocator(4))
    ax[1, 0].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1, 0].set_ylabel(r"$\mathrm{In} \: \mathrm{GCs}  / \mathrm{Total}$")

    ax[1, 0].set_xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))

    # =================================30%============================================
    ax[0, 1].plot(f3_t_sim_myr, f3_total_mass, label=r"Total", color=cmap[2], lw=4)
    ax[0, 1].plot(
        f3_t_sim_myr, f3_mass_in_gc, label=r"In GCs,  profiled", color=cmap[3], lw=4
    )
    ax[0, 1].legend(
        title="Pop II Stars",
        loc="lower right",
        fontsize=12,
    )

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

    f3_bound_total_ratio = f3_mass_in_gc / f3_total_mass
    ax[1, 1].axhline(y=0.5, ls="--", c="grey", alpha=0.8)
    ax[1, 1].plot(f3_t_sim_myr, f3_bound_total_ratio, linewidth=4, c="grey")
    ax[1, 1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1, 1].set_xlim(left=np.min(f3_t_sim_myr), right=np.max(f3_t_sim_myr))

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
    f7_field_mass = f7_total_mass - f7_mass_in_gc
    f7_bound_field = f7_mass_in_gc / f7_field_mass
    mask = f7_field_mass > 10

    ax[0, 0].plot(
        f7_t_sim_myr[mask], f7_total_mass[mask], label=r"Total", color=cmap[0], lw=4
    )
    ax[0, 0].plot(
        f7_t_sim_myr[mask],
        f7_mass_in_gc[mask],
        label=r"In GCs, profiled",
        color=cmap[1],
        lw=4,
    )
    ax[0, 0].plot(
        f7_t_sim_myr[mask], f7_field_mass[mask], label=r"Field", linewidth=4, c=cmap[4]
    )

    ax[0, 0].set_yscale("log")
    ax[0, 0].set_ylabel((r"$\mathrm{M}_{*} \: (\mathrm{M}_{\odot})$"))
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

    # add a twin axis

    ax1_twin = ax[0, 0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(f7_redshift), right=np.min(f7_redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    ax[1, 0].axhline(y=1, ls="--", c="grey", alpha=0.8)

    ax[1, 0].plot(f7_t_sim_myr[mask], f7_bound_field[mask], linewidth=4, c="grey")

    ax[1, 0].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1, 0].set_ylabel(r"$\mathrm{In} \: \mathrm{GCs}  / \mathrm{Field}$")

    ax[1, 0].set_xlim(left=np.min(f7_t_sim_myr), right=np.max(f7_t_sim_myr))
    ax[1, 0].set_yscale("log")

    # =================================30%============================================
    f3_field_mass = f3_total_mass - f3_mass_in_gc
    f3_bound_field = f3_mass_in_gc / f3_field_mass
    mask = f3_field_mass > 10

    ax[0, 1].plot(f3_t_sim_myr, f3_total_mass, label=r"Total", color=cmap[2], lw=4)
    ax[0, 1].plot(
        f3_t_sim_myr,
        f3_mass_in_gc,
        label=r"In GCs,  profiled",
        color=cmap[3],
        lw=4,
    )
    ax[0, 1].plot(
        f3_t_sim_myr[mask], f3_field_mass[mask], label=r"Field", linewidth=4, c=cmap[5]
    )
    ax[0, 1].legend(
        title="Pop II Stars",
        loc="lower right",
        fontsize=12,
    )

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

    ax[1, 1].axhline(y=1, ls="--", c="grey", alpha=0.8)
    ax[1, 1].plot(f3_t_sim_myr[mask], f3_bound_field[mask], linewidth=4, c="grey")
    ax[1, 1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")

    ax[1, 1].set_xlim(left=np.min(f3_t_sim_myr), right=np.max(f3_t_sim_myr))

    plt.subplots_adjust(hspace=0, wspace=0)
