import sys

sys.path.append("..")
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from modules.macros import filter_snapshots, common_filter_snapshots
import os

fof_ds = np.loadtxt("./fof_time_series/fs035_fof_series_results.txt")

fail_mask = fof_ds[:, 3] > 30

time = fof_ds[:, 0][fail_mask]
redshift = fof_ds[:, 1][fail_mask]
fof_bound_mass = fof_ds[:, 2][fail_mask]
fof_field_mass = fof_ds[:, 3][fail_mask]
fof_bound_lumi = fof_ds[:, 4][fail_mask]
fof_field_lumi = fof_ds[:, 5][fail_mask]
# prof_bound_mass = fof_ds[:, 6][fail_mask]
# profiler_field_mass = fof_ds[:, 7][fail_mask]
# prof_bound_lumi = fof_ds[:, 8][fail_mask]
# prof_field_lumi = fof_ds[:, 9][fail_mask]

fof_total_mass = fof_bound_mass + fof_field_mass
fof_total_lumi = fof_bound_lumi + fof_field_lumi
#%%# FOF bound and unbound mass
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

    ax[0].plot(time, fof_total_mass, label=r"Total", linewidth=4, c=cmap[2])
    ax[0].plot(time, fof_bound_mass, label=r"In GCs, FOF", linewidth=4, c=cmap[3])
    # ax[0].plot(t_sim_myr, core_mass_in_gc, label=r"Core Mass", linewidth=4, c=cmap[4])

    ax[0].set_yscale("log")
    ax[0].set_ylabel((r"$\mathrm{M}_{*} \: (\mathrm{M}_{\odot})$"))
    ax[0].legend(
        title="Pop II Stars",
        loc="lower right",
        fontsize=12,
    )
    ax[0].set_xlim(left=np.min(time), right=np.max(time))
    ax[0].set_ylim(4e2, 5e5)
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
    ax[0].text(
        0.05,
        0.90,
        textstr,
        transform=ax[0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    # the limits are controlled by 0.35 efficiency
    # add a twin axis
    ax[0].set_xlim(left=np.min(time), right=np.max(time))
    ax1_twin = ax[0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(redshift), right=np.min(redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    bound_total_ratio = fof_bound_mass / fof_total_mass
    ax[1].axhline(y=0.5, ls="--", c="black", alpha=0.5)
    ax[1].plot(time, bound_total_ratio, linewidth=4, c="grey")

    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1].set_ylabel(r"$\mathrm{In} \: \mathrm{GCs}  / \mathrm{Total}$")
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

    ax[0].plot(time, fof_total_mass, label=r"Total", linewidth=4, c=cmap[2])
    ax[0].plot(time, fof_bound_mass, label=r"In GCs, FOF", linewidth=4, c=cmap[3])
    ax[0].plot(time, fof_field_mass, label=r"Field", linewidth=4, c=cmap[1])

    ax[0].set_yscale("log")
    ax[0].set_ylabel((r"$\mathrm{M}_{*} \: (\mathrm{M}_{\odot})$"))
    ax[0].legend(
        title="Pop II Stars",
        loc="lower right",
        fontsize=12,
    )
    ax[0].set_xlim(left=np.min(time), right=np.max(time))
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
    ax[0].text(
        0.05,
        0.90,
        textstr,
        transform=ax[0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    # the limits are controlled by 0.35 efficiency
    # add a twin axis
    plt.xlim(left=np.min(time), right=np.max(time))
    ax1_twin = ax[0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(redshift), right=np.min(redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    bound_field = fof_bound_mass / fof_field_mass
    ax[1].axhline(y=1, ls="--", c="black", alpha=0.5)
    ax[1].plot(time, bound_field, linewidth=4, c="grey")

    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1].set_ylabel(r"$\mathrm{In} \: \mathrm{GCs}  / \mathrm{Field}$")
    ax[1].set_yscale("log")
    plt.subplots_adjust(hspace=0)

#%% FOF bound and unbound luminosity

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
    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True,
        figsize=(8, 6),
        dpi=300,
    )

    ax[0].plot(time, fof_total_lumi, label=r"Total", linewidth=4, c=cmap[0])
    ax[0].plot(time, fof_bound_lumi, label=r"In GCs, FOF", linewidth=4, c=cmap[3])
    # ax[0].plot(t_sim_myr, core_mass_in_gc, label=r"Core Mass", linewidth=4, c=cmap[4])

    ax[0].set_yscale("log")
    ax[0].set_ylabel(
        (
            r"$\mathrm{L}_{\lambda = 1500 \: \mathrm{\AA}} \:$"
            r"$(\mathrm{erg} \:\mathrm{s}^{-1} \:\mathrm{\AA}^{-1})$"
        )
    )
    ax[0].legend(title="Pop II Stars", loc="lower right", fontsize=12)
    ax[0].set_xlim(left=np.min(time), right=np.max(time))
    # ax[0].set_ylim(4e2, 5e5)
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
    ax[0].text(
        0.05,
        0.90,
        textstr,
        transform=ax[0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    # the limits are controlled by 0.35 efficiency add a twin axis

    ax[0].set_xlim(left=np.min(time), right=np.max(time))
    ax1_twin = ax[0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(redshift), right=np.min(redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    bound_total_light = fof_bound_lumi / fof_total_lumi
    ax[1].axhline(y=0.5, ls="--", c="black", alpha=0.5)
    ax[1].plot(time, bound_total_light, linewidth=4, c="grey")

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
    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True,
        figsize=(8, 6),
        dpi=300,
    )

    ax[0].plot(time, fof_total_lumi, label=r"Total", linewidth=4, c=cmap[0])
    ax[0].plot(time, fof_bound_lumi, label=r"In GCs, FOF", linewidth=4, c=cmap[3])
    ax[0].plot(time, fof_field_lumi, label=r"Field", linewidth=4, c=cmap[4])

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
    ax[0].set_xlim(left=np.min(time), right=np.max(time))
    # ax[0].set_ylim(4e2, 5e5)
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
    ax[0].text(
        0.05,
        0.90,
        textstr,
        transform=ax[0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    # the limits are controlled by 0.35 efficiency add a twin axis
    plt.xlim(left=np.min(time), right=np.max(time))
    ax1_twin = ax[0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(redshift), right=np.min(redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    bound_field = fof_bound_lumi / fof_field_lumi
    ax[1].axhline(y=1, ls="--", c="black", alpha=0.5)
    ax[1].plot(time, bound_field, linewidth=4, c="grey")

    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")

    ax[1].set_ylabel(r"$\mathrm{In} \: \mathrm{GCs} / \mathrm{Field}$")
    ax[1].set_yscale("log")
    # ax[1].set_ylim(0.05, 1.05)
    plt.subplots_adjust(hspace=0)
