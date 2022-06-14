import sys

sys.path.append("..")
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from modules.macros import filter_snapshots, common_filter_snapshots
import os

#%%# fitted bound and unbound mass

dat_set = np.loadtxt(
    "../gc_profiles/profile_runs/fs07_refine/time_series_run_stats.txt"
)[:, :]


fail_mask = dat_set[:, 4] > 0

t_sim_myr = dat_set[:, 1][fail_mask]
redshift = dat_set[:, 2][fail_mask]
total_star_counts = dat_set[:, 3][fail_mask]
stars_in_gc_counts = dat_set[:, 4][fail_mask]
total_mass = dat_set[:, 5][fail_mask]
mass_in_gc = dat_set[:, 6][fail_mask]
core_mass_in_gc = dat_set[:, 7][fail_mask]


with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.size": 14,
    }
):
    cmap = cm.get_cmap("inferno_r")
    cmap = cmap(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6), dpi=300)

    ax[0].plot(t_sim_myr, total_mass, label=r"Total", linewidth=4, c=cmap[2])
    ax[0].plot(t_sim_myr, mass_in_gc, label=r"In GCs, fitted", linewidth=4, c=cmap[3])
    # ax[0].plot(t_sim_myr, core_mass_in_gc, label=r"Core Mass", linewidth=4, c=cmap[4])

    ax[0].set_yscale("log")
    ax[0].set_ylabel((r"$\mathrm{M}_{*} \: (\mathrm{M}_{\odot})$"))
    ax[0].legend(
        title="Pop II Stars",
        loc="lower right",
        fontsize=12,
    )
    ax[0].set_xlim(left=np.min(t_sim_myr), right=np.max(t_sim_myr))
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
    plt.xlim(left=np.min(t_sim_myr), right=np.max(t_sim_myr))
    ax1_twin = ax[0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(redshift), right=np.min(redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    bound_unbound_ratio = mass_in_gc / total_mass
    ax[1].axhline(y=0.5, ls="--", c="grey", alpha=0.8)
    ax[1].plot(t_sim_myr, bound_unbound_ratio, linewidth=4, c=cmap[6])

    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1].set_ylabel(r"$\mathrm{In} \: \mathrm{GCs}  / \mathrm{Total}$")
    ax[1].set_ylim(0.05, 1.05)
    plt.subplots_adjust(hspace=0)
