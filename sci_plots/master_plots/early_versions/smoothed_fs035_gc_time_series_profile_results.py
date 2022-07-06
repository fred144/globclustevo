import sys

sys.path.append("..")
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from modules.macros import filter_snapshots, common_filter_snapshots
import os
from scipy.signal import find_peaks
from scipy.signal import hilbert
from scipy.signal import savgol_filter


#%%

dat_set = np.round_(
    np.loadtxt(
        "../gc_profiles/profile_runs/fs035_ms10/fof_best/time_series_run_stats.txt"
    )[:, :],
    3,
)


fail_mask = dat_set[:, 4] > 0
# all results are fit filtered, if gcs disappear due to no successful fits
t_sim_myr = dat_set[:, 1][fail_mask]
redshift = dat_set[:, 2][fail_mask]
total_star_counts = dat_set[:, 3][fail_mask]
stars_in_gc_counts = dat_set[:, 4][fail_mask]
total_mass = dat_set[:, 5][fail_mask]
mass_in_gc = dat_set[:, 6][fail_mask]
core_mass_in_gc = dat_set[:, 7][fail_mask]
total_lum = dat_set[:, 8][fail_mask]
lum_in_gc = dat_set[:, 9][fail_mask]

end_of_beginnig = 0  # int(np.argwhere(total_mass > 1e4)[0])

strt_t_sim_myr = t_sim_myr[:end_of_beginnig]
strt_redshift = redshift[:end_of_beginnig]
strt_total_star_counts = total_star_counts[:end_of_beginnig]
strt_stars_in_gc_counts = stars_in_gc_counts[:end_of_beginnig]
strt_total_mass = total_mass[:end_of_beginnig]
strt_mass_in_gc = mass_in_gc[:end_of_beginnig]
strt_core_mass_in_gc = core_mass_in_gc[:end_of_beginnig]
strt_total_lum = total_lum[:end_of_beginnig]
strt_lum_in_gc = lum_in_gc[:end_of_beginnig]

# only filtering this using find peaks
to_fltr_t_sim_myr = t_sim_myr[end_of_beginnig:]
to_fltr_redshift = redshift[end_of_beginnig:]
to_fltr_total_star_counts = total_star_counts[end_of_beginnig:]
to_fltr_stars_in_gc_counts = stars_in_gc_counts[end_of_beginnig:]
to_fltr_total_mass = total_mass[end_of_beginnig:]
to_fltr_mass_in_gc = mass_in_gc[end_of_beginnig:]
to_fltr_core_mass_in_gc = core_mass_in_gc[end_of_beginnig:]
to_fltr_total_lum = total_lum[end_of_beginnig:]
to_fltr_lum_in_gc = lum_in_gc[end_of_beginnig:]

filter_1, _ = find_peaks(to_fltr_mass_in_gc)

t_sim_myr = to_fltr_t_sim_myr[filter_1]
redshift = to_fltr_redshift[filter_1]
total_star_counts = to_fltr_total_star_counts[filter_1]
stars_in_gc_counts = to_fltr_stars_in_gc_counts[filter_1]
total_mass = to_fltr_total_mass[filter_1]
mass_in_gc = to_fltr_mass_in_gc[filter_1]
core_mass_in_gc = to_fltr_core_mass_in_gc[filter_1]
total_lum = to_fltr_total_lum[filter_1]
lum_in_gc = to_fltr_lum_in_gc[filter_1]


# double filter
filter_2, _ = find_peaks(mass_in_gc)
t_sim_myr = t_sim_myr[filter_2]
redshift = redshift[filter_2]
total_star_counts = total_star_counts[filter_2]
stars_in_gc_counts = stars_in_gc_counts[filter_2]
total_mass = total_mass[filter_2]
mass_in_gc = mass_in_gc[filter_2]
core_mass_in_gc = core_mass_in_gc[filter_2]
total_lum = total_lum[filter_2]
lum_in_gc = lum_in_gc[filter_2]

# rejoin after filtering after the start
t_sim_myr = np.concatenate((strt_t_sim_myr, t_sim_myr))
redshift = np.concatenate((strt_redshift, redshift))
total_star_counts = np.concatenate((strt_total_star_counts, total_star_counts))
stars_in_gc_counts = np.concatenate((strt_stars_in_gc_counts, stars_in_gc_counts))
total_mass = np.concatenate((strt_total_mass, total_mass))
mass_in_gc = np.concatenate((strt_mass_in_gc, mass_in_gc))
core_mass_in_gc = np.concatenate((strt_core_mass_in_gc, core_mass_in_gc))
total_lum = np.concatenate((strt_total_lum, total_lum))
lum_in_gc = np.concatenate((strt_lum_in_gc, lum_in_gc))


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
    plt.xlim(left=np.min(t_sim_myr), right=np.max(t_sim_myr))
    ax1_twin = ax[0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(redshift), right=np.min(redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    bound_total_ratio = mass_in_gc / total_mass
    ax[1].axhline(y=0.5, ls="--", c="black", alpha=0.5)
    ax[1].plot(t_sim_myr, bound_total_ratio, linewidth=4, c="grey")

    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1].set_ylabel(r"$\mathrm{In} \: \mathrm{GCs}  / \mathrm{Total}$")
    ax[1].set_ylim(0.05, 1.05)
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

    ax[0].plot(t_sim_myr, total_mass, label=r"Total", linewidth=4, c=cmap[2])
    ax[0].plot(t_sim_myr, mass_in_gc, label=r"In GCs, fitted", linewidth=4, c=cmap[3])
    field_mass = total_mass - mass_in_gc
    ax[0].plot(t_sim_myr, field_mass, label=r"Field", linewidth=4, c=cmap[1])

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
    plt.xlim(left=np.min(t_sim_myr), right=np.max(t_sim_myr))
    ax1_twin = ax[0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(redshift), right=np.min(redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    bound_field = mass_in_gc / field_mass
    ax[1].axhline(y=1, ls="--", c="black", alpha=0.5)
    mask = field_mass > 10
    ax[1].plot(t_sim_myr[mask], bound_field[mask], linewidth=4, c="grey")

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
    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True,
        figsize=(8, 6),
        dpi=300,
    )

    ax[0].plot(t_sim_myr, total_lum, label=r"Total", linewidth=4, c=cmap[0])
    ax[0].plot(t_sim_myr, lum_in_gc, label=r"In GCs, fitted", linewidth=4, c=cmap[3])
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
    ax[0].set_xlim(left=np.min(t_sim_myr), right=np.max(t_sim_myr))
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

    plt.xlim(left=np.min(t_sim_myr), right=np.max(t_sim_myr))
    ax1_twin = ax[0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(redshift), right=np.min(redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    bound_total_light = lum_in_gc / total_lum
    ax[1].axhline(y=0.5, ls="--", c="black", alpha=0.5)
    ax[1].plot(t_sim_myr, bound_total_light, linewidth=4, c="grey")

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

    ax[0].plot(t_sim_myr, total_lum, label=r"Total", linewidth=4, c=cmap[0])
    ax[0].plot(t_sim_myr, lum_in_gc, label=r"In GCs, fitted", linewidth=4, c=cmap[3])
    field_lum = total_lum - lum_in_gc

    ax[0].plot(t_sim_myr[mask], field_lum[mask], label=r"Field", linewidth=4, c=cmap[4])

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
    ax[0].set_xlim(left=np.min(t_sim_myr), right=np.max(t_sim_myr))
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

    plt.xlim(left=np.min(t_sim_myr), right=np.max(t_sim_myr))
    ax1_twin = ax[0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(redshift), right=np.min(redshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    bound_field = lum_in_gc / field_lum
    ax[1].axhline(y=1, ls="--", c="black", alpha=0.5)
    ax[1].plot(t_sim_myr[mask], bound_field[mask], linewidth=4, c="grey")

    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")

    ax[1].set_ylabel(r"$\mathrm{In} \: \mathrm{GCs} / \mathrm{Field}$")
    ax[1].set_yscale("log")
    # ax[1].set_ylim(0.05, 1.05)
    plt.subplots_adjust(hspace=0)
