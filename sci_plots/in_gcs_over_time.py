import sys

sys.path.append("..")
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from modules.macros import filter_snapshots, common_filter_snapshots
import os

#%%


dat_set = np.loadtxt(
    "../gc_profiles/profile_runs/fs07_refine/fof_best_fb/time_series_run_stats.txt"
)
# stats for fitted gcs only
t_sim_myr = dat_set[:, 0]
redshift = dat_set[:, 1]
total_star_counts = dat_set[:, 2]
stars_in_gc_counts = dat_set[:, 3]
total_mass = dat_set[:, 4]
mass_in_gc = dat_set[:, 5]
core_mass_in_gc = dat_set[:, 6]
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
    ax[0].plot(t_sim_myr, core_mass_in_gc, label=r"Core Mass", linewidth=4, c=cmap[4])

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

#%% get the gc results from halo finder only
halo_data_directory = r"../halo_data/fs07_refine/fof_best"
pop2_data_directory = r"../particle_data/pop_2_data/fs07_refine"
strt = 113
end = 893
step = 10
pop2 = filter_snapshots(pop2_data_directory, strt, end, step)
halos_ds = filter_snapshots(halo_data_directory, strt, end, step)
pop2_ds = common_filter_snapshots(pop2, halos_ds)
#%%

t = []
rdshift = []
fof_bound_mass = []
fof_field_mass = []

for ds, p2 in zip(halos_ds, pop2_ds):
    t_myr, z = np.loadtxt(p2)[:2, 6]

    bound_mass = np.loadtxt(os.path.join(ds, "bound_stars.txt")).shape[0] * 10  # msun
    field_mass = np.loadtxt(os.path.join(ds, "field_stars.txt")).shape[0] * 10  # msun
    t.append(t_myr)
    rdshift.append(z)
    fof_bound_mass.append(bound_mass)
    fof_field_mass.append(field_mass)
    # x, y, z = np.loadtxt(os.path.join(ds, "field_stars.txt"))
t = np.array(t)
rdshift = np.array(rdshift)
fof_bound_mass = np.array(fof_bound_mass)
fof_field_mass = np.array(fof_field_mass)
fof_total_mass = fof_field_mass + fof_bound_mass
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
    cmap = cm.get_cmap("viridis_r")
    cmap = cmap(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6), dpi=300)

    ax[0].plot(t, fof_total_mass, label=r"Total", linewidth=4, c=cmap[2])
    ax[0].plot(t, fof_bound_mass, label=r"In GCs, FOF", linewidth=4, c=cmap[3])

    ax[0].set_yscale("log")
    ax[0].set_ylabel((r"$\mathrm{M}_{*} \: (\mathrm{M}_{\odot})$"))
    ax[0].legend(
        title="Pop II Stars",
        loc="lower right",
        fontsize=12,
    )
    ax[0].set_xlim(left=np.min(t), right=np.max(t))
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
    plt.xlim(left=np.min(t), right=np.max(t))
    ax1_twin = ax[0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(rdshift), right=np.min(rdshift))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    bound_unbound_ratio = fof_bound_mass / fof_total_mass
    ax[1].axhline(y=0.5, ls="--", c="grey", alpha=0.8)
    ax[1].plot(t, bound_unbound_ratio, linewidth=4, c=cmap[6])

    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1].set_ylabel(r"$\mathrm{In} \: \mathrm{GCs}  / \mathrm{Total}$")
    ax[1].set_ylim(0.05, 1.05)
    plt.subplots_adjust(hspace=0)

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), dpi=300)
(x, y, z) = np.loadtxt(os.path.join(ds, "field_stars.txt"))[:, 2:5].T
ax.scatter(x, y, color="tab:blue", s=0.5, alpha=0.08)
(x, y, z) = np.loadtxt(os.path.join(ds, "bound_stars.txt"))[:, 2:5].T
ax.scatter(x, y, color="tab:r", s=0.5, alpha=0.08)
ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
