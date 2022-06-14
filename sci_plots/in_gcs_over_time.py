import sys

sys.path.append("..")
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from modules.macros import filter_snapshots, common_filter_snapshots
import os


#%% get the gc results from halo finder only
halo_data_directory = r"../halo_data/fs07_refine/fof_best"
pop2_data_directory = r"../particle_data/pop_2_data/fs07_refine"
strt = 113
end = 917
step = 1
pop2 = filter_snapshots(pop2_data_directory, strt, end, step)
halos_ds = filter_snapshots(halo_data_directory, strt, end, step)
pop2_ds = common_filter_snapshots(pop2, halos_ds)
#%% fof bound and unbound

t = []
rdshift = []

fof_bound_mass = []
fof_field_mass = []

prof_bound_mass = []
prof_field_mass = []

fof_bound_lumi = []
fof_field_lumi = []

prof_bound_lumi = []
prof_field_lumi = []


for i, (ds, p2) in enumerate(zip(halos_ds, pop2_ds)):
    print(i)
    p_ii = np.loadtxt(p2)
    t_myr, z = p_ii[:2, 6]
    # pure friends of friends algorithm
    bound = np.loadtxt(os.path.join(ds, "bound_stars.txt"))
    field = np.loadtxt(os.path.join(ds, "field_stars.txt"))
    # exclude if not fitted
    bound_fitted = np.loadtxt(os.path.join(ds, "fitted_bound_stars.txt"))
    field_fitted = np.loadtxt(os.path.join(ds, "fitted_field_stars.txt"))
    try:
        lum_bound = np.sum(bound[:, 2])
        m_bound = np.sum(bound[:, 6])

    except:
        lum_bound = 0
        m_bound = 0

    try:
        lum_field = np.sum(field[:, 2])
        m_field = np.sum(field[:, 6])
    except:
        lum_field = 0
        m_field = 0

    try:
        lum_bound_fitted = np.sum(bound_fitted[:, 2])
        m_bound_fitted = np.sum(bound_fitted[:, 6])
    except:
        lum_bound_fitted = 0
        m_bound_fitted = 0

    try:
        lum_field_fitted = np.sum(field_fitted[:, 2])
        m_field_fitted = np.sum(field_fitted[:, 6])
    except:
        lum_field_fitted = 0
        m_field_fitted = 0

    t.append(t_myr)
    rdshift.append(z)

    fof_bound_mass.append(m_bound)
    fof_field_mass.append(m_field)
    prof_bound_mass.append(m_bound_fitted)
    prof_field_mass.append(m_field_fitted)

    fof_bound_lumi.append(lum_bound)
    fof_field_lumi.append(lum_field)
    prof_bound_lumi.append(lum_bound_fitted)
    prof_field_lumi.append(lum_field_fitted)

# x, y, z = np.loadtxt(os.path.join(ds, "field_stars.txt"))
t = np.array(t)
rdshift = np.array(rdshift)

fof_bound_mass = np.array(fof_bound_mass)
fof_field_mass = np.array(fof_field_mass)

prof_bound_mass = np.array(prof_bound_mass)
prof_field_mass = np.array(prof_field_mass)

fof_bound_lumi = np.array(fof_bound_lumi)
fof_field_lumi = np.array(fof_field_lumi)

prof_bound_lumi = np.array(prof_bound_lumi)
prof_field_lumi = np.array(prof_field_lumi)
#%%
t = np.hstack(
    (
        fof_bound_mass,
        fof_field_mass,
        fof_bound_lumi,
        fof_field_lumi,
        prof_bound_mass,
        prof_field_mass,
        prof_bound_lumi,
        prof_field_lumi,
    )
)

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

    ax[0].plot(
        t, fof_bound_mass + fof_field_mass, label=r"Total", linewidth=4, c=cmap[2]
    )
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

    bound_unbound_ratio = fof_bound_mass / (fof_bound_mass + fof_field_mass)
    ax[1].axhline(y=0.5, ls="--", c="grey", alpha=0.8)
    ax[1].plot(t, bound_unbound_ratio, linewidth=4, c=cmap[6])

    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1].set_ylabel(r"$\mathrm{In} \: \mathrm{GCs}  / \mathrm{Total}$")
    ax[1].set_ylim(0.05, 1.05)
    plt.subplots_adjust(hspace=0)

#%% fof performance map scatter
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), dpi=300)
(x, y, z) = np.loadtxt(os.path.join(ds, "field_stars.txt"))[:, 2:5].T
ax.scatter(x, y, c="tab:blue", s=0.5, alpha=0.08)
(x, y, z) = np.loadtxt(os.path.join(ds, "bound_stars.txt"))[:, 2:5].T
ax.scatter(x, y, c="tab:blue", s=0.5, alpha=0.08)
ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
#%% field stars over bound stars

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
    cmap = cmap(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6), dpi=300)

    ax[0].plot(t, fof_total_mass, label=r"Total", linewidth=4, c=cmap[1])
    ax[0].plot(t, fof_bound_mass, label=r"In GCs, FOF", linewidth=4, c=cmap[2])
    ax[0].plot(
        t,
        fof_total_mass - fof_bound_mass,
        label=r"Field Stars",
        linewidth=4,
        c=cmap[3],
    )

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

    bound_unbound_ratio = fof_bound_mass / (fof_total_mass - fof_bound_mass)
    ax[1].axhline(y=1, ls="--", c="grey", alpha=0.8)
    ax[1].plot(t, bound_unbound_ratio, linewidth=4, c=cmap[4])

    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1].set_ylabel(r"$ \mathrm{Bound}  / \mathrm{Field} \: \mathrm{Stars} $")
    # ax[1].set_ylim(0.05, 1.05)
    ax[1].set_yscale("log")
    plt.subplots_adjust(hspace=0)
