import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))

fs70_color = cmap[1]
fs35_color = cmap[2]


# DM halo finder data
f3_halo = np.loadtxt("./dm_data/fs035_dm_halo_evo.txt")
f7_halo = np.loadtxt("./dm_data/fs070_dm_halo_evo.txt")

# f3_halo[f3_halo[:, 3].argsort()]
# f7_halo[f3_halo[:, 3].argsort()]

f7_dm_mass = f7_halo[:, 3]
f3_dm_mass = f3_halo[:, 3]

f7_star_mass = f7_halo[:, 5]
f3_star_mass = f3_halo[:, 5]

with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 12,
    }
):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4), dpi=300)

    sample_rate = 10

    ax.scatter(
        f7_dm_mass[::sample_rate],
        (f7_star_mass / f7_dm_mass)[::sample_rate],
        color=fs70_color,
        s=10,
        alpha=1,
        # ls=":",
        label="$0.70$",
    )

    ax.scatter(
        (f3_dm_mass)[::sample_rate],
        (f3_star_mass / f3_dm_mass)[::sample_rate],
        color=fs35_color,
        s=10,
        alpha=1,
        # ls=":",
        label="$0.25$",
    )

    ax.set(
        yscale="log",
        xlabel="$\mathrm{M_{DM} \: (M_{\odot})}$",
        ylabel="$\mathrm{M_* / M_{DM}}$",
    )

    ax.legend(title="$f_*$")
plt.savefig(
    "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/halo_eff.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.05,
)
