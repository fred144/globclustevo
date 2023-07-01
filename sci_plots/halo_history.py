import sys

sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from modules.macros import filter_snapshots, z_from_t_myr

cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))

fs70_color = cmap[1]
fs35_color = cmap[2]


def lin(x, m, b):
    return m * x + b


# DM halo finder data

f3_halo = np.loadtxt("./dm_data/fs035_dm_halo_evo.txt")
f7_halo = np.loadtxt("./dm_data/fs070_dm_halo_evo.txt")

# f3_halo[f3_halo[:, 3].argsort()]
# f7_halo[f3_halo[:, 3].argsort()]
# =============================================================================
# header is mislabeled in the file
# =============================================================================
f7_dm_mass = f7_halo[:, 3]
f3_dm_mass = f3_halo[:, 3]

# f7_dm_mass = f7_dm_mass
# f3_dm_mass = f3_dm_mass

f7_star_mass = f7_halo[:, 5]
f3_star_mass = f3_halo[:, 5]

f7_time = f7_halo[:, 1]
f3_time = f3_halo[:, 1]

f7_z = z_from_t_myr(f7_time)
f3_z = z_from_t_myr(f3_time)

f7_z_mask = f7_z >= 11.5
f3_z_mask = f3_z >= 11.5

# check the mass discrepancy.
print(f7_z[f7_z_mask][-1], f3_z[f3_z_mask][-1])

f7_test_mass = f7_dm_mass[f7_z_mask][-1]
f3_test_mass = f3_dm_mass[f3_z_mask][-1]

with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 12,
    }
):

    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(5, 4.5),
        dpi=300,
        gridspec_kw={"height_ratios": [5, 4]},
    )
    plt.subplots_adjust(hspace=0.35)

    sample_rate = 20
    f7_sample_rate = np.geomspace(1, f7_dm_mass.size - 1, sample_rate).astype(int)
    f3_sample_rate = np.geomspace(1, f3_dm_mass.size - 1, sample_rate).astype(int)
    f7_sample_rate[0] = 0
    f3_sample_rate[0] = 0

    thoery_x = np.geomspace(f7_dm_mass.min() * 0.9, f7_dm_mass.max() * 1.2, 100)

    ax[0].scatter(
        np.log10(f3_dm_mass)[f3_sample_rate],
        np.log10(f3_star_mass / f3_dm_mass)[f3_sample_rate],
        color=fs35_color,
        s=15,
        alpha=1
        # ls=":",
        # label="$0.35$",
    )

    f3_params, f3_pcov = curve_fit(
        lin,
        np.log10(f3_dm_mass)[f3_sample_rate],
        np.log10(f3_star_mass / f3_dm_mass)[f3_sample_rate],
    )

    ax[0].plot(
        np.log10(thoery_x),
        lin(np.log10(thoery_x), *f3_params),
        color=fs35_color,
        # ls="--",
        lw=2,
        alpha=0.8,
        label=r"${:.2f} \pm {:.2f}$".format(
            f3_params[0],
            np.sqrt(np.diag(f3_pcov))[0],
            # f3_params[1],
            # np.sqrt(np.diag(f3_pcov))[1],
            # r"$\log_{{10}} \: \varepsilon_{{\: 35 \%}} = {:.2f} \pm {:.2f}$".
        ),
    )

    ax[0].scatter(
        np.log10(f7_dm_mass)[f7_sample_rate],
        np.log10(f7_star_mass / f7_dm_mass)[f7_sample_rate],
        color=fs70_color,
        s=15,
        alpha=1,
        # ls=":",
        # label="$0.70$",
    )

    f7_params, f7_pcov = curve_fit(
        lin,
        np.log10(f7_dm_mass)[f7_sample_rate],
        np.log10(f7_star_mass / f7_dm_mass)[f7_sample_rate],
    )

    ax[0].plot(
        np.log10(thoery_x),
        lin(np.log10(thoery_x), *f7_params),
        color=fs70_color,
        # ls="--",
        lw=2,
        alpha=0.8,
        label=r"${:.2f} \pm {:.2f}$".format(
            f7_params[0],
            np.sqrt(np.diag(f7_pcov))[0],
        ),
    )

    ax[0].tick_params(axis="y", direction="in", which="both")
    ax[0].tick_params(axis="x", direction="in", which="both")
    ax[0].set(
        # yscale="log",
        # xscale="log",
        xlabel=r"$\mathrm{ \log_{10} \: M_{halo} \: \left( \: M_{\odot} \right) }$",
        ylabel=r"$\mathrm{\log_{10} \: ( M_* / M_{halo} } )  $",
        xlim=(
            np.log10(f3_dm_mass.min()),
            np.log10(f3_dm_mass.max()),
        ),
    )

    legend = ax[0].legend(
        title_fontsize=12,
        fontsize=11,
        title=r"$\log_{10}(s)$"
        # loc="lower right"
    )

    ax[1].plot(
        z_from_t_myr(f7_time), np.log10(f7_dm_mass), lw=2, alpha=1, color=fs70_color
    )
    ax[1].plot(
        z_from_t_myr(f3_time), np.log10(f3_dm_mass), lw=2, alpha=1, color=fs35_color
    )
    ax[1].invert_xaxis()
    ax[1].set(
        xlabel="$\mathrm{z}$",
        ylabel=r"$\mathrm{\log_{10} \: M_{halo} \left( \: M_{\odot} \right)}$",
        xlim=(
            z_from_t_myr(f7_time).max(),
            z_from_t_myr(f7_time).min(),
        ),
    )
    ax[1].tick_params(axis="both", direction="in", which="both")

    # legend.get_frame().set_alpha(0)

plt.savefig(
    "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/lowres/halo_history.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
# plt.savefig(
#     "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/halo_history.png",
#     dpi=500,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
