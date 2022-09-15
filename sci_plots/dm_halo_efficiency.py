import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit

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

f7_dm_mass = f7_halo[:, 3]
f3_dm_mass = f3_halo[:, 3]

f7_dm_mass = f7_dm_mass / 1e8
f3_dm_mass = f3_dm_mass / 1e8


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

    sample_rate = 50

    f7_sample_rate = np.geomspace(1, f7_dm_mass.size - 1, sample_rate).astype(int)

    ax.scatter(
        np.log10(f7_dm_mass[f7_sample_rate]),
        np.log10((f7_star_mass / f7_dm_mass)[f7_sample_rate]),
        color=fs70_color,
        s=10,
        alpha=1,
        # ls=":",
        # label="$0.70$",
    )

    f7_params, f7_pcov = curve_fit(
        lin,
        np.log10(f7_dm_mass[f7_sample_rate]),
        np.log10((f7_star_mass / f7_dm_mass)[f7_sample_rate]),
    )

    plt.plot(
        np.log10(f7_dm_mass[f7_sample_rate]),
        lin(np.log10(f7_dm_mass[f7_sample_rate]), *f7_params),
        color=fs70_color,
        label=r"$\alpha_{{\: 70\%}} = {:.2f} \pm {:.2f}$"
        "\n"
        r"$\log_{{10}} \: \varepsilon_{{\: 70 \%}} = {:.2f} \pm {:.2f}$".format(
            f7_params[0],
            np.sqrt(np.diag(f7_pcov))[0],
            f7_params[1],
            np.sqrt(np.diag(f7_pcov))[1],
        ),
    )

    f3_sample_rate = np.geomspace(1, f3_dm_mass.size - 1, sample_rate).astype(int)
    ax.scatter(
        np.log10((f3_dm_mass)[f3_sample_rate]),
        np.log10((f3_star_mass / f3_dm_mass)[f3_sample_rate]),
        color=fs35_color,
        s=10,
        alpha=1,
        # ls=":",
        # label="$0.35$",
    )

    f3_params, f3_pcov = curve_fit(
        lin,
        np.log10(f3_dm_mass[f3_sample_rate]),
        np.log10((f3_star_mass / f3_dm_mass)[f3_sample_rate]),
    )

    plt.plot(
        np.log10(f3_dm_mass[f3_sample_rate]),
        lin(np.log10(f3_dm_mass[f3_sample_rate]), *f3_params),
        color=fs35_color,
        label=r"$\alpha_{{\: 35 \%}} = {:.2f} \pm {:.2f}$"
        "\n"
        r"$\log_{{10}} \: \varepsilon_{{\: 35 \%}} = {:.2f} \pm {:.2f}$".format(
            f3_params[0],
            np.sqrt(np.diag(f3_pcov))[0],
            f3_params[1],
            np.sqrt(np.diag(f3_pcov))[1],
        ),
    )

    ax.set(
        # yscale="log",
        # xscale="log",
        xlabel=r"$\mathrm{ \log_{10} \: M_{DM} \: \left( 10^{-8} \:M_{\odot} \right) }$",
        ylabel=r"$\mathrm{\log_{10} \: ( M_* / M_{DM} \times 10^{-8} } )  $",
    )

    legend = ax.legend(fontsize=11)
    legend.get_frame().set_alpha(0)
plt.savefig(
    "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/halo_eff.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.05,
)
