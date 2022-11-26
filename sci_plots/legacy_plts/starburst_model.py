import numpy as np
import matplotlib.pyplot as plt
import os

lum_model = np.loadtxt("../particle_data/luminosity_look_up_tables/l1500_inst_e.txt")
# Monochromatic luminosity at 1500 Å vs. time. Star-formation law: instantaneous.
# Solid line: alpha -2.35
time_myr = lum_model[:, 0] / 1e6
luminosity_a235 = 10 ** lum_model[:, 1] * 1e-5  # M = 10e6 M_sun, so scale with 1e-5

with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 10,
    }
):
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.8), dpi=400)
    legend = (
        r"$\alpha = 2.35$"
        "\n"
        r"$\mathrm{M = 10 \: M_{\odot}}$"
        "\n"
        r"$\mathrm{Z = 0.001 \: Z_{\odot}}$"
    )

    ax.plot(time_myr, luminosity_a235, label=legend, color="grey", lw=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathrm{Age \: (Myr)}$")
    ax.set_ylabel(r"$\mathrm{Luminosity \: \left(erg\:\:s^{-1}\:\AA^{-1}\right)}$")
    ax.set_xlim(time_myr[0], time_myr[-1])
    ax.legend()

    plt.savefig(
        os.path.expanduser(
            (
                "~/g_drive/Research/AstrophysicsSimulation/sci_plots/final/"
                "starburst_99_model.png"
            )
        ),
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.05,
    )
