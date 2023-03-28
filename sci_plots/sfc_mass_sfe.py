import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# def read_cat(filen):
#     t = []
#     x = []
#     y = []
#     V = []
#     I = []

#     f = open(filen, "r")
#     i = 0
#     for line in f:
#         if i > 0:
#             t0 = float(line.split()[2])
#             x1 = float(line.split()[4])
#             y1 = float(line.split()[5])
#             V1 = float(line.split()[8])
#             I1 = float(line.split()[9])
#             t.append(t0)
#             x.append(x1)
#             y.append(y1)
#             V.append(V1)
#             I.append(I1)
#         i = i + 1

#     print("total lines", i)
#     t = np.array(t)
#     x = np.array(x)
#     y = np.array(y)
#     V = np.array(V)
#     I = np.array(I)
#     return t, x, y, V, I


# def f_star(n_H, mass, met):
#     # n_crit=n_H*0+1.e3/12.0
#     # with shell 6 times less dense. At t_relax 12 times less dense6
#     # f_s reduced by 5 for low met.
#     # f_s increases with stronger B-field
#     n_crit = n_H * 0 + 1.0e3 / (4.0 * 2)

#     efficiency = (
#         2.0e-2
#         / 5.0
#         * (met / 5e-4) ** 0.5
#         * (mass / 1.0e4) ** 0.4
#         * (n_H / n_crit + 1.0) ** (0.91)
#     )
#     # f_s=4.e-3*(mass/1.e4)**0.4*(n_H/n_crit+1.0)**(0.91)
#     efficiency = np.where(efficiency < 0.9, efficiency, 0.9)
#     return efficiency


# z, ra, mass, n_H, met = read_cat("../sim_log_files/fs07_refine/logSFC")

# fig, axs = plt.subplots(1, 1)
# plt.scatter(mass, f_star(n_H, mass, met) * 100, c=np.log10(n_H))
# plt.colorbar(label="Log(n$_H$ [cm$^{-3}$])")
# axs.set(
#     xlabel="Stellar mass [M/M$_\odot$]",
#     ylabel="Star formation efficiency %",
#     xlim=[5e2, 5e4],
#     ylim=[5, 100],
#     xscale="log",
#     yscale="log",
# )

# z, ra, mass, n_H, met = read_cat("../sim_log_files/fs035_ms10/logSFC")
# plt.scatter(mass, f_star(n_H, mass, met) * 100, c=np.log10(n_H), marker="s")

# plt.show()
#%%

"""
Graph for plotting the star formation efficeicney as a function of molecular 
cloud mass for a given mean number density of hydrogen
"""


def star_formation_efficiency(n_h, mass, metallicity):
    """
    star formation efficiency formula
    """
    # n_crit=n_H*0+1.e3/12.0
    # with shell 6 times less dense. At t_relax 12 times less dense
    # f_s reduced by 5 for low met.
    # f_s increases with stronger B-field
    n_crit = 100.0  # (n_h * 0 + 1.0e3) / (4.0 * 2)

    efficiency = (
        (2.0e-2 / 5.0)
        * (mass / 1.0e4) ** 0.4
        * (n_h / n_crit + 1.0) ** (0.91)
        * (metallicity / 1e-3) ** 0.25
    )
    # f_s=4.e-3*(mass/1.e4)**0.4*(n_H/n_crit+1.0)**(0.91)
    efficiency = np.where(efficiency < 0.9, efficiency, 0.9)
    return efficiency


fs070_log_sfc = np.loadtxt("../sim_log_files/fs07_refine/logSFC")
redshft_fs070 = fs070_log_sfc[:, 2]
r_pc_cloud_fs070 = fs070_log_sfc[:, 4]
m_sun_cloud_fs070 = fs070_log_sfc[:, 5]
n_hydrogen_fs070 = fs070_log_sfc[:, 8]
metal_zun_cloud_fs070 = fs070_log_sfc[:, 9]

fs035_log_sfc = np.loadtxt("../sim_log_files/fs035_ms10/logSFC")
redshft_fs035 = fs035_log_sfc[:, 2]
r_pc_cloud_fs035 = fs035_log_sfc[:, 4]
m_sun_cloud_fs035 = fs035_log_sfc[:, 5]
n_hydrogen_fs035 = fs035_log_sfc[:, 8]
metal_zun_cloud_fs035 = fs035_log_sfc[:, 9]

with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
):
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.25), dpi=400)

    cmap = plt.cm.get_cmap("summer")
    ax.axhline(y=70, color="grey", ls="--", zorder=1)
    ax.axhline(y=35, color="grey", ls="--", zorder=1)
    ax.annotate("$70 \%$", (7e2, 75), color="grey")
    ax.annotate("$35 \%$", (7e2, 38), color="grey")

    im2 = ax.scatter(
        m_sun_cloud_fs070,
        star_formation_efficiency(
            n_hydrogen_fs070, m_sun_cloud_fs070, metal_zun_cloud_fs070
        )
        * 100,
        c=np.log10(n_hydrogen_fs070),
        label=r"$f_{*} = 0.70 $",
        cmap=cmap,
        marker="o",
        edgecolors="black",
        linewidth=0.5,
        s=40,
        alpha=0.8,
    )

    ax.scatter(
        m_sun_cloud_fs035,
        star_formation_efficiency(
            n_hydrogen_fs035, m_sun_cloud_fs035, metal_zun_cloud_fs035
        )
        * 100,
        c=np.log10(n_hydrogen_fs035),
        label=r"$f_{*} = 0.35 $",
        cmap=cmap,
        marker="P",
        edgecolors="black",
        linewidth=0.5,
        s=40,
        alpha=0.8,
    )
    # cbar = plt.colorbar(pad=0, orientation = 'horizontal')
    # cbar.set_label(
    #     label=(r"$\log_{10}\:\overline{n_\mathrm{H}}\:\left(\mathrm{cm}^{-3} \right)$"),
    #     fontsize=14,
    # )

    ax_divider = make_axes_locatable(ax)
    cax2 = ax_divider.append_axes("top", size="5%", pad="2%")
    cb2 = fig.colorbar(im2, cax=cax2, orientation="horizontal")
    cb2.set_label(
        label=(r"$\log_{10}\:\overline{n_\mathrm{H}}\:\left(\mathrm{cm}^{-3} \right)$"),
        fontsize=14,
        labelpad=6,
    )
    cb2.ax.tick_params(axis="x", direction="in", which="both")
    cb2.ax.locator_params(nbins=6)
    cb2.ax.xaxis.set_ticks_position("top")
    cb2.ax.xaxis.set_label_position("top")

    ax.set_xlabel(r"$\mathrm{M_{MC}} (\mathrm{M}_{\odot})$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{SFE}\:(\%)$", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # plt.anno(800, 75, "70")
    ax.set_xlim(5e2, 3e4)
    ax.set_ylim(5, 100)
    # manual legend, want to customize colors
    # f70 = mlines.Line2D([], [], color="k", marker="o", ls="", label=r"$0.70$")
    # f35 = mlines.Line2D([], [], color="k", marker="P", ls="", label=r"$0.35$")
    # ax.legend(
    #     title="$\mathrm{SFE} \: (f_{*})$",
    #     loc="lower right",
    #     title_fontsize=12,
    #     fontsize=12,
    #     handles=[f35, f70],
    # )

    ax.tick_params(axis="y", direction="in", which="both")
    ax.tick_params(axis="x", direction="in", which="both")
    # manual legend, want to customize colors
    f70 = mlines.Line2D([], [], color="k", marker="o", ls="", label=r"$f_{*} = 0.70 $")
    f35 = mlines.Line2D([], [], color="k", marker="P", ls="", label=r"$f_{*} = 0.35 $")
    ax.legend(
        loc="lower right",
        title_fontsize=12,
        fontsize=12,
        handles=[f35, f70],
    )

    plt.savefig(
        "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/lowres/sfc_mass_sfe.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    # plt.savefig(
    #     "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/sfc_mass_sfe.png",
    #     dpi=400,
    #     bbox_inches="tight",
    #     pad_inches=0.05,
    # )
