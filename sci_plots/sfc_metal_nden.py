import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.lines as mlines
from matplotlib import cm
import matplotlib
from matplotlib.ticker import MaxNLocator

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
#             t0 = float(line.split()[2])  # zred
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


# def star_formation_efficiency(n_h, mass, metallicity):
#     """
#     star formation efficiency formula
#     """
#     # n_crit=n_H*0+1.e3/12.0
#     # with shell 6 times less dense. At t_relax 12 times less dense6
#     # f_s reduced by 5 for low met.
#     # f_s increases with stronger B-field
#     n_crit = n_h * 0 + 1.0e3 / (4.0 * 2)

#     f_s = (
#         2.0e-2
#         / 5.0
#         * (metallicity / 1e-3) ** 0.5
#         * (mass / 1.0e4) ** 0.4
#         * (n_h / n_crit + 1.0) ** (0.91)
#     )
#     # f_s=4.e-3*(mass/1.e4)**0.4*(n_H/n_crit+1.0)**(0.91)
#     f_s = np.where(f_s < 0.9, f_s, 0.9)
#     return f_s


# z, ra, mass, n_H, met = read_cat("../sim_log_files/fs07_refine/logSFC")

# fig, axs = plt.subplots(1, 1)
# plt.scatter(met, (n_H), c=z)
# plt.colorbar(label="redshift of formation")
# axs.set(
#     xlabel="metallicity [Z/Z$_\odot$]",
#     ylabel="mean gas number density [cm$^{-3}]$",
#     xlim=[1e-4, 5e-3],
#     ylim=[5e3, 5e4],
#     xscale="log",
#     yscale="log",
# )

# # z, ra, mass, n_H, met = read_cat("../sim_log_files/fs035_ms10/logSFC")
# # plt.scatter(met, (n_H), c=z, marker="s")

# plt.show()
#%% clean up
"""
Graph for plotting the mean gas number density as a function of metallicity 
for a given redshift formation time
"""

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
cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))
fs70_color = cmap[1]
fs35_color = cmap[2]

f7_mtot_metal = np.sum(metal_zun_cloud_fs070 * m_sun_cloud_fs070)
f3_mtot_metal = np.sum(metal_zun_cloud_fs035 * m_sun_cloud_fs035)

print("Total mass in metals locked in MCs for 70%", f7_mtot_metal)
print("Total mass in metals locked in MCs for 35%", f3_mtot_metal)
print(f3_mtot_metal / f7_mtot_metal)
#%%
with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
):

    fig, ax = plt.subplots(1, 1, figsize=(3.8, 4.20), dpi=400)
    cmap = plt.cm.get_cmap("autumn_r")

    im2 = ax.scatter(
        metal_zun_cloud_fs070,
        n_hydrogen_fs070,
        c=redshft_fs070,
        label=r"0.70",
        cmap=cmap,
        marker="o",
        edgecolors="black",
        linewidth=0.5,
        s=40,
        alpha=0.8,
    )
    ax.scatter(
        metal_zun_cloud_fs035,
        n_hydrogen_fs035,
        c=redshft_fs035,
        label=r"0.35",
        cmap=cmap,
        marker="P",
        edgecolors="black",
        linewidth=0.5,
        s=40,
        alpha=0.8,
    )
    # cbar = plt.colorbar(pad=0)
    # cbar.ax.invert_yaxis()
    # cbar.set_label(label="$\mathrm{z}_{\mathrm{formation}}$", fontsize=16)

    ax_divider = make_axes_locatable(ax)

    cax2 = ax_divider.append_axes("top", size="5%", pad="2%")
    cb2 = fig.colorbar(im2, cax=cax2, orientation="horizontal")
    cb2.set_label(label="$\mathrm{z}_{\mathrm{formation}}$", fontsize=16, labelpad=6)
    cb2.ax.tick_params(axis="x", direction="in", which="both")
    cb2.ax.xaxis.set_ticks_position("top")
    cb2.ax.xaxis.set_label_position("top")
    cb2.ax.invert_xaxis()
    # cb2.ax.locator_params(nbins=9)
    # cb2.set_ticks([9, 10, 11, 12])
    ax.set_xlabel(r"$ \mathrm{Z_{MC}\:}(\mathrm{Z}_{\odot})$", fontsize=12)
    ax.set_ylabel(
        r"$\overline{n_\mathrm{H}} \: \left( \mathrm{cm} ^{-3} \right)$",
        fontsize=12,
    )
    # ax.set_ylabel(
    #     r"$\overline{n_\mathrm{H}} \: \left( \mathrm{cm} ^{-3} \right)$",
    # )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 8e-3)
    ax.set_ylim(2e3, 5e4)
    # manual legend, want to customize colors
    f70 = mlines.Line2D([], [], color="k", marker="o", ls="", label=r"$f_{*} = 0.70$")
    f35 = mlines.Line2D([], [], color="k", marker="P", ls="", label=r"$f_{*} = 0.35$")
    ax.legend(
        # title="$\mathrm{SFE} \: (f_{*})$",
        loc="upper right",
        title_fontsize=12,
        fontsize=12,
        handles=[f35, f70],
    )
    ax.tick_params(axis="y", direction="in", which="both")
    ax.tick_params(axis="x", direction="in", which="both")

    with plt.rc_context(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    ):

        hist_ax = ax.inset_axes([1.02, 0, 0.30, 1], sharey=ax)

        num_bins = 30
        bins = np.geomspace(2e3, 5e4, num_bins)
        hist_ax.hist(
            n_hydrogen_fs035,
            bins=bins,
            color=fs35_color,
            histtype="step",
            hatch="////",
            edgecolor=fs35_color,
            alpha=0.9,
            linewidth=1.25,
            label=r"$0.35$",
            density=True,
            orientation="horizontal",
        )

        hist_ax.hist(
            n_hydrogen_fs070,
            bins=bins,
            color=fs70_color,
            histtype="step",
            hatch="\\\\\\\\",
            edgecolor=fs70_color,
            alpha=0.9,
            linewidth=1.25,
            label=r"$0.70$",
            density=True,
            orientation="horizontal",
        )
        hist_ax.legend(loc="upper center")
        hist_ax.set_xlabel(
            r"$\mathrm{PDF (\overline{n_\mathrm{H}})}$", fontsize=12, labelpad=6
        )

        hist_ax.tick_params(axis="y", labelleft=False)

        # hist_ax.tick_params(axis="y", direction="in", which="both")
        hist_ax.tick_params(axis="x", direction="in", which="both")
    plt.savefig(
        "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/lowres/sfc_metal_nden.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    # plt.savefig(
    #     "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/sfc_metal_nden.png",
    #     dpi=400,
    #     bbox_inches="tight",
    #     pad_inches=0.05,
    # )
