"""
temp as a functinoi of metallicity
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import cm
import matplotlib
from matplotlib.ticker import MaxNLocator


def read_cat(filen):
    t = []
    x = []
    y = []
    V = []
    I = []

    f = open(filen, "r")
    i = 0
    for line in f:
        if i > 0:
            t0 = float(line.split()[2])
            x1 = float(line.split()[4])
            y1 = float(line.split()[5])
            V1 = float(line.split()[8])
            I1 = float(line.split()[9])
            t.append(t0)
            x.append(x1)
            y.append(y1)
            V.append(V1)
            I.append(I1)
        i = i + 1

    print("total lines", i)
    t = np.array(t)
    x = np.array(x)
    y = np.array(y)
    V = np.array(V)
    I = np.array(I)
    return t, x, y, V, I


# def f_star(n_H,mass,met):
# 	#n_crit=n_H*0+1.e3/12.0
# 	#with shell 6 times less dense. At t_relax 12 times less dense6
# 	# f_s reduced by 5 for low met.
# 	# f_s increases with stronger B-field
# 	n_crit=n_H*0+1.e3/(4.0*2)

# 	f_s=2.e-2/5.0*(met/1e-3)**0.5*(mass/1.e4)**0.4*(n_H/n_crit+1.0)**(0.91)
# 	#f_s=4.e-3*(mass/1.e4)**0.4*(n_H/n_crit+1.0)**(0.91)
# 	f_s=np.where(f_s<0.9,f_s,0.9)
# 	return f_s


def n_crit(tem, z):
    return 5.0e4 * ((1.0 + z) / 10.0) ** 2 * (tem / 100.0)


def fac(tem, z, n_H):
    return n_H / (5.0e3 * ((1.0 + z) / 10.0) ** 2 * (tem / 100.0))


def radius(n_H, tem, ra):
    m_H = 1.6e-24
    G = 6.7e-8
    cs = 1.0e6 * np.sqrt(tem / 1.0e4)
    # x=1.0/ra
    x = 1.0
    n_0 = n_H / (10.0 * x**3 + 3.0 - 3 * x)
    return 2.0 * cs / np.sqrt(2.0 * np.pi * G * m_H * n_0) / 3.0e18


def sfc_temperature(n_h, redshifts, ra):
    """
    temperature of a cloud given its hydrogen density,
    redshift of formation, current temp

    """
    # x = 1.0 / ra
    # n_0=10.*n_H/(10.*x**3+3-3*x)
    n_0 = 3.84 * n_h
    return 100.0 * (n_0 / 5.0e4 / ((1.0 + redshifts) / 10.0) ** 2)


fs070_log_sfc = np.loadtxt("../../sim_log_files/fs07_refine/logSFC")
redshft_fs070 = fs070_log_sfc[:, 2]
r_pc_cloud_fs070 = fs070_log_sfc[:, 4]
m_sun_cloud_fs070 = fs070_log_sfc[:, 5]
n_hydrogen_fs070 = fs070_log_sfc[:, 8]
metal_zun_cloud_fs070 = fs070_log_sfc[:, 9]

fs035_log_sfc = np.loadtxt("../../sim_log_files/fs035_ms10/logSFC")
redshft_fs035 = fs035_log_sfc[:, 2]
r_pc_cloud_fs035 = fs035_log_sfc[:, 4]
m_sun_cloud_fs035 = fs035_log_sfc[:, 5]
n_hydrogen_fs035 = fs035_log_sfc[:, 8]
metal_zun_cloud_fs035 = fs035_log_sfc[:, 9]

# fig, axs = plt.subplots(1, 1)
# # z, ra, mass, n_H, met = read_cat("logSFC.07.txt")
# plt.scatter(
#     metal_zun_cloud_fs070,
#     sfc_temperature(n_hydrogen_fs070, redshft_fs070, r_pc_cloud_fs070),
#     c=r_pc_cloud_fs070,
#     marker="o",
#     alpha=0.5,
# )
# # Tem(n_H,z,ra)
# # z, ra, mass, n_H, met = read_cat("logSFC.035.txt")
# plt.scatter(
#     metal_zun_cloud_fs035,
#     sfc_temperature(n_hydrogen_fs035, redshft_fs035, r_pc_cloud_fs035),
#     c=r_pc_cloud_fs035,
#     marker="x",
#     alpha=0.5,
# )

# plt.colorbar(label="Radius")
# axs.set(
#     xlabel="MC Metallicity [Z$_\odot$]",
#     ylabel="MC Temperature [K]",
#     xlim=[1e-4, 1e-2],
#     ylim=[10, 200],
#     xscale="log",
#     yscale="log",
# )
# plt.show()


cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))
fs70_color = cmap[1]
fs35_color = cmap[2]

cmap = cm.get_cmap("Set3")
cmap = cmap(np.linspace(0, 1, 11))
cvals = [0.1, 3]
colors = ["orangered", "cyan"]
norm = plt.Normalize(min(cvals), max(cvals))
tuples = list(zip(map(norm, cvals), colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
):
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(1, 1, figsize=(3.8, 4.20), dpi=400)

        # cmap = plt.cm.get_cmap("summer")
        # ax.axhline(y=70, color="grey", ls="--", zorder=1)
        # ax.axhline(y=35, color="grey", ls="--", zorder=1)
        # ax.annotate("$70 \%$", (7e2, 75), color="grey")
        # ax.annotate("$35 \%$", (7e2, 38), color="grey")
        f7_temp = sfc_temperature(n_hydrogen_fs070, redshft_fs070, r_pc_cloud_fs070)
        f3_temp = sfc_temperature(n_hydrogen_fs035, redshft_fs035, r_pc_cloud_fs035)
        im2 = ax.scatter(
            metal_zun_cloud_fs035,
            f3_temp,
            c=r_pc_cloud_fs035,
            label=r"0.35",
            cmap=cmap,
            marker="P",
            edgecolors="w",
            linewidth=0.5,
            s=40,
            alpha=0.8,
            vmax=2.8,
        )

        im2 = ax.scatter(
            metal_zun_cloud_fs070,
            f7_temp,
            c=r_pc_cloud_fs070,
            label=r"0.70",
            cmap=cmap,
            marker="o",
            edgecolors="w",
            linewidth=0.5,
            s=40,
            alpha=0.8,
            vmax=2.8,
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
            label=(r"$\mathrm{R_{MC}} (\mathrm{pc})$"), fontsize=16, labelpad=6
        )
        cb2.ax.tick_params(axis="x", direction="in", which="both")
        cb2.ax.locator_params(nbins=6)
        cb2.ax.xaxis.set_ticks_position("top")
        cb2.ax.xaxis.set_label_position("top")

        ax.set(
            xlim=[1e-4, 8e-3],
            ylim=[15, 230],
            xscale="log",
            yscale="log",
        )
        ax.set_xlabel(r"$\mathrm{Z_{MC}} (\mathrm{M}_{\odot})$", fontsize=12)
        ax.set_ylabel(r"$\mathrm{\:T_{MC}\:(K)}$", fontsize=12)
        # plt.anno(800, 75, "70")
        # ax.set_xlim(5e2, 5e4)
        # ax.set_ylim(5, 100)
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

        with plt.rc_context(
            {
                "font.family": "serif",
                "mathtext.fontset": "cm",
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
            }
        ):

            hist_ax = ax.inset_axes([1.02, 0, 0.30, 1], sharey=ax)

            num_bins = 40
            bins = np.geomspace(9, 150, num_bins)
            hist_ax.hist(
                f3_temp,
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
                f7_temp,
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
            # hist_ax.legend(loc="upper center")
            hist_ax.set_xlabel(r"$\mathrm{PDF (T_{MC})}$", fontsize=12, labelpad=6)
            hist_ax.tick_params(axis="y", labelleft=False)

            # hist_ax.tick_params(axis="y", direction="in", which="both")
            hist_ax.tick_params(axis="x", direction="in", which="both")
        plt.savefig(
            "../../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/darkmode/sfc_metal_temp.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
    # plt.savefig(
    #     "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/sfc_metal_temp.png",
    #     dpi=400,
    #     bbox_inches="tight",
    #     pad_inches=0.05,
    # )
