import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from modules.macros import t_myr_from_z
from matplotlib import cm
from scipy import interpolate
import os

cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))

fs70_color = cmap[1]
fs35_color = cmap[2]

fs070_log_sfc = np.loadtxt("../sim_log_files/fs07_refine/logSFC")
redshft_fs070 = fs070_log_sfc[:, 2]
r_pc_cloud_fs070 = fs070_log_sfc[:, 4]
m_star_fs070 = fs070_log_sfc[:, 7]
m_cloud_fs070 = fs070_log_sfc[:, 5]
n_hydrogen_fs070 = fs070_log_sfc[:, 8]
metal_cloud_fs070 = fs070_log_sfc[:, 9]
t_myr_fs070 = t_myr_from_z(redshft_fs070)

fs035_log_sfc = np.loadtxt("../sim_log_files/fs035_ms10/logSFC")
redshft_fs035 = fs035_log_sfc[:, 2]
r_pc_cloud_fs035 = fs035_log_sfc[:, 4]
m_star_fs035 = fs035_log_sfc[:, 7]
m_cloud_fs035 = fs035_log_sfc[:, 5]
n_hydrogen_fs035 = fs035_log_sfc[:, 8]
metal_cloud_fs035 = fs035_log_sfc[:, 9]
t_myr_fs035 = t_myr_from_z(redshft_fs035)

bins = np.linspace(1, 3, 20)
temp_bins = np.linspace(2, 55, 20)


def cloud_temp(hydrogen_density, redshift):
    return (hydrogen_density / 5e4) * (100 / (1 + redshift) ** 2) * 100


f3_cloud_temp = cloud_temp(n_hydrogen_fs035, redshft_fs035)
f7_cloud_temp = cloud_temp(n_hydrogen_fs070, redshft_fs070)
with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 12,
    }
):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 2.5), dpi=300)
    plt.subplots_adjust(hspace=0, wspace=0.25)

    ax[0].hist(
        r_pc_cloud_fs035,
        bins=bins,
        color=fs35_color,
        histtype="step",
        hatch="xxx",
        # edgecolor="white",
        alpha=0.7,
        linewidth=2,
        label=r"$0.35$",
        density=True,
    )
    ax[0].hist(
        r_pc_cloud_fs070,
        bins=bins,
        color=fs70_color,
        histtype="step",
        hatch="++",
        # edgecolor="white",
        alpha=0.7,
        linewidth=2,
        label=r"$0.70$",
        density=True,
    )

    ax[1].hist(
        m_cloud_fs035,
        # bins=temp_bins,
        color=fs35_color,
        histtype="step",
        hatch="xxx",
        # edgecolor="white",
        alpha=0.7,
        linewidth=2,
        label=r"$0.35$",
        density=True,
    )
    ax[1].hist(
        m_cloud_fs070,
        # bins=temp_bins,
        color=fs70_color,
        histtype="step",
        hatch="++",
        # edgecolor="white",
        alpha=0.7,
        linewidth=2,
        label=r"$0.70$",
        density=True,
    )

    ax[0].set(ylabel=r"$f(x)$", xlabel=r"$\mathrm{R_{MC}}$")
    ax[0].legend(title=r"$f_*$")
    ax[1].set(xlabel=r"$\mathrm{Temperature \:(K)}$")

    plt.savefig(
        os.path.expanduser(
            (
                "~/g_drive/Research/AstrophysicsSimulation/sci_plots/final/"
                "mc_radius_temp.png"
            )
        ),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    # ax[0].plot(bin_ctrs,f7_r_cloud,  drawstyle="steps-mid",)
    # ax[0].hist(r_pc_cloud_fs035, bins=bins)

# ax[0].scatter(t_myr_fs070, m_cloud_fs070, s=1, label="70 per cent")
#     ax[0].scatter(t_myr_fs035, m_cloud_fs035, s=1, label="35 per cent")
#     ax[0].set(yscale="log", ylabel=r"MC Mass ($M_{\odot}$)", xlabel="Myr")

#     ax[0].legend()
#     # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3), dpi=300)
#     ax[1].hist(t_myr_fs070, alpha=0.5)
#     ax[1].hist(t_myr_fs035, alpha=0.5)
#     ax[1].set(ylabel=r"Number of MCs Meeting Threshold", xlabel="Myr")
# # ax[1].set(yscale="log")
