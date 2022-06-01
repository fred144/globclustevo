import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.lines as mlines


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
            t0 = float(line.split()[2])  # zred
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


def star_formation_efficiency(n_H, mass, metallicity):
    """
    star formation efficiency formula
    """
    # n_crit=n_H*0+1.e3/12.0
    # with shell 6 times less dense. At t_relax 12 times less dense6
    # f_s reduced by 5 for low met.
    # f_s increases with stronger B-field
    n_crit = n_H * 0 + 1.0e3 / (4.0 * 2)

    f_s = (
        2.0e-2
        / 5.0
        * (metallicity / 1e-3) ** 0.5
        * (mass / 1.0e4) ** 0.4
        * (n_H / n_crit + 1.0) ** (0.91)
    )
    # f_s=4.e-3*(mass/1.e4)**0.4*(n_H/n_crit+1.0)**(0.91)
    f_s = np.where(f_s < 0.9, f_s, 0.9)
    return f_s


z, ra, mass, n_H, met = read_cat("../sim_log_files/fs07_refine/logSFC")

fig, axs = plt.subplots(1, 1)
plt.scatter(met, (n_H), c=z)
plt.colorbar(label="redshift of formation")
axs.set(
    xlabel="metallicity [Z/Z$_\odot$]",
    ylabel="mean gas number density [cm$^{-3}]$",
    xlim=[1e-4, 5e-3],
    ylim=[5e3, 5e4],
    xscale="log",
    yscale="log",
)

# z, ra, mass, n_H, met = read_cat("../sim_log_files/fs035_ms10/logSFC")
# plt.scatter(met, (n_H), c=z, marker="s")

plt.show()
#%% clean up
"""
Graph for plotting the mean gas numbe density as a function of metallicity 
for a given redshift formation time
"""
with plt.rc_context({"font.family": "serif", "mathtext.fontset": "cm"}):

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

    plt.subplots(1, 1, figsize=(7, 6), dpi=300)
    cmap = plt.cm.get_cmap("autumn_r")

    plt.scatter(
        metal_zun_cloud_fs070,
        n_hydrogen_fs070,
        c=redshft_fs070,
        label=r"0.70",
        cmap=cmap,
        marker="o",
        edgecolors="black",
        linewidth=0.5,
        s=40,
        alpha=.8,
    )
    plt.scatter(
        metal_zun_cloud_fs035,
        n_hydrogen_fs035,
        c=redshft_fs035,
        label=r"0.35",
        cmap=cmap,
        marker="P",
        edgecolors="black",
        linewidth=0.5,
        s=40,
        alpha=.8,
    )
    cbar = plt.colorbar(pad=0)
    cbar.set_label(label="z$_{formation}$", fontsize=16)
    cbar.ax.invert_yaxis()

    plt.xlabel(r" Metallicity $(\mathrm{Z}_{\odot})$", fontsize=14)
    plt.ylabel(
        r"Mean Gas Number Density $\left( \mathrm{cm} ^{-3} \right)$", fontsize=14
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e-4, 8e-3)
    plt.ylim(0.4e4, 1e5)
    # manual legend, want to customize colors
    f70 = mlines.Line2D([], [], color="k", marker="o", ls="", label=r"0.70")
    f35 = mlines.Line2D([], [], color="k", marker="P", ls="", label=r"0.35")
    plt.legend(
        title="$\mathrm{SFR} \: (f_{*})$",
        loc="upper left",
        title_fontsize=14,
        fontsize=12,
        handles=[f70, f35],
    )
