import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

#%%


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


def f_star(n_H, mass, met):
    # n_crit=n_H*0+1.e3/12.0
    # with shell 6 times less dense. At t_relax 12 times less dense6
    # f_s reduced by 5 for low met.
    # f_s increases with stronger B-field
    n_crit = n_H * 0 + 1.0e3 / (4.0 * 2)

    f_s = (
        2.0e-2
        / 5.0
        * (met / 1e-3) ** 0.5
        * (mass / 1.0e4) ** 0.4
        * (n_H / n_crit + 1.0) ** (0.91)
    )
    # f_s=4.e-3*(mass/1.e4)**0.4*(n_H/n_crit+1.0)**(0.91)
    f_s = np.where(f_s < 0.9, f_s, 0.9)
    return f_s


z, ra, mass, n_H, met = read_cat("../sim_log_files/fs07_refine/logSFC")

fig, axs = plt.subplots(1, 1)

f_s = f_star(n_H, mass, met)
mass1 = np.where(f_s > 0.20, mass, 1e2)
h0, bins = np.histogram(np.log10(mass), bins=15, range=[3, 5])
# h1, bins1 = np.histogram(np.log10(mass1), bins=25, range=[3, 5])
dlnM = np.log10(10 ** bins[1]) - np.log10(10 ** bins[0])
# dlnM1 = np.log(10 ** bins1[1]) - np.log(10 ** bins1[0])
binc = (bins[1:] + bins[:-1]) / 2.0
# binc1 = (bins1[1:] + bins1[:-1]) / 2.0

axs.step(binc, (h0 / dlnM), where="mid", label="0.70")
# axs.step(binc1,(h1/dlnM1),where='mid',linestyle='dashed',label='bound GC')
axs.set(xlabel="MC mass [M/M$_\odot$]", ylabel="dn/dlogM")
# plt.legend()

z, ra, mass, n_H, met = read_cat("../sim_log_files/fs035_ms10/logSFC")

f_s = f_star(n_H, mass, met)
mass1 = np.where(f_s > 0.20, mass, 1e2)
h0, bins = np.histogram(np.log10(mass), bins=15, range=[3, 5])
# h1, bins1 = np.histogram(np.log10(mass1), bins=25, range=[3, 5])
dlnM = np.log10(10 ** bins[1]) - np.log10(10 ** bins[0])
# dlnM1 = np.log(10 ** bins1[1]) - np.log(10 ** bins1[0])
binc = (bins[1:] + bins[:-1]) / 2.0
# binc1 = (bins1[1:] + bins1[:-1]) / 2.0

axs.step(binc, (h0 / dlnM), where="mid", label="0.35")
plt.legend()

fig, axs = plt.subplots(1, 1)

z, ra, mass, n_H, met = read_cat("../sim_log_files/fs07_refine/logSFC")

f_s = f_star(n_H, mass, met)
met1 = np.where(f_s > 0.20, met, 1e-5)
h0, bins = np.histogram(np.log10(met), bins=15, range=[-4, -2])
h1, bins1 = np.histogram(np.log10(met1), bins=25, range=[-4, -2])
dlnM = np.log(10 ** bins[1]) - np.log(10 ** bins[0])
dlnM1 = np.log(10 ** bins1[1]) - np.log(10 ** bins1[0])
binc = (bins[1:] + bins[:-1]) / 2.0
binc1 = (bins1[1:] + bins1[:-1]) / 2.0

axs.step(binc, (h0 / dlnM), where="mid", label="0.7")
# axs.step(binc1,(h1/dlnM1),where='mid',linestyle='dashed',label='bound GC')
axs.set(xlabel="metallicity [Z/Z$_\odot$]", ylabel="dn/dlogZ")

z, ra, mass, n_H, met = read_cat("../sim_log_files/fs035_ms10/logSFC")
f_s = f_star(n_H, mass, met)
met1 = np.where(f_s > 0.20, met, 1e-5)
h0, bins = np.histogram(np.log10(met), bins=15, range=[-4, -2])
h1, bins1 = np.histogram(np.log10(met1), bins=25, range=[-4, -2])
dlnM = np.log(10 ** bins[1]) - np.log(10 ** bins[0])
dlnM1 = np.log(10 ** bins1[1]) - np.log(10 ** bins1[0])
binc = (bins[1:] + bins[:-1]) / 2.0
binc1 = (bins1[1:] + bins1[:-1]) / 2.0

axs.step(binc, (h0 / dlnM), where="mid", label="0.35")
plt.legend()

fig, axs = plt.subplots(1, 1)
met = np.array([1e-5, 1e-4, 1e-3, 1.0 / 40, 0.1, 1])
f_s = [1e-2, 1e-2, 3e-2, 3.3e-2, 3.4e-2, 0.137]
f_st = 0.03 * (1.0 + (3.0 * met) ** 1.3)
axs.loglog(met, f_s, "o-")
axs.loglog(met, f_st, "-")
axs.set(xlabel="metallicity [Z/Z$_\odot$]", ylabel="f_*")

plt.show()
#%%


"""
initial mass functions and metaliccity function for the molecular clouds or
star forming clouds.
"""

fs070_log_sfc = np.loadtxt("../sim_log_files/fs07_refine/logSFC")
redshft_fs070 = fs070_log_sfc[:, 2]
r_pc_cloud_fs070 = fs070_log_sfc[:, 4]
m_sun_cloud_fs070 = fs070_log_sfc[:, 5]
n_hydrogen_fs070 = fs070_log_sfc[:, 8]
metal_cloud_fs070 = fs070_log_sfc[:, 9]

fs035_log_sfc = np.loadtxt("../sim_log_files/fs035_ms10/logSFC")
redshft_fs035 = fs035_log_sfc[:, 2]
r_pc_cloud_fs035 = fs035_log_sfc[:, 4]
m_sun_cloud_fs035 = fs035_log_sfc[:, 5]
n_hydrogen_fs035 = fs035_log_sfc[:, 8]
metal_cloud_fs035 = fs035_log_sfc[:, 9]


def log_data_function(data, num_bins, bin_range: tuple):
    bin_range = np.log10(bin_range)
    log_data = np.log10(data)
    count, bin_edges = np.histogram(log_data, num_bins, bin_range)
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]
    bin_ctrs = 0.5 * (left_edges + right_edges)
    # normalize with width of the bins
    counts_per_log_solar_mass = count / (right_edges - left_edges)

    return 10**bin_ctrs, counts_per_log_solar_mass


with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
):
    # mass function
    fs70_mass, fs70_counts = log_data_function(m_sun_cloud_fs070, 20, (3e2, 8e4))
    fs35_mass, fs35_counts = log_data_function(m_sun_cloud_fs035, 20, (3e2, 8e4))

    cmap = cm.get_cmap("Set2")
    cmap = cmap(np.linspace(0, 1, 8))

    fs70_color = cmap[1]
    fs35_color = cmap[2]

    plt.figure(figsize=(12, 4), dpi=400)
    plt.subplot(1, 2, 1)
    # 35% efficiency
    plt.plot(
        fs35_mass,
        fs35_counts,
        label=r"0.35",
        drawstyle="steps-mid",
        linewidth=4,
        alpha=0.8,
        color=fs35_color,
    )
    plt.fill_between(
        fs35_mass,
        fs35_counts,
        step="mid",
        alpha=0.4,
        color=fs35_color,
    )
    # 70% efficiency
    plt.plot(
        fs70_mass,
        fs70_counts,
        label=r"0.70",
        drawstyle="steps-mid",
        linewidth=4,
        alpha=0.8,
        color=fs70_color,
    )
    plt.fill_between(
        fs70_mass,
        fs70_counts,
        step="mid",
        alpha=0.4,
        color=fs70_color,
    )
    plt.xlabel(
        r"$  \mathrm{M_{MC}} \:\:  \left( \mathrm{M}_{\odot} \right) $",
        fontsize=14,
    )
    plt.ylabel(
        r"$\mathrm{dN / d\log} \: \mathrm{M_{MC}} \:\: \left( \mathrm{M}_{\odot} \right )   $",
        fontsize=14,
    )
    plt.xlim((3e2, 8e4))
    # plt.ylim(.8, 2.75)
    plt.xscale("log")
    # plt.yscale("log")
    plt.legend(
        title="$\mathrm{SFE} \: (f_{*})$",
        loc="upper right",
        title_fontsize=14,
        fontsize=12,
    )

    # metalicitty function

    fs70_z, fs70_z_counts = log_data_function(metal_cloud_fs070, 20, (1e-4, 1e-2))
    fs35_z, fs35_z_counts = log_data_function(metal_cloud_fs035, 20, (1e-4, 1e-2))

    plt.subplot(1, 2, 2)
    # 35% efficiency
    plt.plot(
        fs35_z,
        fs35_z_counts,
        label=r"0.35",
        drawstyle="steps-mid",
        linewidth=4,
        alpha=0.8,
        color=fs35_color,
    )
    plt.fill_between(
        fs35_z,
        fs35_z_counts,
        step="mid",
        alpha=0.4,
        color=fs35_color,
    )
    # 70% efficiency
    plt.plot(
        fs70_z,
        fs70_z_counts,
        label=r"0.70",
        drawstyle="steps-mid",
        linewidth=4,
        alpha=0.8,
        color=fs70_color,
    )
    plt.fill_between(
        fs70_z,
        fs70_z_counts,
        step="mid",
        alpha=0.4,
        color=fs70_color,
    )

    plt.xlabel(
        r"$\mathrm{Z_{MC}} \:\:  \left( \mathrm{Z}_{\odot} \right) $",
        fontsize=14,
    )
    plt.ylabel(
        r"$\mathrm{dN/d\log}\:\mathrm{Z_{MC}}\:\:\left(\mathrm{Z}_{\odot}\right)$",
        fontsize=14,
    )
    plt.xlim(1e-4, 1e-2)
    # plt.ylim(.8, 2.75)
    plt.xscale("log")
    plt.yscale("log")
