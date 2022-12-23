import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.optimize import curve_fit
import matplotlib.lines as mlines

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

#     f_s = (
#         2.0e-2
#         / 5.0
#         * (met / 1e-3) ** 0.5
#         * (mass / 1.0e4) ** 0.4
#         * (n_H / n_crit + 1.0) ** (0.91)
#     )
#     # f_s=4.e-3*(mass/1.e4)**0.4*(n_H/n_crit+1.0)**(0.91)
#     f_s = np.where(f_s < 0.9, f_s, 0.9)
#     return f_s


# z, ra, mass, n_H, met = read_cat("../sim_log_files/fs07_refine/logSFC")

# fig, axs = plt.subplots(1, 1)

# f_s = f_star(n_H, mass, met)
# mass1 = np.where(f_s > 0.20, mass, 1e2)
# h0, bins = np.histogram(np.log10(mass), bins=15, range=[3, 5])
# # h1, bins1 = np.histogram(np.log10(mass1), bins=25, range=[3, 5])
# dlnM = np.log10(10 ** bins[1]) - np.log10(10 ** bins[0])
# # dlnM1 = np.log(10 ** bins1[1]) - np.log(10 ** bins1[0])
# binc = (bins[1:] + bins[:-1]) / 2.0
# # binc1 = (bins1[1:] + bins1[:-1]) / 2.0

# axs.step(binc, (h0 / dlnM), where="mid", label="0.70")
# # axs.step(binc1,(h1/dlnM1),where='mid',linestyle='dashed',label='bound GC')
# axs.set(xlabel="MC mass [M/M$_\odot$]", ylabel="dn/dlogM")
# # plt.legend()

# z, ra, mass, n_H, met = read_cat("../sim_log_files/fs035_ms10/logSFC")

# f_s = f_star(n_H, mass, met)
# mass1 = np.where(f_s > 0.20, mass, 1e2)
# h0, bins = np.histogram(np.log10(mass), bins=15, range=[3, 5])
# # h1, bins1 = np.histogram(np.log10(mass1), bins=25, range=[3, 5])
# dlnM = np.log10(10 ** bins[1]) - np.log10(10 ** bins[0])
# # dlnM1 = np.log(10 ** bins1[1]) - np.log(10 ** bins1[0])
# binc = (bins[1:] + bins[:-1]) / 2.0
# # binc1 = (bins1[1:] + bins1[:-1]) / 2.0

# axs.step(binc, (h0 / dlnM), where="mid", label="0.35")
# plt.legend()

# fig, axs = plt.subplots(1, 1)

# z, ra, mass, n_H, met = read_cat("../sim_log_files/fs07_refine/logSFC")

# f_s = f_star(n_H, mass, met)
# met1 = np.where(f_s > 0.20, met, 1e-5)
# h0, bins = np.histogram(np.log10(met), bins=15, range=[-4, -2])
# h1, bins1 = np.histogram(np.log10(met1), bins=25, range=[-4, -2])
# dlnM = np.log(10 ** bins[1]) - np.log(10 ** bins[0])
# dlnM1 = np.log(10 ** bins1[1]) - np.log(10 ** bins1[0])
# binc = (bins[1:] + bins[:-1]) / 2.0
# binc1 = (bins1[1:] + bins1[:-1]) / 2.0

# axs.step(binc, (h0 / dlnM), where="mid", label="0.7")
# # axs.step(binc1,(h1/dlnM1),where='mid',linestyle='dashed',label='bound GC')
# axs.set(xlabel="metallicity [Z/Z$_\odot$]", ylabel="dn/dlogZ")

# z, ra, mass, n_H, met = read_cat("../sim_log_files/fs035_ms10/logSFC")
# f_s = f_star(n_H, mass, met)
# met1 = np.where(f_s > 0.20, met, 1e-5)
# h0, bins = np.histogram(np.log10(met), bins=15, range=[-4, -2])
# h1, bins1 = np.histogram(np.log10(met1), bins=25, range=[-4, -2])
# dlnM = np.log(10 ** bins[1]) - np.log(10 ** bins[0])
# dlnM1 = np.log(10 ** bins1[1]) - np.log(10 ** bins1[0])
# binc = (bins[1:] + bins[:-1]) / 2.0
# binc1 = (bins1[1:] + bins1[:-1]) / 2.0

# axs.step(binc, (h0 / dlnM), where="mid", label="0.35")
# plt.legend()

# fig, axs = plt.subplots(1, 1)
# met = np.array([1e-5, 1e-4, 1e-3, 1.0 / 40, 0.1, 1])
# f_s = [1e-2, 1e-2, 3e-2, 3.3e-2, 3.4e-2, 0.137]
# f_st = 0.03 * (1.0 + (3.0 * met) ** 1.3)
# axs.loglog(met, f_s, "o-")
# axs.loglog(met, f_st, "-")
# axs.set(xlabel="metallicity [Z/Z$_\odot$]", ylabel="f_*")

# plt.show()
#%%


"""
initial mass functions and metaliccity function for the molecular clouds or
star forming clouds.
"""
latest_redshift = 8.0

fs070_log_sfc = np.loadtxt("../sim_log_files/fs07_refine/logSFC")
redshft_fs070 = fs070_log_sfc[:, 2]
mask = redshft_fs070 > latest_redshift
redshft_fs070 = redshft_fs070[mask]
r_pc_cloud_fs070 = fs070_log_sfc[:, 4][mask]
m_sun_cloud_fs070 = fs070_log_sfc[:, 5][mask]
n_hydrogen_fs070 = fs070_log_sfc[:, 8][mask]
metal_cloud_fs070 = fs070_log_sfc[:, 9][mask]

fs035_log_sfc = np.loadtxt("../sim_log_files/fs035_ms10/logSFC")
redshft_fs035 = fs035_log_sfc[:, 2]
mask = redshft_fs035 > latest_redshift
redshft_fs035 = redshft_fs035[mask]
r_pc_cloud_fs035 = fs035_log_sfc[:, 4][mask]
m_sun_cloud_fs035 = fs035_log_sfc[:, 5][mask]
n_hydrogen_fs035 = fs035_log_sfc[:, 8][mask]
metal_cloud_fs035 = fs035_log_sfc[:, 9][mask]

print("Total mass in MCs for 70%", np.sum(m_sun_cloud_fs070))
print("Total mass in MCs for 35%", np.sum(m_sun_cloud_fs035))


def gauss(x, amp, mean, sigma):
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def bimodal(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
    return amp1 * np.exp(-0.5 * ((x - mean1) / sigma1) ** 2) + amp2 * np.exp(
        -0.5 * ((x - mean2) / sigma2) ** 2
    )


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


cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))

fs70_color = cmap[1]
fs35_color = cmap[2]
with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 12,
    }
):
    x_range = (3e2, 4e4)  # (10, 5e5)
    bns = 15
    metal_xrange = (1.5e-4, 1e-2)
    # mass function
    fs70_mass, fs70_counts = log_data_function(m_sun_cloud_fs070, bns, x_range)
    fs35_mass, fs35_counts = log_data_function(m_sun_cloud_fs035, bns, x_range)

    f7_fit_params, _ = curve_fit(
        f=gauss, xdata=np.nan_to_num(np.log10(fs70_mass), neginf=0), ydata=fs70_counts
    )
    f7_theory_x = np.log10(np.geomspace(fs70_mass.min(), fs70_mass.max(), 100))
    f7_theory_y = gauss(f7_theory_x, *f7_fit_params)

    f3_fit_params, _ = curve_fit(
        f=gauss, xdata=np.nan_to_num(np.log10(fs35_mass), neginf=0), ydata=fs35_counts
    )
    f3_theory_x = np.log10(np.geomspace(fs35_mass.min(), fs35_mass.max(), 100))
    f3_theory_y = gauss(f3_theory_x, *f3_fit_params)

    fig, ax = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(10, 3),
        dpi=500,
    )

    # f70_leg = mlines.Line2D(
    #     [], [], color=fs70_color, ls="-", lw=4, label="$f_{*} = 0.70$"
    # )
    # f35_leg = mlines.Line2D(
    #     [], [], color=fs35_color, ls="-", lw=4, label="$f_{*} =0.35$"
    # )
    # leg_title = mlines.Line2D(
    #     [], [], color="white", ls="", label="$\mathrm{SFE} \: (f_{*})$"
    # )
    # leg_title = mlines.Line2D(
    #     [],
    #     [],
    #     color="white",
    #     ls="",
    #     label="$\mathrm{{z = {:.2f}}}$".format(
    #         np.min(np.concatenate([redshft_fs070, redshft_fs035]))
    #     ),
    # )
    # leg = fig.legend(
    #     title="$\mathrm{{z = {:.2f}}}$".format(
    #         np.min(np.concatenate([redshft_fs070, redshft_fs035]))
    #     ),
    #     loc="upper left",
    #     handles=[f70_leg, f35_leg],
    #     bbox_to_anchor=(0.40, 0.89),
    #     ncol=1,
    #     # edgecolor="grey",
    #     # fontsize=10,
    # )
    # leg.get_frame().set_boxstyle("Square")

    # 35% efficiency
    ax[0].plot(
        fs35_mass,
        fs35_counts,
        drawstyle="steps-mid",
        linewidth=2.5,
        alpha=0.8,
        color=fs35_color,
    )
    ax[0].fill_between(
        fs35_mass,
        fs35_counts,
        step="mid",
        alpha=0.4,
        color=fs35_color,
    )

    # 70% efficiency
    ax[0].plot(
        fs70_mass,
        fs70_counts,
        drawstyle="steps-mid",
        linewidth=2.5,
        alpha=0.8,
        color=fs70_color,
    )
    ax[0].fill_between(
        fs70_mass,
        fs70_counts,
        step="mid",
        alpha=0.4,
        color=fs70_color,
    )

    # plot the fits
    ax[0].plot(
        10**f3_theory_x,
        f3_theory_y,
        ls=":",
        linewidth=2,
        alpha=1,
        color="grey",
        label=(r"$ ({:.1f}, {:.1f})$").format(
            f3_fit_params[1], np.abs(f3_fit_params[2])
        ),
    )
    ax[0].plot(
        10**f7_theory_x,
        f7_theory_y,
        ls=":",
        linewidth=2,
        alpha=1,
        color="k",
        label=(r"$ ({:.1f}, {:.1f})$").format(
            f7_fit_params[1], np.abs(f7_fit_params[2])
        ),
    )

    ax[0].set_xlabel(
        r"$  \mathrm{M_{MC}} \:\:  \left( \mathrm{M}_{\odot} \right) $",
    )
    ax[0].set_ylabel(
        r"$\mathrm{dN / d\log} \:\: \left(\mathrm{M_{MC}}/\mathrm{M}_{\odot}\right )$",
        labelpad=2,
    )

    ax[0].set_xlim(x_range[0], x_range[1])
    ax[0].set_ylim(1, 4e4)

    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].legend(
        title=r"$\log_{{10}}\:(\mu,\:\Sigma)$",
        loc="upper left",
        # ncol=2,
        fontsize=10,
        # title_fontsize=10,
    )

    #!!! metalicitty function

    fs70_z, fs70_z_counts = log_data_function(metal_cloud_fs070, bns, metal_xrange)
    fs35_z, fs35_z_counts = log_data_function(metal_cloud_fs035, bns, metal_xrange)

    # f7_fit_params, _ = curve_fit(
    #     f=bimodal,
    #     xdata=np.log10(fs70_z),
    #     ydata=np.nan_to_num(fs70_z_counts, neginf=0),
    # )

    # f7_theory_x = np.log10(np.geomspace(fs70_z.min(), fs70_z.max(), 100))
    # f7_theory_y = bimodal(f7_theory_x, *f7_fit_params)

    # f3_fit_params, _ = curve_fit(
    #     f=bimodal,
    #     xdata=np.log10(fs35_z),
    #     ydata=np.nan_to_num(fs35_z_counts, neginf=0),
    # )

    # f3_theory_x = np.log10(np.geomspace(fs35_z.min(), fs35_z.max(), 100))
    # f3_theory_y = bimodal(f3_theory_x, *f3_fit_params)

    # 35% efficiency
    ax[1].plot(
        fs35_z,
        fs35_z_counts,
        drawstyle="steps-mid",
        linewidth=2.5,
        alpha=0.8,
        color=fs35_color,
        label=r"$f_* = 0.35$",
    )
    ax[1].fill_between(
        fs35_z,
        fs35_z_counts,
        step="mid",
        alpha=0.4,
        color=fs35_color,
    )
    # 70% efficiency
    ax[1].plot(
        fs70_z,
        fs70_z_counts,
        drawstyle="steps-mid",
        linewidth=2.5,
        alpha=0.8,
        color=fs70_color,
        label=r"$f_* = 0.70$",
    )
    ax[1].fill_between(
        fs70_z,
        fs70_z_counts,
        step="mid",
        alpha=0.4,
        color=fs70_color,
    )

    ax[1].legend(
        title=r"$\mathrm{{z = {:.2f}}}$".format(
            np.min(np.concatenate([redshft_fs070, redshft_fs035]))
        ),
        fontsize=10,
    )
    # first_bump_mu_f7 = np.min([f7_fit_params[1], f7_fit_params[4]])
    # first_bump_sig_f7 = np.max(np.abs([f7_fit_params[2], f7_fit_params[5]]))
    # second_bump_mu_f7 = np.max([f7_fit_params[1], f7_fit_params[4]])
    # second_bump_sig_f7 = np.min(np.abs([f7_fit_params[2], f7_fit_params[5]]))

    # first_bump_mu_f3 = np.min([f3_fit_params[1], f3_fit_params[4]])
    # first_bump_sig_f3 = np.max(np.abs([f3_fit_params[2], f3_fit_params[5]]))
    # second_bump_mu_f3 = np.max([f3_fit_params[1], f3_fit_params[4]])
    # second_bump_sig_f3 = np.min(np.abs([f3_fit_params[2], f3_fit_params[5]]))
    # # plot the fits

    # ax[1].plot(
    #     10**f3_theory_x,
    #     f3_theory_y,
    #     ls=":",
    #     linewidth=2,
    #     alpha=1,
    #     color="grey",
    #     label=(r"$({:.1f}, {:.1f})$" "\n" r"$({:.1f}, {:.1f})$").format(
    #         first_bump_mu_f3,
    #         first_bump_sig_f3,
    #         second_bump_mu_f3,
    #         second_bump_sig_f3,
    #     ),
    # )
    # ax[1].plot(
    #     10**f7_theory_x,
    #     f7_theory_y,
    #     ls=":",
    #     linewidth=2,
    #     alpha=1,
    #     color="black",
    #     label=(r"$({:.1f}, {:.1f})$" "\n" r"$({:.1f}, {:.1f})$").format(
    #         first_bump_mu_f7,
    #         first_bump_sig_f7,
    #         second_bump_mu_f7,
    #         second_bump_sig_f7,
    #     ),
    # )
    ax[1].set_xlabel(
        r"$\mathrm{Z_{MC}} \:\:  \left( \mathrm{Z}_{\odot} \right) $",
    )
    ax[1].set_ylabel(
        r"$\mathrm{dN / d\log} \:\: \left(\mathrm{Z_{MC}}/\mathrm{Z}_{\odot}\right )$",
        labelpad=2,
    )

    # ax[1].yaxis.tick_right()
    # ax[1].yaxis.set_label_position("right")

    ax[1].set_xlim(metal_xrange[0], metal_xrange[1] + 0.001)
    ax[1].set_ylim(1, 4e4)

    ax[1].set_xscale("log")
    ax[1].set_yscale("log")

    f7_count, bin_edges = np.histogram(
        r_pc_cloud_fs070, bins=np.arange(0.6, 3.6, 0.2), density=True
    )
    f3_count, bin_edges = np.histogram(
        r_pc_cloud_fs035, bins=np.arange(0.6, 3.6, 0.2), density=True
    )
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]
    bin_ctrs = 0.5 * (left_edges + right_edges)

    ax[2].plot(
        bin_ctrs,
        f3_count,
        drawstyle="steps-mid",
        linewidth=2.5,
        alpha=0.8,
        color=fs35_color,
        label=r"$\mu = {:.2f}$".format(np.mean(r_pc_cloud_fs035)),
    )
    ax[2].fill_between(
        bin_ctrs,
        f3_count,
        step="mid",
        alpha=0.4,
        color=fs35_color,
    )
    ax[2].plot(
        bin_ctrs,
        f7_count,
        drawstyle="steps-mid",
        linewidth=2.5,
        alpha=0.8,
        color=fs70_color,
        label=r"$\mu = {:.2f}$".format(np.mean(r_pc_cloud_fs070)),
    )
    ax[2].fill_between(
        bin_ctrs,
        f7_count,
        step="mid",
        alpha=0.4,
        color=fs70_color,
    )
    ax[2].set(
        xlabel=r"$\mathrm{R_{MC} \: (pc)}$",
        ylabel=r"$\mathrm{PDF \: (R_{MC})}$",
        ylim=(0, 1.70),
    )
    ax[2].legend(fontsize=10)
    # ax[1].legend(
    #     title=r"$\log_{{10}}\:(\mu,\:\sigma)$",
    #     loc="upper center",
    #     ncol=2,
    #     # fontsize=10,
    #     # title_fontsize=10,
    # )
    # plt.legend(
    #     title="$\mathrm{SFE} \: (f_{*})$",
    #     loc="upper left",
    # )
    ax[2].tick_params(axis="both", direction="in", which="both")
    ax[1].tick_params(axis="both", direction="in", which="both")
    ax[0].tick_params(axis="both", direction="in", which="both")

    plt.subplots_adjust(hspace=0, wspace=0.32)

    plt.savefig(
        "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/sfc_mfunc.png",
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.05,
    )
