import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from modules.macros import t_myr_from_z
from matplotlib import cm

#%%


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


# def read_cat1(filen):
#     t = []
#     x = []
#     y = []
#     V = []
#     I = []

#     f = open(filen, "r")
#     i = 0
#     for line in f:
#         if i > 0:
#             t0 = float(line.split()[0])
#             x1 = float(line.split()[1])
#             y1 = float(line.split()[2])
#             V1 = float(line.split()[3])
#             I1 = float(line.split()[4])
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


# def read_cat2(filen):
#     t = []
#     x = []
#     y = []
#     V = []
#     I = []
#     F = []

#     f = open(filen, "r")
#     i = 0
#     for line in f:
#         if i > 0:
#             t0 = float(line.split()[2])
#             x1 = float(line.split()[3])
#             y1 = float(line.split()[9])
#             V1 = float(line.split()[10])
#             I1 = float(line.split()[11])
#             F1 = float(line.split()[12])
#             t.append(t0)
#             x.append(x1)
#             y.append(y1)
#             V.append(V1)
#             I.append(I1)
#             F.append(F1)
#         i = i + 1

#     print("total lines", i)
#     t = np.array(t)
#     x = np.array(x)
#     y = np.array(y)
#     V = np.array(V)
#     I = np.array(I)
#     F = np.array(F)
#     return t, x, y, V, I, F


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


# # rough estimate, off by 10 Myr
# def time(z):
#     t0 = 480.0
#     return t0 * ((1 + z) / 11.0) ** -1.5


# def moving_average(x, w):
#     return np.convolve(x, np.ones(w), "valid") / w


# z, ra, mass, n_H, met = read_cat("../sim_log_files/fs07_refine/logSFC")

# fig, axs = plt.subplots(2, 1)

# t = time(z)
# print(time(13.25))
# ti = np.linspace(325, 615.6, 800)
# massc = np.cumsum(mass)
# massci = np.interp(ti, t, massc)

# dt = (t[1:] - t[:-1]) * 1e6
# dti = (ti[1:] - ti[:-1]) * 1e6
# sfr = (massc[1:] - massc[:-1]) / dt
# sfri = (massci[1:] - massci[:-1]) / dti
# # plt.plot(t[1:],sfr)
# axs[0].plot(ti[1:], sfri)
# # plt.plot(t,massc)
# axs[1].plot(ti, massci, label="0.7")


# axs[0].set(
#     xlabel="",
#     ylabel="SFR [M$_\odot$/yr]",
#     xlim=[325, 615.6],
#     ylim=[0, 0.125],
#     xscale="linear",
#     yscale="linear",
# )
# ax2 = axs[0].twiny()
# ax2.set(xlabel="Redshift", xlim=[13.25, 9.62], xscale="linear")


# z, ra, mass, n_H, met = read_cat("../sim_log_files/fs035_ms10/logSFC")
# t = time(z)
# ti = np.linspace(325, 552, 400)
# massc = np.cumsum(mass)
# massci = np.interp(ti, t, massc)
# dti = (ti[1:] - ti[:-1]) * 1e6
# sfri = (massci[1:] - massci[:-1]) / dti


# axs[0].plot(ti[1:], sfri)

# axs[1].plot(ti, massci, label="0.35")

# axs[1].set(
#     xlabel="time [Myr]",
#     ylabel="M$_*$ [M$_\odot$]",
#     xlim=[325, 615.6],
#     ylim=[1.0e4, 5e6],
#     xscale="linear",
#     yscale="log",
# )


# # [redshift, current_time, dm, pop2, pop3, sn, dead, BH, sfc, psc,gas mass]
# # z, tim, mass_dm, mass_p2, mass_p3 = read_cat1("mtot_0-100.txt")
# # axs[1].plot(tim, mass_p2)
# # axs[1].plot(tim, mass_p3)

# # # next file
# # # step, zred, tUniv[Myr], Mvir[Msun], Rvir[kpc], pos[
# # #     code_length
# # # ], Mdm, Mpop2, Mpop3, Mgas, Nbh
# # tim, mass_dm, mass_p2, mass_p3, mass_g, Nbh = read_cat2("halo1_history.f05.dat")
# # axs[1].plot(tim, mass_p2, label="f=0.5")
# # axs[1].plot(tim, Nbh * 120.0)
# # axs[1].plot(tim, mass_dm)
# # # axs[1].plot(tim,mass_g)

# # tim, mass_dm, mass_p2, mass_p3, mass_g, Nbh = read_cat2("halo1_history.f01.dat")
# # axs[1].plot(tim, mass_p2, label="f=0.1")

# # tim, mass_dm, mass_p2, mass_p3, mass_g, Nbh = read_cat2("halo1_historyHe19.dat")
# # axs[1].plot(tim, mass_p2, label="He+19")
# axs[1].legend()

# plt.show()

#%%
fs070_log_sfc = np.loadtxt("../sim_log_files/fs07_refine/logSFC")
redshft_fs070 = fs070_log_sfc[:, 2]
r_pc_cloud_fs070 = fs070_log_sfc[:, 4]
m_star_fs070 = fs070_log_sfc[:, 7]
n_hydrogen_fs070 = fs070_log_sfc[:, 8]
metal_cloud_fs070 = fs070_log_sfc[:, 9]
t_myr_fs070 = t_myr_from_z(redshft_fs070)

fs035_log_sfc = np.loadtxt("../sim_log_files/fs035_ms10/logSFC")
redshft_fs035 = fs035_log_sfc[:, 2]
r_pc_cloud_fs035 = fs035_log_sfc[:, 4]
m_star_fs035 = fs035_log_sfc[:, 7]
n_hydrogen_fs035 = fs035_log_sfc[:, 8]
metal_cloud_fs035 = fs035_log_sfc[:, 9]
t_myr_fs035 = t_myr_from_z(redshft_fs035)

# plt.figure(figsize=(7, 10), dpi=300)
# ax1 = plt.subplot(2, 1, 1)
# plt.plot(t_myr_fs070, np.cumsum(m_star_fs070))
# plt.yscale("log")
# ax1_twin = ax1.twiny()
# ax1_twin.plot(redshft_fs070 ,  np.cumsum(m_star_fs070))
# ax1_twin.invert_xaxis()

with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 12,
    }
):
    cmap = cm.get_cmap("Set2")
    cmap = cmap(np.linspace(0, 1, 8))

    fs70_color = cmap[1]
    fs35_color = cmap[2]

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(5, 4.5), dpi=300)
    plt.subplots_adjust(hspace=0)
    ax[0].plot(
        t_myr_fs035,
        np.cumsum(m_star_fs035),
        label=r"$0.35$",
        color=fs35_color,
        linewidth=4,
        alpha=0.8,
    )
    ax[0].plot(
        t_myr_fs070,
        np.cumsum(m_star_fs070),
        label=r"$0.70$",
        color=fs70_color,
        linewidth=4,
        alpha=0.8,
    )
    # ax[0].set_ylim(bottom=5e3,top=2e6)
    # increase data points for 35 % efficiency
    fs035_interp_points = np.arange(t_myr_fs035.min(), t_myr_fs035.max(), 1)
    fs035_interp = np.interp(
        fs035_interp_points,
        xp=t_myr_fs035,
        fp=np.cumsum(m_star_fs035),
    )
    # ax[0].scatter(fs035_interp_points, fs035_interp, label=r"0.35", s=1)

    # increase data points for 70 % efficiency
    fs070_interp_points = np.arange(t_myr_fs070.min(), t_myr_fs070.max(), 1)
    fs070_interp = np.interp(
        fs070_interp_points,
        xp=t_myr_fs070,
        fp=np.cumsum(m_star_fs070),
    )
    # ax[0].scatter(fs070_interp_points, fs070_interp, label=r"0.70", s=1)

    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$\mathrm{M_{*, total}} \: (\mathrm{M}_{\odot})$ ", labelpad=10)
    ax[0].legend(
        title="$\mathrm{SFE} \: (f_{*})$",
        loc="lower right",
        fontsize=12,
    )

    # the limits are controlled by 0.70 efficiency
    # add a twin axis
    plt.xlim(left=np.min(t_myr_fs070), right=np.max(t_myr_fs070))
    ax1_twin = ax[0].twiny()
    ax1_twin.invert_xaxis()
    ax1_twin.set_xlim(left=np.max(redshft_fs070), right=np.min(redshft_fs070))
    ax1_twin.set(xlabel="$\mathrm{z}$")

    # plot the star formation rates, or the derivatives of the lines
    sfr_fs035 = np.gradient(fs035_interp) / 1e6
    ax[1].plot(fs035_interp_points, sfr_fs035, color=fs35_color)

    sfr_fs07 = np.gradient(fs070_interp) / 1e6
    ax[1].plot(fs070_interp_points, sfr_fs07, color=fs70_color)

    ax[2].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1].set_ylabel(
        r"$\mathrm{SFR} \: \left( \mathrm{M}_{\odot} \: \mathrm{yr}^{-1} \right)$"
    )
    ax[1].set_ylim(bottom=0)

    # do the efficiencies using kazu's data

    # dm_data = np.loadtxt("kazu_data/halo1_historyHe19.dat")
    # dm_tmyr = t_myr_from_z(dm_data[:, 1])
    # dm_mass = dm_data[:, 3]

    # fs070_dm_interp = np.interp(
    #     fs070_interp_points,
    #     xp=dm_tmyr,
    #     fp=dm_mass,
    # )
    # fs070_mask = dm_tmyr >= fs070_interp_points.min()
    # fs070_dm_mask = fs070_interp_points <= dm_tmyr.max()
    # fs070_efficiency = (
    #     fs070_interp[fs070_dm_mask] / fs070_dm_interp[fs070_dm_mask]
    # )  # * 100

    # # ax[2].scatter(dm_tmyr[fs070_mask], dm_mass[fs070_mask])
    # ax[2].plot(
    #     fs070_interp_points[fs070_dm_mask],
    #     fs070_efficiency,
    #     c=fs70_color,
    #     linewidth=4,
    #     alpha=0.8,
    # )

    # fs035_dm_interp = np.interp(
    #     fs035_interp_points,
    #     xp=dm_tmyr,
    #     fp=dm_mass,
    # )
    # fs035_mask = dm_tmyr >= fs035_interp_points.min()
    # fs035_dm_mask = fs035_interp_points <= dm_tmyr.max()
    # fs035_efficiency = (
    #     fs035_interp[fs035_dm_mask] / fs035_dm_interp[fs035_dm_mask]
    # )  # * 100

    # # ax[2].scatter(dm_tmyr[fs035_mask], dm_mass[fs035_mask])
    # ax[2].plot(
    #     fs035_interp_points[fs035_dm_mask],
    #     fs035_efficiency,
    #     c=fs35_color,
    #     linewidth=4,
    #     alpha=0.8,
    # )
    #
    # # ax[2].set_ylim(bottom=0, top=0.37)

    # ax[0].plot(fs035_interp_points[fs035_dm_mask], fs035_dm_interp[fs035_dm_mask])
    # ax[0].plot(fs070_interp_points[fs070_dm_mask], fs070_dm_interp[fs070_dm_mask])
    # year_annotations = [
    #     468.1533733654368,
    #     503.74494157253463,
    #     553.1560071977984,
    #     592.4140693990269,
    # ]
    # for t in year_annotations:
    #     ax[0].axvline(t, ls=":", color="grey")
    #     ax[1].axvline(t, ls=":", color="grey")

    # DM halo finder data
    f3_halo = np.loadtxt("./dm_data/fs035_dm_halo_evo.txt")
    f7_halo = np.loadtxt("./dm_data/fs070_dm_halo_evo.txt")

    f7_dm_mass = f7_halo[:, 3]
    f3_dm_mass = f3_halo[:, 3]

    f7_star_mass = f7_halo[:, 5]
    f3_star_mass = f3_halo[:, 5]

    ax[2].plot(
        f7_halo[:, 1],
        f7_star_mass / f7_dm_mass,
        c=fs70_color,
        linewidth=4,
        alpha=1,
        # ls=":",
    )

    ax[2].plot(
        f3_halo[:, 1],
        f3_star_mass / f3_dm_mass,
        c=fs35_color,
        linewidth=4,
        alpha=1,
        # ls=":",
    )

    ax[2].set_xlim(right=f7_halo[:, 1].max())
    ax[2].set_yscale("log")
    ax[2].set_ylabel(r"$\mathrm{M_{*} / M_{DM}}$", labelpad=10)

    plt.savefig(
        "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/sfc_sfr.png",
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.05,
    )
