import matplotlib.pyplot as plt
import numpy as np


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


def read_cat1(filen):
    t = []
    x = []
    y = []
    V = []
    I = []

    f = open(filen, "r")
    i = 0
    for line in f:
        if i > 0:
            t0 = float(line.split()[0])
            x1 = float(line.split()[1])
            y1 = float(line.split()[2])
            V1 = float(line.split()[3])
            I1 = float(line.split()[4])
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


def read_cat2(filen):
    t = []
    x = []
    y = []
    V = []
    I = []
    F = []

    f = open(filen, "r")
    i = 0
    for line in f:
        if i > 0:
            t0 = float(line.split()[2])
            x1 = float(line.split()[3])
            y1 = float(line.split()[9])
            V1 = float(line.split()[10])
            I1 = float(line.split()[11])
            F1 = float(line.split()[12])
            t.append(t0)
            x.append(x1)
            y.append(y1)
            V.append(V1)
            I.append(I1)
            F.append(F1)
        i = i + 1

    print("total lines", i)
    t = np.array(t)
    x = np.array(x)
    y = np.array(y)
    V = np.array(V)
    I = np.array(I)
    F = np.array(F)
    return t, x, y, V, I, F


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


def time(z):
    t0 = 480.0
    return t0 * ((1 + z) / 11.0) ** -1.5


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


z, ra, mass, n_H, met = read_cat("../sim_log_files/fs07_refine/logSFC")

fig, axs = plt.subplots(2, 1)

t = time(z)
print(time(13.25))
ti = np.linspace(325, 615.6, 800)
massc = np.cumsum(mass)
massci = np.interp(ti, t, massc)

dt = (t[1:] - t[:-1]) * 1e6
dti = (ti[1:] - ti[:-1]) * 1e6
sfr = (massc[1:] - massc[:-1]) / dt
sfri = (massci[1:] - massci[:-1]) / dti
# plt.plot(t[1:],sfr)
axs[0].plot(ti[1:], sfri)
# plt.plot(t,massc)
axs[1].plot(ti, massci, label="0.7")
axs[0].set(
    xlabel="",
    ylabel="SFR [M$_\odot$/yr]",
    xlim=[325, 615.6],
    ylim=[0, 0.125],
    xscale="linear",
    yscale="linear",
)
axs[1].set(
    xlabel="time [Myr]",
    ylabel="M$_*$ [M$_\odot$]",
    xlim=[325, 615.6],
    ylim=[1.0e4, 5e6],
    xscale="linear",
    yscale="log",
)

ax2 = axs[0].twiny()
ax2.set(xlabel="Redshift", xlim=[13.25, 9.62], xscale="linear")

z, ra, mass, n_H, met = read_cat("../sim_log_files/fs035_ms10/logSFC")
t = time(z)
ti = np.linspace(325, 552, 400)
massc = np.cumsum(mass)
massci = np.interp(ti, t, massc)

dti = (ti[1:] - ti[:-1]) * 1e6
sfri = (massci[1:] - massci[:-1]) / dti
# axs[0].plot(ti[1:], sfri)
axs[1].plot(ti, massci, label="0.35")

# # [redshift, current_time, dm, pop2, pop3, sn, dead, BH, sfc, psc,gas mass]
# z, tim, mass_dm, mass_p2, mass_p3 = read_cat1("mtot_0-100.txt")
# axs[1].plot(tim, mass_p2)
# axs[1].plot(tim, mass_p3)

# # next file
# # step, zred, tUniv[Myr], Mvir[Msun], Rvir[kpc], pos[
# #     code_length
# # ], Mdm, Mpop2, Mpop3, Mgas, Nbh
# tim, mass_dm, mass_p2, mass_p3, mass_g, Nbh = read_cat2("halo1_history.f05.dat")
# axs[1].plot(tim, mass_p2, label="f=0.5")
# axs[1].plot(tim, Nbh * 120.0)
# axs[1].plot(tim, mass_dm)
# # axs[1].plot(tim,mass_g)

# tim, mass_dm, mass_p2, mass_p3, mass_g, Nbh = read_cat2("halo1_history.f01.dat")
# axs[1].plot(tim, mass_p2, label="f=0.1")

# tim, mass_dm, mass_p2, mass_p3, mass_g, Nbh = read_cat2("halo1_historyHe19.dat")
# axs[1].plot(tim, mass_p2, label="He+19")
axs[1].legend()

plt.show()
