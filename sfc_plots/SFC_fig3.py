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
h1, bins1 = np.histogram(np.log10(mass1), bins=25, range=[3, 5])
dlnM = np.log(10 ** bins[1]) - np.log(10 ** bins[0])
dlnM1 = np.log(10 ** bins1[1]) - np.log(10 ** bins1[0])
binc = (bins[1:] + bins[:-1]) / 2.0
binc1 = (bins1[1:] + bins1[:-1]) / 2.0

axs.step(binc, (h0 / dlnM), where="mid", label="0.70")
# axs.step(binc1,(h1/dlnM1),where='mid',linestyle='dashed',label='bound GC')
axs.set(xlabel="MC mass [M/M$_\odot$]", ylabel="dn/dlogM")
# plt.legend()

z, ra, mass, n_H, met = read_cat("../sim_log_files/fs035_ms10/logSFC")

f_s = f_star(n_H, mass, met)
mass1 = np.where(f_s > 0.20, mass, 1e2)
h0, bins = np.histogram(np.log10(mass), bins=15, range=[3, 5])
h1, bins1 = np.histogram(np.log10(mass1), bins=25, range=[3, 5])
dlnM = np.log(10 ** bins[1]) - np.log(10 ** bins[0])
dlnM1 = np.log(10 ** bins1[1]) - np.log(10 ** bins1[0])
binc = (bins[1:] + bins[:-1]) / 2.0
binc1 = (bins1[1:] + bins1[:-1]) / 2.0

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
