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
        * (met / 5e-4) ** 0.5
        * (mass / 1.0e4) ** 0.4
        * (n_H / n_crit + 1.0) ** (0.91)
    )
    # f_s=4.e-3*(mass/1.e4)**0.4*(n_H/n_crit+1.0)**(0.91)
    f_s = np.where(f_s < 0.9, f_s, 0.9)
    return f_s


z, ra, mass, n_H, met = read_cat("../sim_log_files/fs07_refine/logSFC")

fig, axs = plt.subplots(1, 1)
plt.scatter(mass, f_star(n_H, mass, met) * 100, c=np.log10(n_H))
plt.colorbar(label="Log(n$_H$ [cm$^{-3}$])")
axs.set(
    xlabel="Stellar mass [M/M$_\odot$]",
    ylabel="Star formation efficiency %",
    xlim=[5e2, 5e4],
    ylim=[5, 100],
    xscale="log",
    yscale="log",
)

# z,ra,mass,n_H,met=read_cat('logSFC.0.3')
# plt.scatter(mass,f_star(n_H,mass,met)*100,c=np.log10(n_H),marker='s')

plt.show()
