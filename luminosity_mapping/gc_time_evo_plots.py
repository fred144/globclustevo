import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# =============================================================================
# import matplotlib.pyplot as plt
# import numpy as np
#
# def read_cat(filen):
#    t=[]
#    x=[]
#    y=[]
#    V=[]
#    I=[]
#
#    f=open(filen, 'r')
#    i=0
#    for line in f:
#      if (i >0):
#         t0=float(line.split()[2])
#         x1=float(line.split()[4])
#         y1=float(line.split()[5])
#         V1=float(line.split()[8])
#         I1=float(line.split()[9])
#         t.append(t0)
#         x.append(x1)
#         y.append(y1)
#         V.append(V1)
#         I.append(I1)
#      i=i+1
#
#    print('total lines',i)
#    t=np.array(t)
#    x=np.array(x)
#    y=np.array(y)
#    V=np.array(V)
#    I=np.array(I)
#    return t,x,y,V,I
#
# def f_star(n_H,mass,met):
# 	#n_crit=n_H*0+1.e2/1.8
# 	n_crit=n_H*0+1.e2/1.3
# 	f_s=4.e-3*(met/1e-3)**0.5*(mass/1.e4)**0.4*(n_H/n_crit+1.0)**(0.91)
# 	f_s=np.where(f_s<0.9,f_s,0.9)
# 	return f_s
#
# z,ra,mass,n_H,met=read_cat('logSFC.txt')
#
# fig, axs= plt.subplots(2, 2)
#
# axs[0,0].loglog(met,n_H,'o')
# axs[0,0].set(xlabel='met',ylabel='$n_H$')
#
# f_s=f_star(n_H,mass,met)
#
# axs[0,1].loglog(mass,f_s,'o')
# axs[0,1].set(xlabel='mass',ylabel='$f_*$')
#
# axs[1,0].loglog(n_H,f_s,'o')
# axs[1,0].set(xlabel='$n_H$',ylabel='$f_*$')
#
# h0,bins=np.histogram(np.log10(mass),bins=10)
# h1,bins1=np.histogram(np.log10(mass),bins=20)
# #h1,bins=np.histogram(np.log10(mass[0:50]),bins=10)
# binc=(bins[1:]+bins[:-1])/2.0
# binc1=(bins1[1:]+bins1[:-1])/2.0
#
# axs[1,1].plot(binc,np.log10(h0),'o-')
# axs[1,1].plot(binc1,np.log10(h1),'o-')
# axs[1,1].set(xlabel='mass',ylabel='dn/dlogM')
#
# plt.show()
# =============================================================================
# plt.figure(figsize = (8,8), dpi=200)
# plt.hist(core_masses, bins=np.geomspace(core_masses.min(), core_masses.max(),10), histtype='step',  fill=False)
# plt.xscale('log')

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html


def mass_function(masses, t_sim, num_bins, m_core=None):

    bins = np.geomspace(np.min(masses), np.max(masses), num=num_bins, endpoint=True)
    count, bin_edges = np.histogram(masses, bins=bins)
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]
    bin_ctrs = 0.5 * (left_edges + right_edges)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    # ax.errorbar(bin_ctrs, count, fmt="-o")
    ax.hist(masses, bins, histtype="step", linewidth=4, alpha=0.5, label=r"$M_{trunc}$")

    if m_core is not None:

        count, bin_edges = np.histogram(m_core, bins=bins)
        right_edges = bin_edges[1:]
        left_edges = bin_edges[:-1]
        bin_ctrs = 0.5 * (left_edges + right_edges)

        # ax.errorbar(bin_ctrs, count, fmt="-o")
        ax.hist(
            m_core, bins, histtype="step", linewidth=4, alpha=0.5, label=r"$M_{core}$"
        )

    ax.set_title(r"$t_{{sim}}$ = {} Myr".format(t_sim), fontsize=18)

    ax.set_xlabel(r"$ M$ / M$_{\odot} $", fontsize=18)
    ax.set_ylabel(r"dN / d$\log \left( \frac{M}{M_{\odot}} \right)$", fontsize=18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=100)
    ax.legend(fontsize=18, loc="upper left")

    # another type
    # fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    # ax.bar(
    #     np.log10(bin_ctrs), np.log10(count), width=np.ptp(np.log10(bin_ctrs)) / num_bins
    # )
    # ax.errorbar(np.log10(bin_ctrs), np.log10(count), fmt="-ok")
    # ax.set_xlabel(r"$\log_{10} \left(  M / M_{\odot} \right)$", fontsize=18)
    # ax.set_ylabel(r"dN / d$\log_{10}  \left( \frac{M}{M_{\odot}} \right)$", fontsize=18)
    # ax.set_ylim(top=2)
    # # ax.set_yscale("log")


def bubble_plot(masses, core_radii, ages, current_time):
    core_diameter = core_radii * 2

    colors = np.random.uniform(size=masses.size)
    norm = 5
    # map to differnt sizes for better plotting
    core_diameter_per_size = (500 * core_diameter) / norm

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

    scatter = ax.scatter(
        ages,
        masses,
        c="black",
        s=core_diameter_per_size,
        cmap="Set3",
        alpha=0.2,
        linewidths=2,
    )

    # remap to actual sizes for legend
    legend_properties = dict(
        prop="sizes",
        num=[0.50, 1.0, 1.50],
        color="black",
        fmt=" {x:.2f}",
        func=lambda d: (d * norm) / 500,
    )
    legend = ax.legend(
        *scatter.legend_elements(**legend_properties),
        loc="lower left",
        title="$d_{core}$ (pc)",
        title_fontsize=18,
        fontsize=15,
    )
    plt.grid(visible=True)

    ax.set_title(r"$t_{{sim}}$ = {} Myr".format(current_time), fontsize=18)
    ax.set_ylabel(r"GC Truncation Mass ($M_{\odot}$)", fontsize=18)
    ax.set_xlabel(r"Formation Time (Myr)", fontsize=18)
    ax.set_xlim(300, 500)
    ax.set_ylim(10, 1e6)
    ax.set_yscale("log")

    # fig.close()


if __name__ == "__main__":

    data_directory = r"./gc_profiles/"
    # enable discrete selection of time range based on snapshot number

    files = sorted(os.listdir(data_directory))[-2:-1]  # [300:400:2]
    # strt_idx = [i for i, s in enumerate(files) if strt_snapshot in s][0]
    # end_idx = [i for i, s in enumerate(files) if end_snapshot in s][0]
    # filtered_files = files [strt_idx:end_idx:1]

    for file_name in files:

        try:
            data_file = data_directory + file_name
            info_file = np.loadtxt(data_file + "/info.txt")

            (
                t_myr,
                gc_labels,
                gc_char_age,
                gc_out_masses,
                gc_m_core,
                gc_r_trunc,
                gc_r_core,
                gc_err_rc,
                gc_alpha,
                gc_err_alpha,
                gc_sigma0,
                gc_err_sigma_0,
                gc_sigmabg,
                gc_err_sigma_bg,
            ) = zip(*info_file)

            gc_m_core = np.array(gc_m_core)
            gc_out_masses = np.array(gc_out_masses)
            gc_r_core = np.array(gc_r_core)
            gc_char_age = np.array(gc_char_age)
            t_myr = t_myr[0]

            bubble_plot(
                masses=gc_out_masses,
                core_radii=gc_r_core,
                ages=t_myr - gc_char_age,
                current_time=t_myr,
            )

            mass_function(
                masses=gc_out_masses, t_sim=t_myr, num_bins=10, m_core=gc_m_core
            )

        except Exception as e:
            print(e)
            print("> Missing info file:", data_file)
            pass

        # if not os.path.exists(folder_name):
        #     print("# Creating new sequence directory", folder_name)
        #     os.makedirs(folder_name)

        # fig.savefig(folder_name + "/scatter.png", dpi=300)
