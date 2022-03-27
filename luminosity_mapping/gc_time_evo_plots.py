import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# plt.figure(figsize = (8,8), dpi=200)
# plt.hist(core_masses, bins=np.geomspace(core_masses.min(), core_masses.max(),10), histtype='step',  fill=False)
# plt.xscale('log')

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html


def bubble_plot(masses, core_radii, ages, current_time):
    core_diameter = core_radii * 2

    colors = np.random.uniform(size=masses.size)
    norm = 1.5
    # map to differnt sizes for better plotting
    core_diameter_per_size = (500 * core_diameter) / norm

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

    scatter = ax.scatter(
        ages,
        masses,
        c="black",
        s=core_diameter_per_size,
        cmap="Set3",
        alpha=0.6,
        linewidths=2,
    )

    # remap to actual sizes for legend
    legend_properties = dict(
        prop="sizes",
        num=[0.25,0.50,0.75], 
        color="black",
        fmt=" {x:.2f}",
        func=lambda d: (d * norm) / 500,
    )
    legend = ax.legend(
        *scatter.legend_elements(**legend_properties),
        loc="upper right",
        title="$d_{core}$ (pc)",
        title_fontsize=16,
        fontsize=15,
    )
    plt.grid(visible=True)
   
    ax.set_title(r"$t_{{sim}}$ = {} Myr".format(current_time), fontsize=16)
    ax.set_ylabel(r"Total GC Mass ($M_{\odot}$)", fontsize=16)
    ax.set_xlabel(r"Age (Myr)", fontsize=16)
    #ax.set_xlim(-45,300)
    #ax.set_ylim(10,1e7) 
    ax.set_yscale("log")
    
    # fig.close()


if __name__ == "__main__":

    data_directory = r"./gc_profiles/"
    # enable discrete selection of time range based on snapshot number
    strt_snapshot = "00200"
    end_snapshot = "00710"
    files = sorted(os.listdir(data_directory))  # [-2:-1]  [300:400:2]
    # strt_idx = [i for i, s in enumerate(files) if strt_snapshot in s][0]
    # end_idx = [i for i, s in enumerate(files) if end_snapshot in s][0]
    # filtered_files = files [strt_idx:end_idx:1]

    for file_name in files:

        try:
            data_file = data_directory + file_name
            info_file = np.loadtxt(data_file + "/info.txt")

            (
                t_myr,
                gc_char_age,
                gc_tot_masses,
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
            gc_tot_masses = np.array(gc_tot_masses)
            gc_r_core = np.array(gc_r_core)
            gc_char_age = np.array(gc_char_age)
            t_myr = t_myr[0]
            
            bubble_plot(
                masses=gc_m_core,
                core_radii=gc_r_core,
                ages=gc_char_age,
                current_time=t_myr,
            )

        except:
            print("> Missing info file:", data_file)
            pass

        # if not os.path.exists(folder_name):
        #     print("# Creating new sequence directory", folder_name)
        #     os.makedirs(folder_name)

        # fig.savefig(folder_name + "/scatter.png", dpi=300)
