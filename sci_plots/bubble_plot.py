import sys

sys.path.append("..")
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
from modules.macros import filter_snapshots, common_filter_snapshots


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
    cmap = cm.get_cmap("Set2")
    cmap = cmap(np.linspace(0, 1, 8))

    x_range = (1000, 1e5)
    bns = 15

    vir_clr = cmap[1]
    mc_imf_clr = cmap[2]
    data_directory = r"../gc_profiles/profile_runs/fs07_refine/fof_best_fb/"
    data_sets = filter_snapshots(data_directory, 243, 893, 5)

    for ds in data_sets:
        info_file = np.loadtxt(os.path.join(ds, "info.txt"))
        t_myr = info_file[0, 0]
        labels = info_file[:, 1]
        age = info_file[:, 2]
        tot_mass = info_file[:, 3]
        core_mass = info_file[:, 4]
        vir_rad = info_file[:, 5]
        core_rad = info_file[:, 6]
        err_core_rad = info_file[:, 7]
        alpha = info_file[:, 8]
        err_alpha = info_file[:, 9]
        sigma_0 = info_file[:, 10]
        err_sigma_0 = info_file[:, 11]
        sigma_bg = info_file[:, 12]
        err_sigma_bg = info_file[:, 13]
        half_light_r = info_file[:, 14]
        half_mass_r = info_file[:, 15]

        with plt.rc_context(
            {
                "font.family": "serif",
                "mathtext.fontset": "cm",
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "font.size": 14,
            }
        ):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), dpi=300)

    #     bubble_plot(
    #         masses=gc_out_masses,
    #         core_radii=gc_r_core,
    #         ages=t_myr - gc_char_age,
    #         current_time=t_myr,
    #     )

    # mass_function(masses=gc_out_masses, t_sim=t_myr, num_bins=10, m_core=gc_m_core)

    # except Exception as e:
    #     print(e)
    #     print("> Missing info file:", data_file)
    #     pass

    # if not os.path.exists(folder_name):
    #     print("# Creating new sequence directory", folder_name)
    #     os.makedirs(folder_name)

    # fig.savefig(folder_name + "/scatter.png", dpi=300)
