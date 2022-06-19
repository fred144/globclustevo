import sys

sys.path.append("..")
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
from modules.macros import filter_snapshots, common_filter_snapshots, t_myr_from_z


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


def mass_function(masses, t_sim, num_bins, m_core=None):
    """!!! Wrong calculation Plot mass function for truncation and core masses"""

    bins = np.geomspace(np.min(masses), np.max(masses), num=num_bins, endpoint=True)
    count, bin_edges = np.histogram(masses, bins=bins)
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]
    bin_ctrs = 0.5 * (left_edges + right_edges)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    # ax.errorbar(bin_ctrs, count, fmt="-o")
    ax.hist(
        masses,
        bins,
        histtype="step",
        hatch="\\",
        linewidth=4,
        alpha=1,
        label=r"$M_{trunc}$",
    )

    if m_core is not None:

        count, bin_edges = np.histogram(m_core, bins=bins)
        right_edges = bin_edges[1:]
        left_edges = bin_edges[:-1]
        bin_ctrs = 0.5 * (left_edges + right_edges)

        # ax.errorbar(bin_ctrs, count, fmt="-o")
        ax.hist(
            m_core,
            bins,
            histtype="step",
            hatch="/",
            linestyle="--",
            linewidth=4,
            alpha=1,
            label=r"$M_{core}$",
        )

    ax.set_title(r"$t_{{sim}}$ = {} Myr".format(t_sim), fontsize=18)
    ax.set_xlabel(r"$ M \left( M_{\odot} \right ) $", fontsize=18)
    ax.set_ylabel(r"dN / d$\log \left( M \right)$", fontsize=18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xlim(1, 2e5)
    ax.set_ylim(bottom=1, top=150)

    ax.legend(fontsize=18, loc="upper left")


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

    x_range = (1e3, 1e5)
    bns = 15

    vir_clr = cmap[1]
    mc_imf_clr = cmap[2]
    data_directory = r"../gc_profiles/profile_runs/fs07_refine/fof_best_v3/"
    data_sets = filter_snapshots(data_directory, 150, 918, 5)

    fs070_log_sfc = np.loadtxt("../sim_log_files/fs07_refine/logSFC")

    for ds in data_sets:
        info_file = np.loadtxt(os.path.join(ds, "info.txt"))
        t_myr = info_file[0, 0]
        labels = info_file[:, 1]
        age = info_file[:, 2]
        tot_masses = info_file[:, 3]
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

        birth_z_fs070 = fs070_log_sfc[:, 2]
        birth_myr_fs070 = t_myr_from_z(birth_z_fs070)
        mc_mask = birth_myr_fs070 <= t_myr
        m_sun_cloud_fs070 = fs070_log_sfc[:, 5][mc_mask]
        fs70_mass, fs70_counts = log_data_function(m_sun_cloud_fs070, bns, x_range)

        tot_mass, tot_counts = log_data_function(tot_masses, bns, x_range)
        cor_mass, cor_counts = log_data_function(core_mass, bns, x_range)

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

            ax.plot(
                fs70_mass,
                fs70_counts,
                label=r"$\mathrm{{MC}} \: \mathrm{{IMF}} ({})$".format(
                    len(m_sun_cloud_fs070)
                ),
                drawstyle="steps-mid",
                linewidth=4,
                alpha=0.8,
                color=mc_imf_clr,
            )
            ax.fill_between(
                fs70_mass, fs70_counts, step="mid", alpha=0.4, color=mc_imf_clr
            )

            ax.plot(
                tot_mass,
                tot_counts,
                label=r"$\mathrm{{ GC \: M_{{vir}} }} ({})$".format(len(tot_masses)),
                drawstyle="steps-mid",
                linewidth=4,
                alpha=0.8,
                color=vir_clr,
            )
            ax.fill_between(tot_mass, tot_counts, step="mid", alpha=0.4, color=vir_clr)

            # ax.plot(
            #     cor_mass,
            #     cor_counts,
            #     label=r"Core",
            #     drawstyle="steps-mid",
            #     linewidth=4,
            #     alpha=0.8,
            #     color="royalblue",
            # )
            # ax.fill_between(
            #     cor_mass, cor_counts, step="mid", color="royalblue", alpha=0.4
            # )

            ax.set_xscale("log")
            ax.set_yscale("log")
            plt.xlabel(
                r"$  \mathrm{M} \:\:  \left( \mathrm{M}_{\odot} \right) $",
                fontsize=14,
            )
            plt.ylabel(
                (
                    r"$\mathrm{dN / d\log}\:\mathrm{M}"
                    r"\:\:\left(\mathrm{M}_{\odot} \right)$"
                ),
                fontsize=14,
            )
            ax.legend(
                title=r"$\mathrm{SFE} \: (f_{*}) = 0.70$",
                loc="upper right",
                title_fontsize=14,
            )
            # current simulation time annotation
            props = dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.5,
                linewidth=0.8,
                edgecolor="gray",
            )
            textstr = r"$\mathrm{{t}} = {:.1f} \: \mathrm{{Myr}}$".format(t_myr)
            # place a text box in upper left in axes coords
            ax.text(
                0.03,
                0.96,
                textstr,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=props,
            )
            ax.set_xlim(x_range)
            ax.set_ylim(1, 250)

            plt.show()

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
