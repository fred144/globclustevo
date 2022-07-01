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


if __name__ == "__main__":
    cmap = cm.get_cmap("Set2")
    cmap = cmap(np.linspace(0, 1, 8))

    x_range = (100, 1e5)
    bns = 15

    vir_clr = cmap[2]
    mc_imf_clr = cmap[3]
    data_directory = r"../halo_data/fs035_ms10/fof_best"
    data_sets = filter_snapshots(data_directory, 210, 918, 5)

    fs035_log_sfc = np.loadtxt("../sim_log_files/fs035_ms10/logSFC")

    for ds in data_sets:
        info_file = np.loadtxt(os.path.join(ds, "fof_info.txt"))
        t_myr = info_file[0, 0]
        redshift = info_file[1, 0]
        gc_ages_per_snapshot = info_file[:, 1]
        gc_masses_per_snapshot = info_file[:, 2]
        gc_lums_per_snapshot = info_file[:, 3]

        birth_z_fs035 = fs035_log_sfc[:, 2]
        birth_myr_fs035 = t_myr_from_z(birth_z_fs035)
        mc_mask = birth_myr_fs035 <= t_myr
        m_sun_cloud_fs035 = fs035_log_sfc[:, 7][mc_mask]
        fs70_mass, fs70_counts = log_data_function(m_sun_cloud_fs035, bns, x_range)

        tot_mass, tot_counts = log_data_function(gc_masses_per_snapshot, bns, x_range)

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
                label=r"$\mathrm{{MC \: M_{{*}}}} \: \mathrm{{IMF}} ({})$".format(
                    len(m_sun_cloud_fs035)
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
                label=r"$\mathrm{{FOF \:GC \: M_{{vir}} }} ({})$".format(len(tot_mass)),
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
                title=r"$\mathrm{SFE} \: (f_{*}) = 0.35$",
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
            ax.set_ylim(1, 300)

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
