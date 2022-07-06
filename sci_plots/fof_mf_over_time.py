import sys

sys.path.append("..")
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
from modules.macros import filter_snapshots, t_myr_from_z
from modules.match_t_sims import find_matching_time, get_snapshots


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

    x_range = (10, 3e5)
    bns = 15

    f7_mc_imf_clr = cmap[0]
    f7_bsc_mf_clr = cmap[1]
    f3_mc_imf_clr = cmap[2]
    f3_bsc_mf_clr = cmap[3]

    # use particle data to initiate matching of frames
    fs070 = filter_snapshots("../particle_data/pop_2_data/fs07_refine", 113, 918, 1)
    fs035 = filter_snapshots("../particle_data/pop_2_data/fs035_ms10", 154, 917, 1)
    # find matching fs = 0.35 snapshots in terms of time to fs = 0.70
    # smaller goes fist
    # in general, use the simulation with more snapshots as a lookup table and match
    # the ones with less
    _, f7_matched_nums = find_matching_time(sequence=fs035, look_up_sequence=fs070)

    fs070_dat_dir = r"../halo_data/fs07_refine/fof_best"
    fs035_dat_dir = r"../halo_data/fs035_ms10/fof_best"

    fs070_matched = get_snapshots(
        snapshot_file_list=filter_snapshots(fs070_dat_dir, 113, 918, 1),
        get_list=f7_matched_nums,
    )[::500]
    fs035_matched = filter_snapshots(fs035_dat_dir, 154, 917, 1)[::500]

    fs035_log_sfc = np.loadtxt("../sim_log_files/fs035_ms10/logSFC")
    fs070_log_sfc = np.loadtxt("../sim_log_files/fs07_refine/logSFC")

    for f7_ds, f3_ds in zip(fs070_matched, fs035_matched):
        f7_info_file = np.loadtxt(os.path.join(f7_ds, "fof_info.txt"))
        f3_info_file = np.loadtxt(os.path.join(f3_ds, "fof_info.txt"))

        try:
            f7_t_myr = f7_info_file[0, 0]

            f7_redshift = f7_info_file[0, 1]

            f7_masses_per_snapshot = f7_info_file[:, 3]

        except:  # if there is only once cluster
            f7_t_myr = f7_info_file[0]

            f7_redshift = f7_info_file[1]

            f7_masses_per_snapshot = f7_info_file[3]

        try:
            f3_t_myr = f3_info_file[0, 0]
            f3_redshift = f3_info_file[0, 1]
            f3_masses_per_snapshot = f3_info_file[:, 3]
        except:
            f3_t_myr = f3_info_file[0]
            f3_redshift = f3_info_file[1]
            f3_masses_per_snapshot = f3_info_file[3]

        # print(f7_t_myr, f3_t_myr)
        # print(f7_redshift, f3_redshift)

        # gc_ages_per_snapshot = info_file[:, 1]
        # gc_lums_per_snapshot = info_file[:, 3]

        # from the fof profiler

        # from the logSFC
        f3_birth_z = fs035_log_sfc[:, 2]
        f3_mc_mask = fs035_log_sfc[:, 2] >= f3_redshift
        f3_mc_star_mass = fs035_log_sfc[:, 7][f3_mc_mask]

        f7_birth_z = fs070_log_sfc[:, 2]
        f7_mc_mask = fs070_log_sfc[:, 2] >= f7_redshift
        f7_mc_star_mass = fs070_log_sfc[:, 7][f7_mc_mask]

        mc_f7_mass, mc_f7_counts = log_data_function(f7_mc_star_mass, bns, x_range)
        mc_f3_mass, mc_f3_counts = log_data_function(f3_mc_star_mass, bns, x_range)

        f7_vir_mass, f7_vir_counts = log_data_function(
            f7_masses_per_snapshot, bns, x_range
        )

        f3_vir_mass, f3_vir_counts = log_data_function(
            f3_masses_per_snapshot, bns, x_range
        )

        with plt.rc_context(
            {
                "font.family": "serif",
                "mathtext.fontset": "cm",
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "font.size": 12,
            }
        ):
            fig, ax = plt.subplots(
                nrows=1,
                ncols=2,
                sharex=True,
                sharey=True,
                figsize=(9, 3.5),
                dpi=300,
            )

            ax[0].plot(
                mc_f7_mass,
                mc_f7_counts,
                label=r"$ \mathrm{{MC}} \: \mathrm{{IMF}}$",
                drawstyle="steps-mid",
                linewidth=4,
                alpha=0.8,
                color=f7_mc_imf_clr,
            )

            ax[0].plot(
                f7_vir_mass,
                f7_vir_counts,
                label=r"$\mathrm{{BSC \: M_{{vir}} }} $",
                drawstyle="steps-mid",
                linewidth=4,
                alpha=0.8,
                color=f7_bsc_mf_clr,
            )
            ax[0].fill_between(
                mc_f7_mass, mc_f7_counts, step="mid", alpha=0.4, color=f7_mc_imf_clr
            )
            ax[0].fill_between(
                f7_vir_mass, f7_vir_counts, step="mid", alpha=0.4, color=f7_bsc_mf_clr
            )

            ax[1].plot(
                mc_f3_mass,
                mc_f3_counts,
                label=r"$ \mathrm{{MC}} \: \mathrm{{IMF}}$",
                drawstyle="steps-mid",
                linewidth=4,
                alpha=0.8,
                color=f3_mc_imf_clr,
            )

            ax[1].plot(
                f3_vir_mass,
                f3_vir_counts,
                label=r"$\mathrm{{BSC \: M_{{vir}} }} $",
                drawstyle="steps-mid",
                linewidth=4,
                alpha=0.8,
                color=f3_bsc_mf_clr,
            )

            ax[1].fill_between(
                mc_f3_mass, mc_f3_counts, step="mid", alpha=0.4, color=f3_mc_imf_clr
            )
            ax[1].fill_between(
                f3_vir_mass, f3_vir_counts, step="mid", alpha=0.4, color=f3_bsc_mf_clr
            )

            ax[0].set_xscale("log")
            ax[0].set_yscale("log")
            # ax[0].set_xlabel(
            #     r"$  \mathrm{M} \:\:  \left( \mathrm{M}_{\odot} \right) $",
            #     fontsize=14,
            # )
            # ax[0].ylabel(
            #     (
            #         r"$\mathrm{dN / d\log}\:\mathrm{M}"
            #         r"\:\:\left(\mathrm{M}_{\odot} \right)$"
            #     ),
            #     fontsize=14,
            # )
            ax[0].legend(
                title=r"$\mathrm{SFE} \: (f_{*}) = 0.70$",
                loc="upper right",
            )
            # current simulation time annotation
            props = dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.5,
                linewidth=0.8,
                edgecolor="gray",
            )
            textstr_f7 = (
                r"$\mathrm{{t}} = {:.1f} \: \mathrm{{Myr}}$"
                "\n"
                r"$\mathrm{{z}} = {:.1f}$"
            ).format(f7_t_myr, f7_redshift)
            ax[0].text(
                0.03,
                0.96,
                textstr_f7,
                transform=ax[0].transAxes,
                verticalalignment="top",
                bbox=props,
            )

            # textstr_f3 = (
            #     r"$\mathrm{{t}} = {:.1f} \: \mathrm{{Myr}}$"
            #     "\n"
            #     r"$\mathrm{{z}} = {:.1f}$"
            # ).format(f3_t_myr, f3_redshift)
            # ax[1].text(
            #     0.03,
            #     0.96,
            #     textstr_f3,
            #     transform=ax[1].transAxes,
            #     verticalalignment="top",
            #     bbox=props,
            # )
            ax[0].set_xlim(left=f7_vir_mass[0])
            ax[0].set_ylim(1, 500)

            plt.subplots_adjust(hspace=0, wspace=0)
            plt.show()

        # bubble_plot(
        #     masses=gc_out_masses,
        #     core_radii=gc_r_core,
        #     ages=t_myr - gc_char_age,
        #     current_time=t_myr,
        # )

    # mass_function(masses=gc_out_masses, t_sim=t_myr, num_bins=10, m_core=gc_m_core)

    # except Exception as e:
    #     print(e)
    #     print("> Missing info file:", data_file)
    #     pass

    # if not os.path.exists(folder_name):
    #     print("# Creating new sequence directory", folder_name)
    #     os.makedirs(folder_name)

    # fig.savefig(folder_name + "/scatter.png", dpi=300)
