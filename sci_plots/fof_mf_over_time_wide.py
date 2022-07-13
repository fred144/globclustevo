import sys

sys.path.append("..")
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
from modules.macros import filter_snapshots, t_myr_from_z
from modules.match_t_sims import find_matching_time, get_snapshots
from scipy.optimize import curve_fit


def pwr_law(x, a, coeff):
    return coeff * x**a


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

    x_range = (10, 5e5)
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
    )[200::100]
    fs035_matched = filter_snapshots(fs035_dat_dir, 154, 917, 1)[200::100]

    fs035_log_sfc = np.loadtxt("../sim_log_files/fs035_ms10/logSFC")
    fs070_log_sfc = np.loadtxt("../sim_log_files/fs07_refine/logSFC")
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
            nrows=int(len(fs070_matched) / 2),
            ncols=4,
            sharex=True,
            sharey=True,
            figsize=(10, 2.5 * int(len(fs070_matched) / 2)),
            dpi=300,
        )
        plt.subplots_adjust(hspace=0, wspace=0)
        fig.text(
            0.5,
            0.06,
            r"$\mathrm{M} \:\:  \left( \mathrm{M}_{\odot} \right) $",
            ha="center",
        )
        fig.text(
            0.06,
            0.5,
            r"$\mathrm{dN / d\log}\:\mathrm{M}\:\:\left(\mathrm{M}_{\odot} \right)$",
            va="center",
            rotation="vertical",
        )
        f7_imf_label = r"$ \mathrm{{MC}} \: \mathrm{{M_{*}}}$"
        f7_bsc_label = r"$\mathrm{{BSC \: M_{{vir}} }} $"
        f3_imf_label = r"$ \mathrm{{MC}} \: \mathrm{{M_{*}}}$"
        f3_bsc_label = r"$\mathrm{{BSC \: M_{{vir}} }} $"
        f7_legend_title = r"$f_{*} = 0.70$"
        f3_legend_title = r"$f_{*} = 0.35$"
        f70_imf = mlines.Line2D(
            [], [], color=f7_mc_imf_clr, ls="-", lw=4, label=f7_imf_label
        )
        f70_bsc = mlines.Line2D(
            [], [], color=f7_bsc_mf_clr, ls="-", lw=4, label=f7_bsc_label
        )
        f35_imf = mlines.Line2D(
            [], [], color=f3_mc_imf_clr, ls="-", lw=4, label=f3_imf_label
        )
        f35_bsc = mlines.Line2D(
            [], [], color=f3_bsc_mf_clr, ls="-", lw=4, label=f3_bsc_label
        )
        f7_leg_title = mlines.Line2D(
            [], [], color=f7_mc_imf_clr, ls="", label=f7_legend_title
        )
        f3_leg_title = mlines.Line2D(
            [], [], color=f3_mc_imf_clr, ls="", label=f3_legend_title
        )
        fig.legend(
            # title="$\mathrm{SFE} \: (f_{*})$",
            loc="lower center",
            title_fontsize=11,
            fontsize=11,
            handles=[f7_leg_title, f70_imf, f70_bsc, f3_leg_title, f35_imf, f35_bsc],
            bbox_to_anchor=(0.5, 0.88),
            ncol=2,
            edgecolor="k",
        )
        axs = ax.ravel()

    for i, (f7_ds, f3_ds) in enumerate(zip(fs070_matched, fs035_matched)):
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

        try:  # if there is only once cluster
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
        # fit
        f7_fitting_mask = f7_vir_mass >= 2e3
        f3_fitting_mask = f3_vir_mass >= 3e2

        f7_fit_params, _ = curve_fit(
            f=pwr_law,
            xdata=f7_vir_mass[f7_fitting_mask],
            ydata=f7_vir_counts[f7_fitting_mask],
            #
        )
        f7_theory_x = np.linspace(f7_vir_mass.min(), f7_vir_mass.max(), 100)
        f7_theory_y = pwr_law(f7_theory_x, *f7_fit_params)

        f3_fit_params, _ = curve_fit(
            f=pwr_law,
            xdata=f3_vir_mass[f3_fitting_mask],
            ydata=f3_vir_counts[f3_fitting_mask],
        )
        f3_theory_x = np.linspace(f3_vir_mass.min(), f3_vir_mass.max(), 100)
        f3_theory_y = pwr_law(f3_theory_x, *f3_fit_params)

        with plt.rc_context(
            {
                "font.family": "serif",
                "mathtext.fontset": "cm",
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "font.size": 12,
            }
        ):

            f7_imf_label = "_nolegend_"
            f7_bsc_label = "_nolegend_"
            f3_imf_label = "_nolegend_"
            f3_bsc_label = "_nolegend_"

            f7_legend_title = ""
            f3_legend_title = ""

            axs[2 * i].plot(
                mc_f7_mass,
                mc_f7_counts,
                label=f7_imf_label,
                drawstyle="steps-mid",
                linewidth=4,
                alpha=0.8,
                color=f7_mc_imf_clr,
            )
            axs[2 * i].plot(
                f7_vir_mass,
                f7_vir_counts,
                label=f7_bsc_label,
                drawstyle="steps-mid",
                linewidth=4,
                alpha=0.8,
                color=f7_bsc_mf_clr,
            )

            axs[2 * i + 1].plot(
                mc_f3_mass,
                mc_f3_counts,
                label=f3_imf_label,
                drawstyle="steps-mid",
                linewidth=4,
                alpha=0.8,
                color=f3_mc_imf_clr,
            )
            axs[2 * i + 1].plot(
                f3_vir_mass,
                f3_vir_counts,
                label=f3_bsc_label,
                drawstyle="steps-mid",
                linewidth=4,
                alpha=0.8,
                color=f3_bsc_mf_clr,
            )
            axs[2 * i].fill_between(
                mc_f7_mass, mc_f7_counts, step="mid", alpha=0.4, color=f7_mc_imf_clr
            )
            axs[2 * i].fill_between(
                f7_vir_mass, f7_vir_counts, step="mid", alpha=0.4, color=f7_bsc_mf_clr
            )
            axs[2 * i + 1].fill_between(
                mc_f3_mass, mc_f3_counts, step="mid", alpha=0.4, color=f3_mc_imf_clr
            )
            axs[2 * i + 1].fill_between(
                f3_vir_mass, f3_vir_counts, step="mid", alpha=0.4, color=f3_bsc_mf_clr
            )
            # plot the theoretical power laws
            axs[2 * i].plot(
                f7_theory_x,
                f7_theory_y,
                ls="--",
                linewidth=2,
                alpha=0.8,
                color="grey",
                label=r"${:.2f}$".format(f7_fit_params[0]),
            )
            axs[2 * i + 1].plot(
                f3_theory_x,
                f3_theory_y,
                ls="--",
                linewidth=2,
                alpha=0.8,
                color="grey",
                label=r"${:.2f}$".format(f3_fit_params[0]),
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
            axs[2 * i].text(
                0.03,
                0.95,
                textstr_f7,
                transform=axs[2 * i].transAxes,
                verticalalignment="top",
                bbox=props,
                fontsize=10,
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
            axs[2 * i].set_xlim(left=f7_vir_mass[0])
            axs[2 * i].set_ylim(1, 800)
            axs[2 * i].set_xscale("log")
            axs[2 * i].set_yscale("log")

            axs[2 * i].legend(
                title=f7_legend_title, loc="upper right", fontsize=10, title_fontsize=10
            )

            axs[2 * i + 1].legend(
                title=f3_legend_title, loc="upper right", fontsize=10, title_fontsize=10
            )

    plt.savefig(
        os.path.expanduser(
            (
                "~/g_drive/Research/AstrophysicsSimulation/sci_plots/final/"
                "mf_overtime_wide.png"
            )
        ),
        dpi=800,
        bbox_inches="tight",
        pad_inches=0.05,
        format="png",
    )
