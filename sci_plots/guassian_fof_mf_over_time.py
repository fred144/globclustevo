"""
BSC MF over time using strictly fof 
"""

import sys

sys.path.append("..")
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
from modules.macros import filter_snapshots, t_myr_from_z
import matplotlib.lines as mlines
from modules.match_t_sims import find_matching_time, get_snapshots
from scipy.optimize import curve_fit


def pwr_law(x, a, coeff):
    return coeff * x**a


def gauss(x, amp, mean, sigma):
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def lgauss(x, amp, mean, sigma):
    return np.log10(amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2))


def log_data_function(data, num_bins, bin_range: tuple, return_counts=False):
    bin_range = np.log10(bin_range)
    log_data = np.log10(data)
    count, bin_edges = np.histogram(log_data, num_bins, bin_range)
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]
    bin_ctrs = 0.5 * (left_edges + right_edges)
    # normalize with width of the bins
    counts_per_log_solar_mass = count / (right_edges - left_edges)

    if return_counts is True:
        return 10**bin_ctrs, count  # * count * 10
    else:
        return 10**bin_ctrs, counts_per_log_solar_mass


if __name__ == "__main__":

    cmap = cm.get_cmap("Set2")
    cmap = cmap(np.linspace(0, 1, 8))

    x_range = (60, 6e5)
    bns = 20

    f7_mc_imf_clr = cmap[0]
    f7_bsc_mf_clr = cmap[1]
    f3_mc_imf_clr = cmap[2]
    f3_bsc_mf_clr = cmap[3]

    # use particle data to initiate matching of frames
    fs070 = filter_snapshots("../particle_data/pop_2_data/fs07_refine", 113, 1196, 1)
    fs035 = filter_snapshots("../particle_data/pop_2_data/fs035_ms10", 154, 1316, 1)
    # find matching fs = 0.35 snapshots in terms of time to fs = 0.70
    # smaller goes fist
    # in general, use the simulation with more snapshots as a lookup table and match
    # the ones with less
    _, f3_matched_nums = find_matching_time(sequence=fs070, look_up_sequence=fs035)

    # fs070_dat_dir = r"../halo_data/fs07_refine/fof_best"
    # fs035_dat_dir = r"../halo_data/fs035_ms10/fof_best"

    profiler_data = (
        "../../g_drive/Research/AstrophysicsSimulation/DesktopEnvironment/"
        "data_globular_cluster/gc_profiles/profile_runs/"
    )

    fs070_dat_dir = profiler_data + "fs07_refine/fof_best"
    fs035_dat_dir = profiler_data + "fs035_ms10/fof_best"

    fs070_matched = filter_snapshots(fs070_dat_dir, 113, 1196, 1)

    fs035_matched = get_snapshots(
        snapshot_file_list=filter_snapshots(fs035_dat_dir, 154, 1368, 1),
        get_list=f3_matched_nums,
    )

    wanted_idxs = [300, 450, 800, 1005]
    fs070_matched = [fs070_matched[x] for x in wanted_idxs]
    fs035_matched = [fs035_matched[x] for x in wanted_idxs]

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
            nrows=len(fs070_matched),
            ncols=2,
            sharex=True,
            sharey="col",
            figsize=(5, 2.5 * len(fs070_matched)),
            dpi=300,
        )
        plt.subplots_adjust(hspace=0, wspace=0)
        fig.text(
            0.5,
            0.08,
            r"$\mathrm{M} \:\:  \left( \mathrm{M}_{\odot} \right) $",
            ha="center",
        )
        # fig.text(
        #     0.01,
        #     0.5,
        #     r"$\mathrm{dN / d\log}\:\:\left(\mathrm{M} / \mathrm{M}_{\odot} \right)$",
        #     va="center",
        #     rotation="vertical",
        # )
        fig.text(
            0.01,
            0.5,
            r"$\mathrm{N_{BSC}}$",
            va="center",
            rotation="vertical",
        )
        f7_imf_label = r"$ \mathrm{{MC}} \: \mathrm{{M_{*}}}$"
        f7_bsc_label = r"$\mathrm{{CMF }} $"
        f3_imf_label = r"$ \mathrm{{MC}} \: \mathrm{{M_{*}}}$"
        f3_bsc_label = r"$\mathrm{{CMF}} $"
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
        leg = fig.legend(
            # title="$\mathrm{SFE} \: (f_{*})$",
            loc="lower center",
            handles=[f7_leg_title, f70_imf, f70_bsc, f3_leg_title, f35_imf, f35_bsc],
            bbox_to_anchor=(0.5, 0.885),
            ncol=2,
            edgecolor="k",
        )
        leg.get_frame().set_boxstyle("Square")

    for i, (f7_ds, f3_ds) in enumerate(zip(fs070_matched, fs035_matched)):
        f7_info_file = np.loadtxt(os.path.join(f7_ds, "info.txt"))
        f3_info_file = np.loadtxt(os.path.join(f3_ds, "info.txt"))

        star_mass_min = 250  # minimum solar mass minimum oer bsc
        max_alpha = 4.2
        try:
            f7_t_myr = f7_info_file[0, 0]
            # f7_redshift = f7_info_file[0, 1]
            # f7_redshift = f7_redshift
            f7_masses_per_snapshot = f7_info_file[:, 3]
            f7_alpha = f7_info_file[:, 8]
            f7_mask = (f7_masses_per_snapshot > star_mass_min) & (f7_alpha < max_alpha)

            f7_masses_per_snapshot = f7_masses_per_snapshot[f7_mask]
        except:  # if there is only once cluster
            f7_t_myr = f7_info_file[0]
            # f7_redshift = f7_info_file[1]
            f7_masses_per_snapshot = f7_info_file[3]
            f7_alpha = f7_info_file[8]
            f7_mask = (f7_masses_per_snapshot > star_mass_min) & (f7_alpha < max_alpha)

            f7_masses_per_snapshot = f7_masses_per_snapshot[f7_mask]

        try:  # if there is only once cluster
            f3_t_myr = f3_info_file[0, 0]
            # f3_redshift = f3_info_file[0, 1]
            f3_masses_per_snapshot = f3_info_file[:, 3]
            f3_alpha = f3_info_file[:, 8]
            f3_mask = (f3_masses_per_snapshot > star_mass_min) & (f3_alpha < max_alpha)

            f3_masses_per_snapshot = f3_masses_per_snapshot[f3_mask]
        except:
            f3_t_myr = f3_info_file[0]
            # f3_redshift = f3_info_file[1]
            f3_masses_per_snapshot = f3_info_file[3]
            f3_alpha = f3_info_file[8]
            f3_mask = (f3_masses_per_snapshot > star_mass_min) & (f3_alpha < max_alpha)

            f3_masses_per_snapshot = f3_masses_per_snapshot[f3_mask]

        print(f7_t_myr, f3_t_myr)
        # print(f7_redshift, f3_redshift)

        # gc_ages_per_snapshot = info_file[:, 1]
        # gc_lums_per_snapshot = info_file[:, 3]

        # from the fof profiler

        # from the logSFC
        f3_birth_myr = t_myr_from_z(fs035_log_sfc[:, 2])
        f3_mc_mask = f3_birth_myr <= f3_t_myr
        f3_mc_star_mass = fs035_log_sfc[:, 7][f3_mc_mask]

        f7_birth_myr = t_myr_from_z(fs070_log_sfc[:, 2])
        f7_mc_mask = f7_birth_myr <= f7_t_myr
        f7_mc_star_mass = fs070_log_sfc[:, 7][f7_mc_mask]

        # log sfc
        mc_f7_mass, mc_f7_counts = log_data_function(
            f7_mc_star_mass, bns, x_range, return_counts=True
        )
        mc_f3_mass, mc_f3_counts = log_data_function(
            f3_mc_star_mass, bns, x_range, return_counts=True
        )
        # BSC
        f7_vir_mass, f7_vir_counts = log_data_function(
            f7_masses_per_snapshot, bns, x_range, return_counts=True
        )

        f3_vir_mass, f3_vir_counts = log_data_function(
            f3_masses_per_snapshot, bns, x_range, return_counts=True
        )
        print(len(f7_masses_per_snapshot), len(f7_mc_star_mass))
        print(len(f3_masses_per_snapshot), len(f3_mc_star_mass))

        # fit the BSC
        # f7_fitting_mask = f7_vir_mass >= 500  # 2e2
        # f3_fitting_mask = f3_vir_mass >= 500

        # count below 500, set it to 0, prevents the fit from weighing low mass

        # make the errors not strictly dependent on mass.
        #

        # f7_bsc_fitting = np.where(f7_vir_mass >= 500, f7_vir_counts, 0)
        # f3_bsc_fitting = np.where(f3_vir_mass >= 500, f3_vir_counts, 0)

        f7_weights = np.where(f7_vir_mass > 600, 1 / f7_vir_mass, 1)
        f3_weights = np.where(f3_vir_mass > 600, 1 / f3_vir_mass, 1)
        # f7_vir_counts[f7_fitting_mask] = 0
        # f7_vir_mass[f7_fitting_mask] = 0
        # f3_vir_counts[f3_fitting_mask] = 0
        # f3_vir_mass[f3_fitting_mask] = 0

        f7_fit_params, _ = curve_fit(
            f=gauss,
            xdata=np.log10(f7_vir_mass),
            ydata=f7_vir_counts,
            sigma=f7_weights,
            absolute_sigma=True
            # bounds=(
            #     [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            #     [600, np.inf, np.inf, 600, np.inf, np.inf],
            # ),
        )
        f7_theory_x = np.log10(np.geomspace(f7_vir_mass.min(), f7_vir_mass.max(), 100))
        f7_theory_y = gauss(f7_theory_x, *f7_fit_params)

        f3_fit_params, _ = curve_fit(
            f=gauss,
            xdata=np.log10(f3_vir_mass),
            ydata=f3_vir_counts,
            sigma=f3_weights,
            absolute_sigma=True
            # bounds=(
            #     [0, -np.inf, -np.inf, 0, -np.inf, -np.inf],
            #     [600, np.inf, np.inf, 600, np.inf, np.inf],
            # ),
        )
        f3_theory_x = np.log10(np.geomspace(f3_vir_mass.min(), f3_vir_mass.max(), 100))
        f3_theory_y = gauss(f3_theory_x, *f3_fit_params)

        #!!! fit the logSFC
        f7_mc_fit_params, _ = curve_fit(
            f=gauss,
            xdata=np.log10(mc_f7_mass),
            ydata=mc_f7_counts,
        )
        f7_mc_theory_x = np.log10(np.geomspace(mc_f7_mass.min(), mc_f7_mass.max(), 100))
        f7_mc_theory_y = gauss(f7_mc_theory_x, *f7_mc_fit_params)

        f3_mc_fit_params, _ = curve_fit(
            f=gauss,
            xdata=np.log10(mc_f3_mass),
            ydata=mc_f3_counts,
        )
        f3_mc_theory_x = np.log10(np.geomspace(mc_f3_mass.min(), mc_f3_mass.max(), 100))
        f3_mc_theory_y = gauss(f3_mc_theory_x, *f3_mc_fit_params)

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

            if i == 0:
                f7_legend_title = r"$\log_{{10}}\:(\mu,\:\sigma)$"
                f3_legend_title = r"$\log_{{10}}\:(\mu,\:\sigma)$"
            else:
                f7_legend_title = ""
                f3_legend_title = ""

            ax[i, 0].plot(
                mc_f7_mass,
                mc_f7_counts,
                label=f7_imf_label,
                drawstyle="steps-mid",
                linewidth=4,
                alpha=0.8,
                color=f7_mc_imf_clr,
            )
            ax[i, 0].plot(
                f7_vir_mass,  # [f7_fitting_mask]
                f7_vir_counts,  # [f7_fitting_mask]
                label=f7_bsc_label,
                drawstyle="steps-mid",
                linewidth=4,
                alpha=0.8,
                color=f7_bsc_mf_clr,
            )

            ax[i, 1].plot(
                mc_f3_mass,
                mc_f3_counts,
                label=f3_imf_label,
                drawstyle="steps-mid",
                linewidth=4,
                alpha=0.8,
                color=f3_mc_imf_clr,
            )
            ax[i, 1].plot(
                f3_vir_mass,  # [f3_fitting_mask]
                f3_vir_counts,  # [f3_fitting_mask]
                label=f3_bsc_label,
                drawstyle="steps-mid",
                linewidth=4,
                alpha=0.8,
                color=f3_bsc_mf_clr,
            )
            ax[i, 0].fill_between(
                mc_f7_mass, mc_f7_counts, step="mid", alpha=0.4, color=f7_mc_imf_clr
            )
            ax[i, 0].fill_between(
                f7_vir_mass, f7_vir_counts, step="mid", alpha=0.4, color=f7_bsc_mf_clr
            )
            ax[i, 1].fill_between(
                mc_f3_mass, mc_f3_counts, step="mid", alpha=0.4, color=f3_mc_imf_clr
            )
            ax[i, 1].fill_between(
                f3_vir_mass, f3_vir_counts, step="mid", alpha=0.4, color=f3_bsc_mf_clr
            )
            # plot the theoretical curves
            ax[i, 0].plot(
                10**f7_mc_theory_x,
                f7_mc_theory_y,
                ls=":",
                linewidth=2,
                alpha=0.8,
                color="grey",
                label=(r"$ ({:.2f}, {:.2f})$").format(
                    f7_mc_fit_params[1], np.abs(f7_mc_fit_params[2])
                ),
            )
            ax[i, 1].plot(
                10**f3_mc_theory_x,
                f3_mc_theory_y,
                ls=":",
                linewidth=2,
                alpha=0.8,
                color="grey",
                label=(r"$({:.2f}, {:.2f})$").format(
                    f3_mc_fit_params[1], np.abs(f3_mc_fit_params[2])
                ),
            )
            #!!! plot the BSC fit

            ax[i, 0].plot(
                10**f7_theory_x,
                f7_theory_y,
                ls="--",
                linewidth=2,
                alpha=0.8,
                color="black",
                label=(r"$({:.2f}, {:.2f})$").format(
                    f7_fit_params[1],
                    abs(f7_fit_params[2]),
                ),
            )

            ax[i, 1].plot(
                10**f3_theory_x,
                f3_theory_y,
                ls="--",
                linewidth=2,
                alpha=0.8,
                color="black",
                label=(r"$({:.2f}, {:.2f})$").format(
                    f3_fit_params[1],
                    abs(f3_fit_params[2]),
                ),
            )

            # current simulation time annotation
            props = dict(
                boxstyle="round",
                facecolor="white",
                alpha=1,
                linewidth=0.5,
                edgecolor="black",
            )
            textstr_f7 = (r"$\mathrm{{t}} = {:.1f} \: \mathrm{{Myr}}$").format(f7_t_myr)

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
            ax[i, 0].set_xlim(left=50, right=8e4)
            ax[i, 0].set_xscale("log")
            ax[i, 0].set_yscale("log")
            ax[i, 0].set_ylim(0.8, 380)
            # ax[i, 0].set_ylim(10, 500000)

            ax[i, 1].yaxis.tick_right()
            ax[i, 1].set_yticklabels([])
            ax[i, 1].set_xscale("log")
            ax[i, 1].set_yscale("log")
            ax[i, 1].set_ylim(0.8, 380)
            # ax[i, 1].set_ylim(10, 500000)

            ax[i, 0].legend(
                title=f7_legend_title,
                loc="upper left",
                framealpha=0.5,
                title_fontsize=11,
                fontsize=10,
            )

            ax[i, 1].legend(
                title=f3_legend_title,
                loc="upper right",
                framealpha=0.5,
                title_fontsize=11,
                fontsize=10,
            )

            ax[i, 1].text(
                0,
                0.94,
                textstr_f7,
                transform=ax[i, 1].transAxes,
                verticalalignment="top",
                horizontalalignment="center",
                bbox=props,
                clip_on=False,
            )
            ax[i, 1].tick_params(axis="both", direction="in", which="both")
            ax[i, 0].tick_params(axis="both", direction="in", which="both")
            ax[i, 1].set_yticklabels([])

    plt.savefig(
        os.path.expanduser(
            (
                "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/"
                "cmf_overtime.png"
            )
        ),
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.08,
    )
