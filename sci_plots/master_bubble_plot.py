import sys

sys.path.append("../")
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
from modules.macros import filter_snapshots, common_filter_snapshots, t_myr_from_z
from modules.match_t_sims import find_matching_time, get_snapshots
import matplotlib.lines as mlines
import matplotlib.font_manager as font_manager

cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))

f7_mc_imf_clr = cmap[0]
f7_bsc_mf_clr = cmap[1]
f3_mc_imf_clr = cmap[2]
f3_bsc_mf_clr = cmap[3]

f7_strt = 113
f7_end = 1000
f3_strt = 154
f3_end = 1177
step = 1

profiler_data = (
    "../../g_drive/Research/AstrophysicsSimulation/DesktopEnvironment/"
    "data_globular_cluster/gc_profiles/profile_runs/"
)

# f7_prof_dir = r"../gc_profiles/profile_runs/fs07_refine/fof_best"
# f3_prof_dir = r"../gc_profiles/profile_runs/fs035_ms10/fof_best"

# point to google drice for profiler runs.
f7_prof_dir = profiler_data + "fs07_refine/fof_best"
f3_prof_dir = profiler_data + "fs035_ms10/fof_best"

f7_halo_dir = r"../halo_data/fs07_refine/fof_best"
f3_halo_dir = r"../halo_data/fs035_ms10/fof_best"

# using pop2 data to construct lookuptables
f7_pop2_ds = filter_snapshots(
    r"../particle_data/pop_2_data/fs07_refine", f7_strt, f7_end, step
)
f3_pop2_ds = filter_snapshots(
    r"../particle_data/pop_2_data/fs035_ms10", f3_strt, f3_end, step
)


_, f3_matched_nums = find_matching_time(
    sequence=f7_pop2_ds, look_up_sequence=f3_pop2_ds
)
f3_pro_ds = filter_snapshots(f3_prof_dir, f3_strt, f3_end, step)

# sample the matched snapshots for plotting by indexing
# try snapshot 873
strt = 887
end = f3_matched_nums.size
st = 1

f7_pro_ds = filter_snapshots(f7_prof_dir, f7_strt, f7_end, step)[strt:end:st]
f3_pro_ds = get_snapshots(f3_pro_ds, get_list=f3_matched_nums)[strt:end:st]

f7_halo_ds = filter_snapshots(f7_halo_dir, f7_strt, f7_end, step)[strt:end:st]
f3_halo_ds = get_snapshots(
    filter_snapshots(f3_halo_dir, f3_strt, f3_end, step), get_list=f3_matched_nums
)[strt:end:st]

f3_pop2_ds = get_snapshots(f3_pop2_ds, get_list=f3_matched_nums)[strt:end:st]
f7_pop2_ds = f7_pop2_ds[strt:end:st]


def metal_lookup(log_sfc_path, bsc_form_times):

    log = np.loadtxt(log_sfc_path)
    z_form = log[:, 2]
    t_form = t_myr_from_z(z_form)
    m_sun_form = log[:, 7]  # mass in stars
    z_sun = log[:, 9]  # solar metallicity

    residuals = np.abs(t_form - bsc_form_times[:, np.newaxis])
    closest_match_idxs = np.argmin(residuals, axis=1)
    bsc_metals = z_sun[closest_match_idxs]
    bsc_m_sun_form = m_sun_form[closest_match_idxs]
    return bsc_metals, bsc_m_sun_form


#%%

for i, (f7, f3) in enumerate(zip(f7_pro_ds, f3_pro_ds)):
    # time series loop
    # if i == 0:
    #     continue
    f7_prof_data = np.loadtxt(os.path.join(f7, "info.txt"))
    f3_prof_data = np.loadtxt(os.path.join(f3, "info.txt"))

    # load the profiled BSCs data
    f7_t_myr = f7_prof_data[0, 0]
    f7_labels = f7_prof_data[:, 1]
    f7_ages = f7_prof_data[:, 2]
    f7_bes = f7_t_myr - f7_ages
    f7_mass = f7_prof_data[:, 3]
    f7_core_mass = f7_prof_data[:, 4]
    f7_trunc_rad = f7_prof_data[:, 5]
    # derived qunatities
    f7_core_rad = f7_prof_data[:, 6]
    f7_core_err = f7_prof_data[:, 7]
    f7_alpha = f7_prof_data[:, 8]
    f7_alpha_err = f7_prof_data[:, 9]
    f7_sig_0 = f7_prof_data[:, 10]
    f7_sig_0_err = f7_prof_data[:, 11]
    f7_sig_bg = f7_prof_data[:, 12]
    f7_sig_bg_err = f7_prof_data[:, 13]
    f7_fit_pval = f7_prof_data[:, 14]
    f7_half_mass_rad = f7_prof_data[:, 15]
    f7_half_light_rad = f7_prof_data[:, 16]
    f7_tot_light = f7_prof_data[:, 17]
    f7_metal, f7_orig_mass = metal_lookup("../sim_log_files/fs07_refine/logSFC", f7_bes)

    # half efficiency
    f3_t_myr = f3_prof_data[0, 0]
    f3_labels = f3_prof_data[:, 1]
    f3_ages = f3_prof_data[:, 2]
    f3_bes = f3_t_myr - f3_ages
    f3_mass = f3_prof_data[:, 3]
    f3_core_mass = f3_prof_data[:, 4]
    f3_trunc_rad = f3_prof_data[:, 5]
    # derived qunatities
    f3_core_rad = f3_prof_data[:, 6]
    f3_core_err = f3_prof_data[:, 7]
    f3_alpha = f3_prof_data[:, 8]
    f3_alpha_err = f3_prof_data[:, 9]
    f3_sig_0 = f3_prof_data[:, 10]
    f3_sig_0_err = f3_prof_data[:, 11]
    f3_sig_bg = f3_prof_data[:, 12]
    f3_sig_bg_err = f3_prof_data[:, 13]
    f3_fit_pval = f3_prof_data[:, 14]
    f3_half_mass_rad = f3_prof_data[:, 15]
    f3_half_light_rad = f3_prof_data[:, 16]
    f3_tot_light = f3_prof_data[:, 17]
    f3_metal, f3_orig_mass = metal_lookup("../sim_log_files/fs035_ms10/logSFC", f3_bes)

    with plt.rc_context(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "font.size": 10,
        }
    ):
        leg_font = font_manager.FontProperties(
            family="serif", math_fontfamily="cm", size=10
        )

        cmap = plt.cm.get_cmap("winter")
        scale_factor = 20  # scale factor for the sizes
        # map to differnt sizes for better plotting
        f7_half_radii = scale_factor * f7_half_mass_rad
        f3_half_radii = scale_factor * f3_half_mass_rad

        x_vars = [
            np.nan,
            np.nan,
            # (f7_metal, f3_metal),
            (f7_metal, f3_metal),
            (f7_sig_0, f3_sig_0),
            (f7_metal, f3_metal),
            # (f7_orig_mass, f3_orig_mass),
            # (f7_orig_mass, f3_orig_mass),
            (f7_alpha, f3_alpha),
            # (f7_metal, f3_metal),
            (f7_core_mass / f7_orig_mass, f3_core_mass / f3_orig_mass),
            (f7_mass / f7_orig_mass, f3_mass / f3_orig_mass),
            (f7_alpha, f3_alpha),
            # (f7_core_mass / f7_orig_mass, f3_core_mass / f3_orig_mass),
        ]
        y_vars = [
            np.nan,
            np.nan,
            # (f7_sig_0, f3_sig_0),
            (f7_tot_light, f3_tot_light),
            (f7_mass / f7_orig_mass, f3_mass / f3_orig_mass),
            (f7_mass / f7_orig_mass, f3_mass / f3_orig_mass),
            # (f7_mass, f3_mass),
            # (f7_sig_0, f3_sig_0),
            (f7_sig_0, f3_sig_0),
            # (f7_mass, f3_mass),
            (f7_tot_light, f3_tot_light),
            (f7_core_mass, f3_core_mass),
            (f7_core_rad, f3_core_rad),
            # (f7_tot_light, f3_tot_light),
        ]
        x_labels = [
            "padax",
            "padax",
            # r"$\mathrm{Z_{BSC}\:\left(Z_{\odot}\right)}$",
            r"$\log_{10}\:\mathrm{Z_{BSC}\:\left(Z_{\odot}\right)}$",
            r"$\log_{10}\:\mathrm{\Sigma_0\:\left(M_{\odot}\:pc^{-2}\right)}$",
            r"$\log_{10}\:\mathrm{Z_{BSC}\:\left(Z_{\odot}\right)}$",
            # r"$\mathrm{M_{SFC}}$",
            # r"$\mathrm{M_{SFC}}$",
            r"$\log_{10}\:\alpha$",
            # r"$\mathrm{Z_{BSC}\:\left(Z_{\odot}\right)}$",
            r"$\log_{10}\:\mathrm{M_{core}\: / \: M_{SFC} }$",
            r"$\log_{10}\:\mathrm{M_{BSC}\: / \: M_{SFC} }$",
            r"$\log_{10}\:\alpha$",
            # r"$\mathrm{M_{core}\: / \: M_{BSC} }$",
        ]
        y_labels = [
            "padax",
            "padax",
            # r"$\mathrm{\Sigma_0\:\left(M_{\odot}\:pc^{-2}\right)}$",
            (
                r"$\log_{10}\:\mathrm{L_{BSC, \: \mathrm{\lambda = 1500 \: \AA \:}}}$"
                "\n"
                r"$\mathrm{\left(erg\:\:s^{-1}\:\AA^{-1}\right)}$"
            ),
            r"$\log_{10}\:\mathrm{M_{BSC}\: / \: M_{SFC} }$" "\n",
            r"$\log_{10}\:\mathrm{M_{BSC}\: / \: M_{SFC} }$",
            # r"$\mathrm{M_{BSC}}$",
            # r"$\mathrm{\Sigma_0\:\left(M_{\odot}\:pc^{-2}\right)}$",
            r"$\log_{10}\:\mathrm{\Sigma_0\:\left(M_{\odot}\:pc^{-2}\right)}$",
            # r"$\mathrm{M_{BSC}}$",
            (
                r"$\log_{10}\:\mathrm{L_{BSC, \: \mathrm{\lambda = 1500 \: \AA \:}}}$"
                "\n"
                r"$\mathrm{\left(erg\:\:s^{-1}\:\AA^{-1}\right)}$"
            ),
            r"$\log_{10}\:M_{\mathrm{core}}$",
            r"$\log_{10}\:R_{\mathrm{core}}$",
            # (
            #     r"$\mathrm{L_{BSC}}$"
            #     r"$, \mathrm{\lambda = 1500 \: \AA \:}$"
            #     r"$\mathrm{\left(erg\:\:s^{-1}\:\AA^{-1}\:pc^{-2}\right)}$"
            # ),
        ]
        xlims = [
            np.nan,
            np.nan,
            # (1e-4, 1e-2),
            (1e-4, 7e-3),
            (5, 5e4),
            (3e-4, 1e-2),
            # (1e2, 3e4),
            # (2e2, 2e4),
            (1, 11),
            # (1e-4, 1e-2),
            (5e-4, 20),
            (5e-3, 20),
            (1, 11),
            # (2e-2, 2),
        ]
        ylims = [
            np.nan,
            np.nan,
            # (5e0, 1e5),
            (1e33, 3e37),
            (6e-3, 20),
            (5e-3, 20),
            # (2e1, 1e5),
            # (5e0, 1e5),
            (5e0, 1e5),
            # (1e1, 8e4),
            (1e33, 3e37),
            (5, 1e5),
            (2e-2, 10),
            # (1e33, 3e37),
        ]
        fig, ax = plt.subplots(
            nrows=3,
            ncols=3,
            figsize=(7, 6),
            dpi=400,
            # sharex=True,
            # sharey=True,
        )
        plt.subplots_adjust(hspace=0.35, wspace=0.28)
        axs = ax.ravel()

        vmin = 330
        vmax = 610

        for i, (x, y) in enumerate(zip(x_vars, y_vars)):

            if i == 0 or i == 1:
                axs[i].set_visible(False)
                continue

            # print("test", i)

            f3_scatter = axs[i].scatter(
                x[1],
                y[1],
                c=f3_bes,
                s=f3_half_radii,
                linewidths=1,
                alpha=0.8,
                edgecolors="k",
                marker="o",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            f7_scatter = axs[i].scatter(
                x[0],
                y[0],
                c=f7_bes,
                edgecolors="None",
                s=f7_half_radii,
                alpha=0.8,
                marker="o",
                cmap=cmap,
                linewidths=0,
                vmin=vmin,
                vmax=vmax,
            )

            # color bars
            cbar_ax = fig.add_axes([0.09, 0.80, 0.5, 0.04])
            cbar = fig.colorbar(
                f3_scatter,
                cax=cbar_ax,
                pad=0,
                orientation="horizontal",
            )
            cbar.set_alpha(0.8)
            cbar_ax.set_title(
                label="$\mathrm{Time \: of \: Formation \: (Myr)}$",
                fontproperties=leg_font,
            )
            cbar.ax.tick_params(labelsize=7)

            axs[i].set_xscale("log")
            axs[i].set_yscale("log")
            axs[i].minorticks_off()
            axs[i].set_xlabel(x_labels[i], labelpad=0)
            axs[i].set_ylabel(y_labels[i], labelpad=0)
            axs[i].set_xlim(left=xlims[i][0], right=xlims[i][1])
            axs[i].set_ylim(bottom=ylims[i][0], top=ylims[i][1])

        # =============================================================================
        # manual legend, want to set sfes
        # title = mlines.Line2D(
        #     [],
        #     [],
        #     color="white",
        #     marker="o",
        #     ls="",
        #     label="SFE$\:(f_{*})$",
        #     alpha=0.0,
        #     markeredgewidth=0,
        #     markersize=0,
        # )
        f70 = mlines.Line2D(
            [],
            [],
            color="grey",
            marker="o",
            ls="",
            label=r"0.70",
            alpha=0.8,
            markeredgewidth=0,
            markersize=8,
        )
        f35 = mlines.Line2D(
            [],
            [],
            color="grey",
            marker="o",
            ls="",
            label=r"0.35",
            alpha=0.8,
            markeredgecolor="k",
            markersize=8,
        )
        sfe_legend = fig.legend(
            loc="lower left",
            title="$\mathrm{SFE}\:(f_{*})$",
            title_fontsize=10,
            fontsize=8,
            handles=[f70, f35],
            ncol=1,
            bbox_to_anchor=(0.1, 0.63),
        )
        sfe_legend.get_frame().set_edgecolor("k")
        sfe_legend.get_frame().set_boxstyle("Square")

        # legend mapped to size
        legend_properties = dict(
            prop="sizes",
            num=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            color="grey",
            fmt=" {x:.2f}",
            func=lambda d: d / scale_factor,
        )
        size_legend = fig.legend(
            *f7_scatter.legend_elements(**legend_properties),
            loc="lower left",
            title="$\mathrm{R_{half}\:(pc)}$ ",
            title_fontsize=10,
            fontsize=8,
            ncol=3,
            bbox_to_anchor=(0.22, 0.63),
        )
        size_legend.get_frame().set_edgecolor("k")
        size_legend.get_frame().set_boxstyle("Square")
        plt.gca().add_artist(size_legend)

        fig.canvas.draw()
        # for i in range(2, 9):
        #     x_labels = [
        #         t.get_text().replace("10^", "") for t in axs[i].get_xticklabels()
        #     ]
        #     y_labels = [
        #         t.get_text().replace("10^", "") for t in axs[i].get_yticklabels()
        #     ]
        #     axs[i].set_xticklabels(x_labels)
        #     axs[i].set_yticklabels(y_labels)
        # ax.grid(visible=True, zorder=0.5)

# plt.savefig(
#     os.path.expanduser(
#         (
#             "~/g_drive/Research/AstrophysicsSimulation/sci_plots/final/"
#             "bubble_plot_dashboard.png"
#         )
#     ),
#     dpi=500,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
