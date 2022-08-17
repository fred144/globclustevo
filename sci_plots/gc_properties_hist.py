"""

histograms of GC params
"""

import sys

sys.path.append("../")
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
from modules.macros import filter_snapshots, common_filter_snapshots, t_myr_from_z
from modules.match_t_sims import find_matching_time, get_snapshots
import matplotlib.lines as mlines

cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))

f7_mc_imf_clr = cmap[0]
f7_bsc_mf_clr = cmap[1]
f3_mc_imf_clr = cmap[2]
f3_bsc_mf_clr = cmap[3]

f7_strt = 113
f7_end = 1110
f3_strt = 154
f3_end = 1316
step = 1


f7_prof_dir = r"../gc_profiles/profile_runs/fs07_refine/fof_best"
f3_prof_dir = r"../gc_profiles/profile_runs/fs035_ms10/fof_best"

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
strt = 921
end = 922
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


cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))

fs70_color = cmap[1]
fs70_face = np.copy(fs70_color)
fs70_face[-1] = 0.40
fs35_color = cmap[2]
fs35_face = np.copy(fs35_color)
fs35_face[-1] = 0.40

# hist_bins = 20

#%%
for i, (f7, f3) in enumerate(zip(f7_pro_ds, f3_pro_ds)):
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
    f7_vir_rad = f7_prof_data[:, 5]
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
    f3_vir_rad = f3_prof_data[:, 5]
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

    f7_mask = f7_alpha < 8  # & (f7_mass < 1e4)   & (f7_vir_rad < 10)
    f3_mask = f3_alpha < 8  # & (f3_mass < 1e4)   & (f3_vir_rad < 10)

    f7_vars = [
        f7_core_rad[f7_mask],
        f7_sig_0[f7_mask],
        f7_alpha[f7_mask],
        (f7_mass / f7_orig_mass)[f7_mask],
    ]
    f3_vars = [
        f3_core_rad[f3_mask],
        f3_sig_0[f3_mask],
        f3_alpha[f3_mask],
        (f3_mass / f3_orig_mass)[f3_mask],
    ]

    labels = [
        r"$R_\mathrm{{core}}\:\mathrm{(pc)}$",
        r"$\Sigma_0\:\mathrm{\left(M_{\odot}\:pc^{-2}\right)}$",
        r"$\alpha$",
        r"$\mathrm{M_{BSC}\: / \: M_{SFC} }$",
    ]

    # hist_ranges = [(0, 2), (10, 800), (1.5, 5), (0, 1)]
    num_bins = 15
    bins = [
        np.linspace(f7_core_rad.min(), f7_core_rad.max(), num_bins),
        np.linspace(f7_sig_0.min(), f7_sig_0.max(), num_bins),
        np.linspace(f7_alpha.min(), f7_alpha.max(), num_bins),
        np.linspace(
            (f7_mass / f7_orig_mass).min(), (f7_mass / f7_orig_mass).max(), num_bins
        ),
    ]
    with plt.rc_context(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.size": 10,
        }
    ):

        fig, ax = plt.subplots(
            nrows=2,
            ncols=2,
            sharey=True,
            figsize=(4, 4.3),
            dpi=300,
        )
        plt.subplots_adjust(wspace=0, hspace=0.4)
        ax = ax.ravel()

        for i, (f7_var, f3_var) in enumerate(zip(f7_vars, f3_vars)):

            ax[i].hist(
                f3_var,
                bins=bins[i],
                color=fs35_color,
                histtype="step",
                hatch="xxx",
                # edgecolor="white",
                alpha=0.7,
                linewidth=2,
                label=r"$0.35$",
            )

            ax[i].hist(
                f7_var,
                bins=bins[i],
                color=fs70_color,
                histtype="step",
                hatch="++",
                # edgecolor="white",
                alpha=0.7,
                linewidth=2,
                label=r"$0.70$",
            )

            # f7_count, f7_bin_edges = np.histogram(f7_var, hist_bins, hist_ranges[i])
            # f7_right_edges = f7_bin_edges[1:]
            # f7_left_edges = f7_bin_edges[:-1]
            # f7_bin_ctrs = 0.5 * (f7_left_edges + f7_right_edges)

            # ax[i].plot(
            #     f7_bin_ctrs,
            #     f7_count,
            #     label=r"$0.70$",
            #     drawstyle="steps-mid",
            #     linewidth=2,
            #     alpha=0.8,
            #     color=fs70_color,
            # )
            # ax[i].fill_between(
            #     f7_bin_ctrs,
            #     f7_count,
            #     step="mid",
            #     alpha=0.4,
            #     color=fs70_color,
            # )

            # f3_count, f3_bin_edges = np.histogram(f3_var, hist_bins, hist_ranges[i])
            # f3_right_edges = f3_bin_edges[1:]
            # f3_left_edges = f3_bin_edges[:-1]
            # f3_bin_ctrs = 0.5 * (f3_left_edges + f3_right_edges)

            # ax[i].plot(
            #     f3_bin_ctrs,
            #     f3_count,
            #     label=r"$0.35$",
            #     drawstyle="steps-mid",
            #     linewidth=2,
            #     alpha=0.8,
            #     color=fs35_color,
            # )
            # ax[i].fill_between(
            #     f3_bin_ctrs,
            #     f3_count,
            #     step="mid",
            #     alpha=0.4,
            #     color=fs35_color,
            # )
            # ax[i].set_xlim(left=f7_bin_ctrs[0], right=f7_bin_ctrs[-1])
            # ax[i].set_ylim(bottom=0, top=25)

            ax[i].set_xlabel(labels[i])

        # ax[0].set_ylabel(r"$\mathrm{Number\:\:of\:\:BSCs}$")

        ax[0].legend(title="$\mathrm{SFE} \: (f_{*})$", loc="upper right", fontsize=10)
        # ax[2].text(
        #     0.06,
        #     0.92,
        #     r"$\mathrm{{t = {:.0f} \: Myr}}$".format(f3_t_myr),
        #     ha="left",
        #     va="top",
        #     transform=ax[2].transAxes,
        #     fontsize=10,
        #     bbox={
        #         "boxstyle": "round",
        #         # have control over edge alpha and face alpha
        #         "linewidth": 1,
        #         "edgecolor": "grey",
        #         "alpha": 0.5,
        #         "facecolor": "white",
        #         # "pad": 0.42,
        #     },
        # )
        fig.text(
            0.01,
            0.5,
            r"$\mathrm{Number\:\:of\:\:BSCs}$",
            va="center",
            rotation="vertical",
        )
plt.savefig(
    os.path.expanduser(
        (
            "~/g_drive/Research/AstrophysicsSimulation/sci_plots/final/"
            "fit_params_hist.png"
        )
    ),
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.05,
)
# ax[0].hist(
#     f7_core_rad,
#     bins=bins,
#     # alpha=0.8,
#     edgecolor=fs70_color,
#     histtype="step",
#     lw=2,
#     facecolor=fs70_face,
#     fill=True,
# )

# ax[0].hist(
#     f3_core_rad,
#     bins=bins,
#     # alpha=0.8,
#     edgecolor=fs35_color,
#     histtype="step",
#     lw=2,
#     facecolor=fs35_face,
#     fill=True,
# )
