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
f7_end = 1000
f3_strt = 154
f3_end = 1177
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
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.size": 12,
        }
    ):

        cmap = plt.cm.get_cmap("winter")
        scale_factor = 50  # scale factor for the sizes
        # map to differnt sizes for better plotting
        f7_half_radii = scale_factor * f7_half_mass_rad
        f3_half_radii = scale_factor * f3_half_mass_rad

        x_vars = [
            (f7_metal, f3_metal),
            (f7_sig_0, f3_sig_0),
            (f7_metal, f3_metal),
            (f7_orig_mass, f3_orig_mass),
            (f7_orig_mass, f3_orig_mass),
            (f7_alpha, f3_alpha),
            (f7_metal, f3_metal),
            (f7_metal, f3_metal),
            (f7_mass / f7_orig_mass, f3_mass / f3_orig_mass),
            (f7_core_mass / f7_orig_mass, f3_core_mass / f3_orig_mass),
            (f7_alpha, f3_alpha),
        ]
        y_vars = [
            (f7_sig_0, f3_sig_0),
            (f7_mass / f7_orig_mass, f3_mass / f3_orig_mass),
            (f7_mass / f7_orig_mass, f3_mass / f3_orig_mass),
            (f7_mass, f3_mass),
            (f7_sig_0, f3_sig_0),
            (f7_sig_0, f3_sig_0),
            (f7_mass, f3_mass),
            (f7_tot_light, f3_tot_light),
            (f7_core_mass, f3_core_mass),
            (f7_tot_light, f3_tot_light),
            (f7_core_rad, f3_core_rad),
        ]
        x_labels = [
            r"$\mathrm{Z_{BSC}\:\left(Z_{\odot}\right)}$",
            r"$\mathrm{\Sigma\:\left(M_{\odot}\:pc^{-2}\right)}$",
            r"$\mathrm{Z_{BSC}\:\left(Z_{\odot}\right)}$",
            r"$\mathrm{M_{SFC}}$",
            r"$\mathrm{M_{SFC}}$",
            r"$\alpha$",
            r"$\mathrm{Z_{BSC}\:\left(Z_{\odot}\right)}$",
            r"$\mathrm{Z_{BSC}\:\left(Z_{\odot}\right)}$",
            r"$\mathrm{M_{BSC}\: / \: M_{SFC} }$",
            r"$\mathrm{M_{core}\: / \: M_{SFC} }$",
            r"$\alpha$",
        ]
        y_labels = [
            r"$\mathrm{\Sigma\:\left(M_{\odot}\:pc^{-2}\right)}$",
            r"$\mathrm{M_{BSC}\: / \: M_{SFC} }$",
            r"$\mathrm{M_{BSC}\: / \: M_{SFC} }$",
            r"$\mathrm{M_{BSC}}$",
            r"$\mathrm{\Sigma\:\left(M_{\odot}\:pc^{-2}\right)}$",
            r"$\mathrm{\Sigma\:\left(M_{\odot}\:pc^{-2}\right)}$",
            r"$\mathrm{M_{BSC}}$",
            (
                r"$\mathrm{Projected\:Luminosity}$"
                r"$, \mathrm{\lambda = 1500 \: \AA \:}$"
                "\n"
                r"$\mathrm{\left(erg\:\:s^{-1}\:\AA^{-1}\:pc^{-2}\right)}$"
            ),
            r"$M_{\mathrm{core}}$",
            (
                r"$\mathrm{Projected\:Luminosity}$"
                r"$, \mathrm{\lambda = 1500 \: \AA \:}$"
                "\n"
                r"$\mathrm{\left(erg\:\:s^{-1}\:\AA^{-1}\:pc^{-2}\right)}$"
            ),
            r"$R_{\mathrm{core}}$",
        ]
        xlims = [
            (1e-4, 1e-2),
            (5e0, 1e5),
            (1e-4, 1e-2),
            (2e1, 1e5),
            (5e0, 1e5),
            (1, 13),
            (1e-4, 1e-2),
            (1e-4, 1e-2),
            (1e-3, 10),
            (5e-4, 10),
            (1, 13),
        ]
        ylims = [
            (5e0, 1e5),
            (5e-3, 1e1),
            (5e-3, 1e1),
            (2e1, 1e5),
            (5e0, 1e5),
            (5e0, 1e5),
            (1e1, 8e4),
            (1e33, 3e37),
            (5, 1e5),
            (1e33, 3e37),
            (2e-2, 10),
        ]

        for i, (x, y) in enumerate(zip(x_vars, y_vars)):

            fig, ax = plt.subplots(1, 1, figsize=(4, 3.5), dpi=400)

            f7_scatter = plt.scatter(
                x[0],
                y[0],
                c=f7_bes,
                edgecolors="None",
                s=f7_half_radii,
                alpha=0.8,
                marker="o",
                cmap=cmap,
                linewidths=0,
            )
            f3_scatter = plt.scatter(
                x[1],
                y[1],
                c=f3_bes,
                s=f3_half_radii,
                linewidths=1,
                alpha=0.8,
                edgecolors="k",
                marker="o",
                cmap=cmap,
            )

            # manual legend, want to set sfes
            f70 = mlines.Line2D(
                [],
                [],
                color="grey",
                marker="o",
                ls="",
                label=r"0.70",
                alpha=0.8,
                markeredgewidth=0,
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
            )
            sfe_legend = plt.legend(
                title="$\mathrm{SFE} \: (f_{*})$",
                loc="lower right",
                title_fontsize=10,
                fontsize=8,
                handles=[f70, f35],
                facecolor=(1, 1, 1, 0.5),
            )
            ax.add_artist(sfe_legend)

            # legend mapped to size
            legend_properties = dict(
                prop="sizes",
                num=[0.50, 1.00, 1.50, 2.0],
                color="grey",
                fmt=" {x:.2f}",
                func=lambda d: d / scale_factor,
            )
            legend = ax.legend(
                *f7_scatter.legend_elements(**legend_properties),
                loc="upper left",
                title="$\mathrm{R_{half}\:(pc)}$ ",
                fontsize=8,
                title_fontsize=10,
                facecolor=(1, 1, 1, 0.5),
                ncol=2,
            )
            # color bars
            cbar = plt.colorbar(
                pad=0,
            )
            cbar.set_label(
                label="$\mathrm{t_{\mathrm{formation}}\: (Myr)}$", fontsize=14
            )
            plt.clim(vmin=330, vmax=600)
            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.set_xlabel(x_labels[i])
            ax.set_ylabel(y_labels[i])
            ax.set_xlim(left=xlims[i][0], right=xlims[i][1])
            ax.set_ylim(bottom=ylims[i][0], top=ylims[i][1])
            # ax.grid(visible=True, zorder=0.5)
