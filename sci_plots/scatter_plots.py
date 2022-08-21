import sys

sys.path.append("../")
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
from modules.macros import filter_snapshots, common_filter_snapshots, t_myr_from_z
from modules.match_t_sims import find_matching_time, get_snapshots
import matplotlib.lines as mlines
from scipy import stats

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
# try snapshot 873
strt = 921  # 800
end = 922  # f3_matched_nums.size
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


def lin_model(x, slope, intercept):
    return (slope * x) + intercept


cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))

fs70_color = cmap[1]
fs35_color = cmap[2]

for sn, (f7, f3) in enumerate(zip(f7_pro_ds, f3_pro_ds)):
    # if i == 0:
    #     continue
    f7_prof_data = np.loadtxt(os.path.join(f7, "info.txt"))
    f3_prof_data = np.loadtxt(os.path.join(f3, "info.txt"))

    f7_halo_data = np.loadtxt(os.path.join(f7_halo_ds[sn], "fof_info.txt"))

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

    with plt.rc_context(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "font.size": 10,
        }
    ):

        cmap = plt.cm.get_cmap("plasma")
        # scale_factor = 50  # scale factor for the sizes
        # # map to differnt sizes for better plotting
        # f7_half_radii = scale_factor * f7_half_mass_rad
        # f3_half_radii = scale_factor * f3_half_mass_rad
        f7_mask = (f7_alpha < 5) & (f7_mass < 1e4)  # & (f7_vir_rad < 10)
        f3_mask = (f3_alpha < 5) & (f3_mass < 1e4)  # & (f3_vir_rad < 10)

        x_vars = [
            np.nan,
            (f7_metal[f7_mask], f3_metal[f3_mask]),
            (f7_mass[f7_mask], f3_mass[f3_mask]),
            (f7_half_mass_rad[f7_mask], f3_half_mass_rad[f3_mask]),
            (f7_mass[f7_mask], f3_mass[f3_mask]),
            (f7_half_mass_rad[f7_mask], f3_half_mass_rad[f3_mask]),
            # (f7_half_mass_rad[f7_mask], f3_half_mass_rad[f3_mask]),
            # (f7_metal[f7_mask], f3_metal[f3_mask]),
            (f7_metal[f7_mask], f3_metal[f3_mask]),
            # (f7_metal[f7_mask], f3_metal[f3_mask]),
            # (f7_half_mass_rad[f7_mask], f3_half_mass_rad[f3_mask]),
            (f7_core_rad[f7_mask], f3_core_rad[f3_mask]),
            # (f7_alpha[f7_mask], f3_alpha[f3_mask]),
            (f7_alpha[f7_mask], f3_alpha[f3_mask]),
            # (f7_half_mass_rad[f7_mask], f3_half_mass_rad[f3_mask]),
            # (f7_bes[f7_mask], f3_bes[f3_mask]),
        ]
        y_vars = [
            np.nan,
            (
                (f7_core_mass / f7_core_rad**3)[f7_mask],
                (f3_core_mass / f3_core_rad**3)[f3_mask],
            ),
            ((f7_core_mass / f7_mass)[f7_mask], (f3_core_mass / f3_mass)[f3_mask]),
            (f7_mass[f7_mask], f3_mass[f3_mask]),
            (f7_core_mass[f7_mask], f3_core_mass[f3_mask]),
            (f7_core_rad[f7_mask], f3_core_rad[f3_mask]),
            # (f7_half_mass_rad[f7_mask], f3_half_mass_rad[f3_mask]),
            # (f7_mass[f7_mask], f3_mass[f3_mask]),
            (f7_half_mass_rad[f7_mask], f3_half_mass_rad[f3_mask]),
            # (f7_core_rad[f7_mask], f3_core_rad[f3_mask]),
            # (f7_sig_0[f7_mask], f3_sig_0[f3_mask]),
            (f7_sig_0[f7_mask], f3_sig_0[f3_mask]),
            # (f7_half_mass_rad[f7_mask], f3_half_mass_rad[f3_mask]),
            (f7_core_rad[f7_mask], f3_core_rad[f3_mask]),
            # (f7_core_rad[f7_mask], f3_core_rad[f3_mask]),
            # (f7_half_mass_rad[f7_mask], f3_half_mass_rad[f3_mask]),
        ]
        x_labels = [
            np.nan,
            r"$\log_{10} \: \mathrm{Z_{BSC}\:\left(Z_{\odot}\right)}$",
            r"$\log_{10} \: \mathrm{M_{BSC}} \: \mathrm{(M_{\odot})}$",
            r"$\log_{10} \: \mathrm{R_{half-mass} \: (pc)}$",
            r"$\log_{10} \: \mathrm{M_{BSC}} \: \mathrm{(M_{\odot})}$",
            r"$\log_{10} \: \mathrm{ R_{half-mass} \:(pc)}$",
            # r"$\mathrm{ R_{half-mass} \:(pc)}$",
            # r"$\mathrm{Z_{BSC}\:\left(Z_{\odot}\right)}$",
            r"$\log_{10} \: \mathrm{Z_{BSC}\:\left(Z_{\odot}\right)}$",
            # r"$\mathrm{Z_{BSC}\:\left(Z_{\odot}\right)}$",
            # r"$\mathrm{ R_{half-mass} \:(pc)}$",
            r"$\log_{10} \: R\mathrm{_{core} \: (pc)}$",
            # r"$\alpha$",
            r"$\alpha$",
            # r"$\mathrm{ R_{half-mass} \:(pc)}$",
            # "birth",
        ]
        y_labels = [
            np.nan,
            r"$\log_{10} \: \mathrm{M_{\odot} \: pc^{-3}}$",
            r"$\mathrm{M_{core}\: / M_{BSC} }$",
            r"$\log_{10} \: \mathrm{M_{BSC}} \: \mathrm{(M_{\odot})} $",
            r"$\log_{10} \: \mathrm{M_{core}}\: \mathrm{(M_{\odot})}$",
            r"$\log_{10} \: R\mathrm{_{core} \:(pc) }$",
            # r"$\mathrm{R_{half-mass}\:(pc)}$ ",
            # r"$\mathrm{M_{BSC}} \: \mathrm{(M_{\odot})} $",
            r"$\log_{10} \: \mathrm{R_{half-mass}\:(pc)}$ ",
            # r"$R\mathrm{_{core}}$",
            # r"$\mathrm{\Sigma_0\:\left(M_{\odot}\:pc^{-2}\right)}$",
            r"$\log_{10} \: \mathrm{\Sigma_0\:\left(M_{\odot}\:pc^{-2}\right)}$",
            # r"$\mathrm{R_{half-mass}\:(pc)}$ ",
            r"$\log_{10} \: R\mathrm{_{core}}$",
            # r"$R\mathrm{_{core}}$",
            # "$\mathrm{R_{half}\:(pc)}$ ",
        ]
        # loop through some possible plots.
        fig, ax = plt.subplots(3, 3, figsize=(7, 6), dpi=400)
        plt.subplots_adjust(hspace=0.35, wspace=0.35)
        axs = ax.ravel()
        axs[0].set_visible(False)

        subplt_lbl = [
            "",
            "$\mathrm{(a)}$",
            "$\mathrm{(b)}$",
            "$\mathrm{(c)}$",
            "$\mathrm{(d)}$",
            "$\mathrm{(e)}$",
            "$\mathrm{(f)}$",
            "$\mathrm{(g)}$",
            "$\mathrm{(h)}$",
        ]
        for i, (x, y) in enumerate(zip(x_vars, y_vars)):
            if i == 0:
                continue

            if i == 8:
                f7_x = x[0]
                f3_x = x[1]

                f7_y = np.log10(y[0])
                f3_y = np.log10(y[1])

            elif i == 2:
                f7_x = np.log10(x[0])
                f3_x = np.log10(x[1])

                f7_y = y[0]
                f3_y = y[1]
            else:
                f7_x = np.log10(x[0])
                f3_x = np.log10(x[1])

                f7_y = np.log10(y[0])
                f3_y = np.log10(y[1])

            f7_scatter = axs[i].scatter(
                f7_x,
                f7_y,
                c=f7_bes[f7_mask],
                # s=f7_half_radii,
                alpha=0.7,
                edgecolors="none",
                marker="o",
                cmap=cmap,
                linewidths=0,
                s=20,
            )
            f3_scatter = axs[i].scatter(
                f3_x,
                f3_y,
                c=f3_bes[f3_mask],
                # s=f3_half_radii,
                alpha=0.7,
                edgecolors="none",
                marker="P",
                s=20,
                cmap=cmap,
            )
            # plot index of not to fit
            dont_fit = [1, 2, 3]
            # fit them
            if i not in dont_fit:

                f7_params = stats.linregress(x=f7_x, y=f7_y)
                f3_params = stats.linregress(x=f3_x, y=f3_y)

                # f7_params, f7_pcov = curve_fit(f=lin_model, xdata=f7_x, ydata=f7_y)
                # f3_params, f3_pcov = curve_fit(f=lin_model, xdata=f3_x, ydata=f3_y)

                theory_x = np.linspace(f3_x.min() - 0.5, f3_x.max() + 0.5, 100)

                axs[i].plot(
                    theory_x,
                    lin_model(theory_x, f7_params[0], f7_params[1]),
                    lw=2,
                    color=fs70_color,
                    label="$\mathrm{{k = {:.2f} \pm {:.2f}}}$, "
                    # "\n"
                    "$\mathrm{{R^2 = {:.2f}}}$".format(
                        f7_params[0], f7_params[4], f7_params[2] ** 2
                    ),
                )
                axs[i].plot(
                    theory_x,
                    lin_model(theory_x, f3_params[0], f3_params[1]),
                    lw=2,
                    ls="--",
                    color=fs35_color,
                    label="$\mathrm{{ k = {:.2f} \pm {:.2f}}}$, "
                    # "\n"
                    "$\mathrm{{R^2 = {:.2f} }}$".format(
                        f3_params[0], f3_params[4], f3_params[2] ** 2
                    ),
                )

                # axs[i].fill_between(
                #     theory_x,
                #     lin_model(theory_x, *(f7_params - np.sqrt(np.diag(f7_pcov)))),
                #     lin_model(theory_x, *(f7_params + np.sqrt(np.diag(f7_pcov)))),
                #     lw=2,
                #     # ls="-.",
                #     color=fs70_color,
                #     alpha=0.5,
                #     edgecolor="none",
                # )

                # axs[i].fill_between(
                #     theory_x,
                #     lin_model(theory_x, *(f3_params - np.sqrt(np.diag(f3_pcov)))),
                #     lin_model(theory_x, *(f3_params + np.sqrt(np.diag(f3_pcov)))),
                #     lw=2,
                #     # ls="-.",
                #     color=fs35_color,
                #     alpha=0.5,
                #     edgecolor="none",
                # )

                # fit parameter legend
                axs[i].set_xlim(theory_x.min(), theory_x.max())
                fit = axs[i].legend(fontsize=6, loc="lower center")
                fit.get_frame().set_edgecolor("w")

            axs[i].set_xlabel(x_labels[i], labelpad=0)
            axs[i].set_ylabel(y_labels[i])

            # supblot letter label
            axs[i].text(
                0.05,
                0.95,
                subplt_lbl[i],
                horizontalalignment="left",
                verticalalignment="top",
                transform=axs[i].transAxes,
                fontsize=8,
            )
            # color bars

        f70 = mlines.Line2D(
            [],
            [],
            color="grey",
            marker="o",
            ls="",
            label=r"$f_{*} = 0.70$",
            alpha=0.8,
            markeredgewidth=0,
        )
        f35 = mlines.Line2D(
            [],
            [],
            color="grey",
            marker="P",
            ls="",
            label=r"$f_{*} = 0.35$",
            alpha=0.8,
            markeredgecolor="none",
        )
        sfe_legend = fig.legend(
            loc=(0.13, 0.69),
            title=r"$\mathrm{{t = {:.0f} \: Myr}}$".format(f3_t_myr),
            # loc="upper left",
            title_fontsize=10,
            fontsize=10,
            handles=[f70, f35],
            # bbox_to_anchor=(0.315, 0.76),
        )
        sfe_legend.get_frame().set_edgecolor("k")
        sfe_legend.get_frame().set_boxstyle("Square")
        # axs[i].add_artist(sfe_legend)
        cbar_ax = fig.add_axes([0.11, 0.80, 0.22, 0.04])
        cbar = fig.colorbar(
            f3_scatter,
            cax=cbar_ax,
            pad=0,
            orientation="horizontal",
        )
        cbar.set_alpha(0.8)
        cbar_ax.set_title(
            label="$\mathrm{Time \: of \: Formation \: (Myr)}$", fontsize=10
        )
        cbar.ax.tick_params(labelsize=7)

plt.savefig(
    os.path.expanduser(
        (
            "~/g_drive/Research/AstrophysicsSimulation/sci_plots/final/"
            "bubble_plot_dashboard.png"
        )
    ),
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.05,
)
