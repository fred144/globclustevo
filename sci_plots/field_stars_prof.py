import sys

sys.path.append("../")
import numpy as np
import os
from modules.macros import filter_snapshots, characterisitc_mass, sci_notation
from modules.profiles.profile_functions import projected_surf_densities
from modules.profiles.profile_models import modified_king_model
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from modules.match_t_sims import find_matching_time, get_snapshots

runsavepath = "../rendering/luminosity/fs07_refine/gc_tracking"
if not os.path.exists(runsavepath):
    print("# Creating new sequence directory", runsavepath)
    os.makedirs(runsavepath)


f7_strt = 113
f7_end = 1110
f3_strt = 154
f3_end = 1316
step = 1

f7_pop2 = filter_snapshots(
    r"../particle_data/pop_2_data/fs07_refine", f7_strt, f7_end, step
)
f7_halo_ds = filter_snapshots(
    r"../halo_data/fs07_refine/fof_best", f7_strt, f7_end, step
)
# matched snapshots

f3_pop2 = filter_snapshots(
    r"../particle_data/pop_2_data/fs035_ms10", f3_strt, f3_end, step
)
f3_halo = filter_snapshots(r"../halo_data/fs035_ms10/fof_best", f3_strt, f3_end, step)

f7_pop2_matched, f7_matched_nums = find_matching_time(
    sequence=f3_pop2, look_up_sequence=f7_pop2
)
f7_halo_matched = get_snapshots(snapshot_file_list=f7_halo_ds, get_list=f7_matched_nums)

# sampple idxs
prof_start = 1160
prof_end = 1161
prof_step = 1

fs070_p2 = f7_pop2_matched[prof_start:prof_end:prof_step]
fs070_ds = f7_halo_matched[prof_start:prof_end:prof_step]
fs035_p2 = f3_pop2[prof_start:prof_end:prof_step]
fs035_ds = f3_halo[prof_start:prof_end:prof_step]

# change font for entire module
# mpl.rc("font", family="serif")


profile_plot_bins = 40
radius = 200
star_bins = 200
pxl_size = (radius * 2 / star_bins) ** 2
cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))
efficiencies = ["$f_{*} = 0.70$", "$f_{*} = 0.35$"]
eff_lcolor = [cmap[0], cmap[2]]
eff_errcol = [cmap[1], cmap[3]]

#%%
# run thorugh each matched pair for time series
for eff_p2, eff_ds in zip(zip(fs070_p2, fs035_p2), zip(fs070_ds, fs035_ds)):

    # with plt.style.context("dark_background"):
    with plt.rc_context(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
        }
    ):
        fig, ax = plt.subplots(
            nrows=5,
            ncols=3,
            gridspec_kw={"height_ratios": [1, 0.7, 0.32, 1, 0.7]},
            sharex="row",
            sharey="row",
            figsize=(8, 11.5),
            dpi=300,
            # facecolor=cm.Greys_r(0),
        )
        for h in range(3):
            ax[2, h].set_visible(False)
        # used to loop through two times to get stacked 2*3s
        for i, (p2, ds) in enumerate(zip(eff_p2, eff_ds)):
            # correction for invisible row
            if i == 0:
                row_handler = 0
                hndlr_2 = 1
            else:
                row_handler = 2 * i + 1
                hndlr_2 = 2

            plt.subplots_adjust(hspace=-0.033, wspace=0)
            print(i)
            print(p2, ds)
            eff_label = efficiencies[i]
            line_color = eff_lcolor[i]
            scatter_color = eff_errcol[i]

            output_num = int(ds.split("/")[-1].split("_")[-1])
            t_myr = np.loadtxt(p2, max_rows=2)[0, 6]
            redshift = np.loadtxt(p2, max_rows=2)[1, 6]

            field_stars = np.loadtxt(os.path.join(ds, "field_stars.txt"))
            x = field_stars[:, 3]
            y = field_stars[:, 4]
            z = field_stars[:, 5]
            field_lums = field_stars[:, 2]
            field_masses = field_stars[:, 6]
            field_ages = field_stars[:, 1]
            field_bes = t_myr - field_ages
            xy_lums, _, _ = np.histogram2d(
                x,
                y,
                bins=star_bins,
                weights=field_lums,
                normed=False,
                range=[[-radius, radius], [-radius, radius]],
            )

            xz_lums, _, _ = np.histogram2d(
                x,
                z,
                bins=star_bins,
                weights=field_lums,
                normed=False,
                range=[[-radius, radius], [-radius, radius]],
            )

            yz_lums, _, _ = np.histogram2d(
                y,
                z,
                bins=star_bins,
                weights=field_lums,
                normed=False,
                range=[[-radius, radius], [-radius, radius]],
            )

            xy_lums = xy_lums.T
            xz_lums = xy_lums.T
            yz_lums = yz_lums.T

            # plot the luminosity projection for 3 viewing angles

            xy = ax[row_handler, 0].imshow(
                xy_lums / pxl_size,
                cmap="inferno",
                # interpolation="gaussian",
                origin="lower",
                extent=[-radius, radius, -radius, radius],
                norm=LogNorm(vmin=2e32, vmax=7e34),
            )
            ax[row_handler, 0].set_facecolor(cm.Greys_r(0))
            xz = ax[row_handler, 1].imshow(
                xz_lums / pxl_size,
                cmap="inferno",
                # interpolation="gaussian",
                origin="lower",
                extent=[-radius, radius, -radius, radius],
                norm=LogNorm(vmin=2e32, vmax=7e34),
            )
            ax[row_handler, 1].set_facecolor(cm.Greys_r(0))
            yz = ax[row_handler, 2].imshow(
                yz_lums / pxl_size,
                cmap="inferno",
                # interpolation="gaussian",
                origin="lower",
                extent=[-radius, radius, -radius, radius],
                norm=LogNorm(vmin=2e32, vmax=7e34),
            )
            ax[row_handler, 2].set_facecolor(cm.Greys_r(0))

            # handle the axes lines and ticks
            for t in range(3):
                ax[row_handler, t].spines["bottom"].set_color("white")
                ax[row_handler, t].spines["top"].set_color("white")
                if t != 2:
                    ax[row_handler, t].spines["right"].set_color("white")
                if t != 0:
                    ax[row_handler, t].spines["left"].set_color("white")
                    ax[row_handler, t].tick_params(colors="white")
                    # ax[row_handler, t].xaxis.set_tick_params(labeltop="on")

            # calculated projected surface densities for each three projections
            xy_r, xy_rho, xy_err, _, _, xy_half_r, _ = projected_surf_densities(
                x_coord=x,
                y_coord=y,
                lums=field_lums,
                masses=field_masses,
                radius=radius,
                num_bins=profile_plot_bins,
                log_bins=True,
                dr=None,
                calc_half_r=True,
            )

            xz_r, xz_rho, xz_err, _, _, _, xz_half_r = projected_surf_densities(
                x_coord=x,
                y_coord=z,
                lums=field_lums,
                masses=field_masses,
                radius=radius,
                num_bins=profile_plot_bins,
                log_bins=True,
                dr=None,
                calc_half_r=True,
            )

            yz_r, yz_rho, yz_err, _, _, _, yz_half_r = projected_surf_densities(
                x_coord=y,
                y_coord=z,
                lums=field_lums,
                masses=field_masses,
                radius=radius,
                num_bins=profile_plot_bins,
                log_bins=True,
                dr=None,
                calc_half_r=True,
            )

            # fit them
            fit_params, cov_matrix = curve_fit(
                f=modified_king_model,
                xdata=xy_r,
                ydata=xy_rho,
                sigma=xy_err,
                absolute_sigma=True,
                p0=[1e5, 0.2, 2, 10],
                bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
            )
            xy_sigma_naught = fit_params[0]
            xy_fit_r_c = fit_params[1]
            xy_fit_alpha = fit_params[2]
            xy_fit_sigma_bg = fit_params[3]
            xy_core_mass = characterisitc_mass(x, y, field_masses, xy_fit_r_c)
            xy_theory_r = np.geomspace(
                0.15, radius, 200, endpoint=False
            )  # smooth version
            xy_theory_rho = modified_king_model(xy_theory_r, *fit_params)
            xy_plot_label = (
                r"$R_{{\mathrm{{core}}}} = {:.1f} \: \mathrm{{pc}}$"
                "\n"
                r"$\alpha = {:.1f} $"
                "\n"
                r"$\Sigma_0 = {} \: \mathrm{{{{M_{{\odot}}}}{{pc^{{-2}}}}}}$"
                "\n"
                r"$M_{{\mathrm{{core}}}} = {} \: \mathrm{{M}}_{{\odot}}$"
                "\n"
                r"$R_{{\mathrm{{half}}}} = {:.1f} \: \mathrm{{pc}}$"
            ).format(
                xy_fit_r_c,
                xy_fit_alpha,
                sci_notation(1, xy_sigma_naught),
                sci_notation(1, xy_core_mass),
                xy_half_r,
            )

            fit_params, cov_matrix = curve_fit(
                f=modified_king_model,
                xdata=xz_r,
                ydata=xz_rho,
                sigma=xz_err,
                absolute_sigma=True,
                p0=[1e5, 0.2, 2, 10],
                bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
            )
            xz_sigma_naught = fit_params[0]
            xz_fit_r_c = fit_params[1]
            xz_fit_alpha = fit_params[2]
            xz_fit_sigma_bg = fit_params[3]
            xz_core_mass = characterisitc_mass(x, z, field_masses, xz_fit_r_c)
            xz_theory_r = np.geomspace(
                0.15, radius, 200, endpoint=False
            )  # smooth version
            xz_theory_rho = modified_king_model(xz_theory_r, *fit_params)
            xz_plot_label = (
                r"$R_{{\mathrm{{core}}}} = {:.1f} \: \mathrm{{pc}}$"
                "\n"
                r"$\alpha = {:.1f} $"
                "\n"
                r"$\Sigma_0 = {} \: \mathrm{{{{M_{{\odot}}}}{{pc^{{-2}}}}}}$"
                "\n"
                r"$M_{{\mathrm{{core}}}} = {} \: \mathrm{{M}}_{{\odot}}$"
                "\n"
                r"$R_{{\mathrm{{half}}}} = {:.1f} \: \mathrm{{pc}}$"
            ).format(
                xz_fit_r_c,
                xz_fit_alpha,
                sci_notation(1, xz_sigma_naught),
                sci_notation(1, xz_core_mass),
                xz_half_r,
            )

            fit_params, cov_matrix = curve_fit(
                f=modified_king_model,
                xdata=yz_r,
                ydata=yz_rho,
                sigma=yz_err,
                absolute_sigma=True,
                p0=[1e5, 0.2, 2, 10],
                bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
            )
            yz_sigma_naught = fit_params[0]
            yz_fit_r_c = fit_params[1]
            yz_fit_alpha = fit_params[2]
            yz_fit_sigma_bg = fit_params[3]
            yz_core_mass = characterisitc_mass(y, z, field_masses, yz_fit_r_c)
            yz_theory_r = np.geomspace(
                0.15, radius, 200, endpoint=False
            )  # smooth version
            yz_theory_rho = modified_king_model(yz_theory_r, *fit_params)
            yz_plot_label = (
                r"$R_{{\mathrm{{core}}}} = {:.1f} \: \mathrm{{pc}}$"
                "\n"
                r"$\alpha = {:.1f} $"
                "\n"
                r"$\Sigma_0 = {} \: \mathrm{{{{M_{{\odot}}}}{{pc^{{-2}}}}}}$"
                "\n"
                r"$M_{{\mathrm{{core}}}} = {} \: \mathrm{{M}}_{{\odot}}$"
                "\n"
                r"$R_{{\mathrm{{half}}}} = {:.1f} \: \mathrm{{pc}}$"
            ).format(
                yz_fit_r_c,
                yz_fit_alpha,
                sci_notation(1, yz_sigma_naught),
                sci_notation(1, yz_core_mass),
                yz_half_r,
            )

            # plot the error bars and theory curves
            leg_font = font_manager.FontProperties(family="serif", math_fontfamily="cm")
            # leg_font
            ax[2 * i + hndlr_2, 0].errorbar(
                xy_r,
                xy_rho,
                yerr=xy_err,
                fmt="o",
                ms=4,
                capsize=5,
                capthick=3,
                elinewidth=3,
                alpha=0.9,
                c=scatter_color,
            )
            ax[2 * i + hndlr_2, 0].plot(
                xy_theory_r,
                xy_theory_rho,
                linewidth=4,
                label=xy_plot_label,
                alpha=0.8,
                zorder=3,
                color=line_color,
            )
            ax[2 * i + hndlr_2, 0].legend(
                loc="lower left",
                framealpha=0.6,
                edgecolor="k",
                fontsize=8,
                prop=leg_font,
                handlelength=0,
                handletextpad=0,
            )

            ax[2 * i + hndlr_2, 1].errorbar(
                xz_r,
                xz_rho,
                yerr=xz_err,
                fmt="o",
                ms=4,
                capsize=5,
                capthick=3,
                elinewidth=3,
                c=scatter_color,
                alpha=0.9,
            )
            ax[2 * i + hndlr_2, 1].plot(
                xz_theory_r,
                xz_theory_rho,
                linewidth=4,
                label=xz_plot_label,
                alpha=0.8,
                zorder=3,
                color=line_color,
            )
            ax[2 * i + hndlr_2, 1].legend(
                loc="lower left",
                framealpha=0.6,
                edgecolor="k",
                fontsize=8,
                prop=leg_font,
                handlelength=0,
                handletextpad=0,
            )

            ax[2 * i + hndlr_2, 2].errorbar(
                yz_r,
                yz_rho,
                yerr=yz_err,
                fmt="o",
                ms=4,
                capsize=5,
                capthick=3,
                elinewidth=3,
                alpha=0.9,
                c=scatter_color,
            )
            ax[2 * i + hndlr_2, 2].plot(
                yz_theory_r,
                yz_theory_rho,
                linewidth=4,
                label=yz_plot_label,
                alpha=0.8,
                zorder=3,
                color=line_color,
            )
            ax[2 * i + hndlr_2, 2].legend(
                loc="lower left",
                framealpha=0.6,
                edgecolor="k",
                fontsize=8,
                prop=leg_font,
                handlelength=0,
                handletextpad=0,
            )

            # edit ticks, remove the numbers
            ax[row_handler, 0].axes.yaxis.set_ticklabels([])
            ax[row_handler, 0].axes.xaxis.set_ticklabels([])

            # axes labels by using the subplot axes
            ax[2 * i + hndlr_2, 0].set_ylabel(
                r"$ \mathrm{\Sigma} \: (\mathrm{M}_{\odot} \; \mathrm{pc}^{-2})$",
                fontproperties=leg_font,
                fontsize=12,
            )
            ax[2 * i + hndlr_2, 1].set_xlabel(
                r"$ \mathrm{R} \: (\mathrm{pc})$",
                fontproperties=leg_font,
                fontsize=12,
                labelpad=0,
            )
            # ax[row_handler, 1].set_xlabel(
            #     r"$ \mathrm{R} \: (\mathrm{pc})$",
            #     fontproperties=leg_font,
            #     fontsize=12,
            #     labelpad=0,
            # )

            # edit bottom row limits
            ax[2 * i + hndlr_2, 0].set_xlim(left=0.15, right=radius * 1.55)
            ax[2 * i + hndlr_2, 0].set_ylim(bottom=3e-1, top=2.5e2)
            ax[2 * i + hndlr_2, 0].set_xscale("log")
            ax[2 * i + hndlr_2, 0].set_yscale("log")

            axes_ind = [("$x$", "$y$"), ("$x$", "$z$"), ("$y$", "$z$")]
            for n, l in enumerate(axes_ind):
                # remove x tick marks for the top row
                ax[row_handler, n].tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    top=False,
                    labelbottom=False,
                )
                # add axes indicators for the top rows
                ax[row_handler, n].text(
                    -radius * 0.55,
                    -radius * 0.9,
                    l[0],
                    size=7,
                    ha="center",
                    va="center",
                    color="white",
                )
                ax[row_handler, n].text(
                    -radius * 0.9,
                    -radius * 0.55,
                    l[1],
                    size=7,
                    ha="center",
                    va="center",
                    color="white",
                )

                ax[row_handler, n].arrow(
                    -radius * 0.9,
                    -radius * 0.9,
                    radius * 0.3,
                    0,
                    head_width=5,
                    head_length=3,
                    linewidth=0.5,
                    color="w",
                    length_includes_head=True,
                )
                ax[row_handler, n].arrow(
                    -radius * 0.9,
                    -radius * 0.9,
                    0,
                    radius * 0.3,
                    head_width=5,
                    head_length=3,
                    linewidth=0.5,
                    color="w",
                    length_includes_head=True,
                )
                # grids for the bottom row
                ax[2 * i + hndlr_2, n].grid(
                    visible=True,
                    which="both",
                    axis="y",
                    ls="--",
                    color="grey",
                    zorder=0.5,
                    alpha=0.8,
                )

            # add a scale
            ax[row_handler, 0].set_ylabel(
                (r"$ \mathrm{100 \: pc}$" "\n"),
                fontproperties=leg_font,
                fontsize=12,
            )

            rect = patches.Rectangle(
                xy=(-radius * 1.2, -50),
                width=8,
                height=100,
                linewidth=0,
                edgecolor="black",
                facecolor="black",
                clip_on=False,
            )
            ax[row_handler, 0].add_patch(rect)
            ax[row_handler, 0].set_xlim(-radius, radius)
            ax[row_handler, 1].set_xlim(-radius, radius)
            ax[row_handler, 2].set_xlim(-radius, radius)

            # add time and redshift
            props = dict(
                boxstyle="round",
                facecolor="black",
                alpha=0.5,
                linewidth=0.8,
                edgecolor="white",
            )
            ax[row_handler, 0].text(
                -radius * 0.9,
                radius * 0.9,
                (
                    r"$\mathrm{{t = {:.1f} \: Myr}}$" "\n" r"$\mathrm{{z = {:.1f} }}$"
                ).format(t_myr, redshift),
                size=12,
                ha="left",
                va="top",
                color="white",
                fontproperties=leg_font,
                bbox=props,
            )

            eff_prop = dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.5,
                linewidth=0.8,
                edgecolor="grey",
            )
            # add efficiency label
            ax[row_handler, 1].text(
                0.5,
                0.87,
                eff_label,
                size=12,
                ha="center",
                va="bottom",
                color="white",
                transform=ax[row_handler, 1].transAxes,
                fontproperties=leg_font,
                bbox=props,
            )

            # add a histogram of ages inside the cluster
            # [left, bottom, width, height]
            with plt.style.context("dark_background"):
                if i == 0:
                    # top inset
                    ax_inset = ax[i, 2].inset_axes([0.55, 0.20, 0.4, 0.4])

                    # 0.70 numbers
                    # ax[2 * i + hndlr_2, 0].axes.xaxis.set_ticklabels([])
                    # ax[2 * i + hndlr_2, 1].axes.xaxis.set_ticklabels([])
                    # ax[2 * i + hndlr_2, 2].axes.xaxis.set_ticklabels([])
                else:
                    # bottom inset
                    ax_inset = ax[3, 2].inset_axes([0.55, 0.20, 0.4, 0.4])

                    # plot only for the bottom row

                ax_inset.patch.set_alpha(0.5)
                bins = np.linspace(300, 650, 20)
                ax_inset.hist(
                    field_bes,
                    bins,
                    weights=field_masses,
                    alpha=0.6,
                    color="w",
                    edgecolor="black",
                )
                ax_inset.axvline(x=t_myr, ls="--", color="white", lw=1)
                ax_inset.tick_params(labelsize=5)
                ax_inset.set_xlabel(
                    "$\mathrm{Star \: Birth \: (Myr)}$",
                    fontproperties=leg_font,
                    fontsize=7,
                    labelpad=0,
                )
                ax_inset.set_ylabel(
                    r"$\mathrm{Field \: Mass \: (M_{\odot})}$",
                    fontproperties=leg_font,
                    fontsize=7,
                    labelpad=0,
                )
                ax_inset.text(
                    t_myr + 10,
                    2e3,
                    r"$\mathrm{t_{sim}}$",
                    fontproperties=leg_font,
                    va="center",
                    rotation=270,
                    fontsize=7,
                )
                ax_inset.set_yscale("log")
                # ax_inset.set_xlim("log")
                ax_inset.set_xlim(300, 700)
                ax_inset.set_ylim(1, 5e5)

        # add the luminosity color bar
        # fig.subplots_adjust(wspace=0, hspace=0, bottom=0.1)
        # [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.125, 0.8525, 0.775, 0.008])
        cbar = fig.colorbar(xz, cax=cbar_ax, pad=0, orientation="horizontal")
        cbar_label = (
            r"$\mathrm{\log_{10}\:\:Surface\:Brightness}"
            r", \mathrm{\lambda = 1500 \: \AA \:}"
            r"\mathrm{\left(erg \:\: s^{-1} \: \AA^{-1} \: pc^{-2} \right)} $"
        )
        cbar.set_label(
            label=cbar_label,
            fontsize=12,
            labelpad=5,
            fontproperties=leg_font,
        )
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.xaxis.set_tick_params(pad=2, labelsize=8)
        # tick label mod
        fig.canvas.draw()
        x_labels = [i.get_text().replace("10^", "") for i in cbar_ax.get_xticklabels()]
        cbar_ax.set_xticklabels(x_labels)
        # save_name = os.path.join(runsavepath, "tracked_{}".format(output_num))
        plt.subplots_adjust(hspace=-0.18, wspace=0)

        plt.savefig(
            os.path.expanduser(
                (
                    "~/g_drive/Research/AstrophysicsSimulation/sci_plots/final/"
                    "field_density_profile.png"
                )
            ),
            dpi=500,
            bbox_inches="tight",
            pad_inches=0.05,
        )
