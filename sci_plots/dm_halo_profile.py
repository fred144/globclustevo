import sys

sys.path.append("..")
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
from modules.macros import filter_snapshots, sci_notation
import matplotlib.lines as mlines
from modules.match_t_sims import find_matching_time, get_snapshots
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from modules.profiles.profile_functions import projected_surf_densities
from modules.profiles.profile_models import modified_king_model
from modules.macros import filter_snapshots, characterisitc_mass
from matplotlib import colors
import matplotlib.patches as patches

fs070 = filter_snapshots("../particle_data/pop_2_data/fs07_refine", 115, 1195, 1)
fs035 = filter_snapshots("../particle_data/pop_2_data/fs035_ms10", 154, 1364, 1)
# find matching fs = 0.35 snapshots in terms of time to fs = 0.70
# smaller goes fist
# in general, use the simulation with more snapshots as a lookup table and match
# the ones with less
_, f7_matched_nums = find_matching_time(sequence=fs035, look_up_sequence=fs070)

fs070_dat_dir = r"../dm/fs07_refine/dm_hop"
fs035_dat_dir = r"../dm/fs035_ms10/dm_hop"

fs035_matched = filter_snapshots(fs035_dat_dir, 154, 1364, 1)[100:281:40]

fs070_matched = get_snapshots(
    snapshot_file_list=filter_snapshots(fs070_dat_dir, 115, 1195, 1),
    get_list=f7_matched_nums,
    verbose=False,
)[10:281:40]
# 100
bins = 800
plt_rad = 2000  # pc
pxl_size = (plt_rad * 2 / bins) ** 2
profile_plot_bins = 25

f3_series = np.loadtxt("../dm/fs07_refine/dm_hop/fs070_dm_halo_evo.txt")
eff = 0.7
for i, f3 in enumerate(fs070_matched):
    try:
        f3_out_num = int(f3.split("/")[-1].split("_")[-1])
    except:
        f3_out_num = int(f3.split("\\")[-1].split("_")[-1])

    f3_dm_part_data = np.loadtxt(os.path.join(f3, "dm_data.txt"))
    f3_time = float(f3_series[:, 1][f3_series[:, 0] == f3_out_num])
    f3_vir_rad = float(f3_series[:, 4][f3_series[:, 0] == f3_out_num])

    f3_mass = f3_dm_part_data[:, 0]
    f3_x = f3_dm_part_data[:, 1]
    f3_y = f3_dm_part_data[:, 2]

    xy_mass, _, _ = np.histogram2d(
        f3_x,
        f3_y,
        bins=bins,
        weights=f3_mass,
        normed=False,
        range=[[-plt_rad, plt_rad], [-plt_rad, plt_rad]],
    )
    xy_mass = xy_mass.T

    with plt.style.context("dark_background"):
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
                ncols=1,
                figsize=(6.0, 6.0),
                dpi=400,
                sharex="row",
                sharey="row",
                facecolor=cm.viridis(0),
            )

            xy = ax.imshow(
                xy_mass,
                cmap="viridis",
                interpolation="gaussian",
                origin="lower",
                extent=[-plt_rad, plt_rad, -plt_rad, plt_rad],
                norm=LogNorm(),
            )
            ax.scatter([0], [0], marker="+", alpha=0.5, color="red", s=40)
            ax.set_facecolor(cm.viridis(0))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position("none")
            ax.yaxis.set_ticks_position("none")

            ax.set_xlim(left=-2000, right=1300)
            ax.set_ylim(bottom=-2000, top=1300)

            # add scale
            master_scale = patches.Rectangle(
                xy=(0, -500),
                width=500,
                height=10,
                linewidth=0,
                edgecolor="white",
                facecolor="white",
                # transform=ax.transAxes,
            )
            ax.text(
                250,
                -600,
                r"$\mathrm{ 500 pc}$",
                ha="center",
                va="center",
                color="white",
                # rotation=270,
                # fontproperties=leg_font,
                # transform=ax.transAxes,
            )
            ax.add_patch(master_scale)

            # surface density profile
            with plt.rc_context(
                {
                    "xtick.labelsize": 7,
                    "ytick.labelsize": 7,
                    "font.size": 10,
                }
            ):
                prof = ax.inset_axes([0.10, 0.10, 0.35, 0.35])
                dm_eff = ax.inset_axes([0.10, 0.47, 0.35, 0.25])

            # plot dm efficiency
            mask = f3_series[:, 1] <= f3_time
            dm_eff.plot(
                f3_series[:, 1][mask],
                (f3_series[:, 5] / f3_series[:, 3])[mask],
                color="white",
            )
            dm_eff.set(
                ylabel="$\mathrm{\log_{10} \:M_* / M_{DM}}$",
                yscale="log",
            )
            dm_eff.axhline(
                y=(f3_series[:, 5] / f3_series[:, 3])[mask][-1], ls=":", alpha=0.5
            )

            dm_eff.set_xlim(right=f3_series[:, 1].max())
            dm_eff.set_title("$\mathrm {t \: (Myr)}$", fontsize=10)
            dm_eff.xaxis.tick_top()
            dm_eff.patch.set_alpha(0.2)
            dm_eff.text(
                0.80,
                0.20,
                # f3_series[:, 1].max() - 100,
                # (f3_series[:, 5] / f3_series[:, 3])[mask][-1],
                "{:.4f}".format((f3_series[:, 5] / f3_series[:, 3])[mask][-1]),
                transform=dm_eff.transAxes,
                fontsize=7,
                va="top",
            )
            dm_eff.set_ylim(
                (f3_series[:, 5] / f3_series[:, 3]).min(),
                (f3_series[:, 5] / f3_series[:, 3]).max() * 1.1,
            )
            fig.canvas.draw()
            y_labels = [
                i.get_text().replace("10^", "") for i in dm_eff.get_yticklabels()
            ]
            dm_eff.set_yticklabels(y_labels)

            r, rho, err, _, _, _, half_r = projected_surf_densities(
                x_coord=f3_x,
                y_coord=f3_y,
                lums=np.ones_like(f3_x),
                masses=f3_mass,
                radius=f3_vir_rad,
                num_bins=profile_plot_bins,
                log_bins=True,
                dr=None,
                calc_half_r=True,
            )

            # fit the error bars
            fit_params, cov_matrix = curve_fit(
                f=modified_king_model,
                xdata=r,
                ydata=rho,
                sigma=err,
                absolute_sigma=True,
                # p0=[1e5, 0.2, 2, 10],
                # bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
            )
            sigma_naught = fit_params[0]
            fit_r_c = fit_params[1]
            fit_alpha = fit_params[2]
            fit_sigma_bg = fit_params[3]
            core_mass = characterisitc_mass(f3_x, f3_y, f3_mass, fit_r_c)
            half_mass = characterisitc_mass(f3_x, f3_y, f3_mass, half_r)
            theory_r = np.geomspace(
                0.1, f3_vir_rad, 200, endpoint=False
            )  # smooth version
            theory_rho = modified_king_model(theory_r, *fit_params)
            theory_rho_sig = modified_king_model(
                theory_r, *fit_params - 2 * np.sqrt(np.diag(cov_matrix))
            )

            prof.errorbar(
                r,
                rho,
                yerr=err,
                fmt="o",
                capsize=3,
                capthick=1,
                elinewidth=1,
                ms=3,
                alpha=1,
                c="w",
            )

            prof.plot(
                theory_r,
                theory_rho,
                ls="--",
                linewidth=2,
                alpha=0.6,
                zorder=3,
                c="w",
            )

            prof.set_xscale("log")
            prof.set_yscale("log")
            prof.set_ylabel(
                r"$\mathrm{\log_{10}\; \rho \:\:(M_{\odot}\:pc^{-2})}$",
            )
            prof.set_xlabel(
                r"$ \mathrm{\log_{10}\;  R \:(pc)}$",
            )

            fit_results = (
                r"$R_{{\mathrm{{vir}}}} = {:.2f} \: \mathrm{{pc}}$"
                "\n"
                r"$R_{{\mathrm{{core}}}} = {:.2f} \: \mathrm{{pc}}$"
                "\n"
                r"$R_{{\mathrm{{half}}}} = {:.2f} \: \mathrm{{pc}}$"
                "\n"
                r"$\alpha = {:.2f} $"
                "\n"
                r"$\Sigma_0={}\:\left(\mathrm{{M_{{\odot}}{{pc^{{2}}}}}}\right)$"
                "\n"
                r"$M_{{\mathrm{{core}}}} = {} \: \mathrm{{M}}_{{\odot}}$"
                "\n"
                r"$M_{{\mathrm{{half}}}} = {} \: \mathrm{{M}}_{{\odot}}$"
            ).format(
                f3_vir_rad,
                fit_r_c,
                half_r,
                fit_alpha,
                sci_notation(2, sigma_naught),
                sci_notation(2, core_mass),
                sci_notation(2, half_mass),
            )
            ax.text(
                0.58,
                0.05,
                fit_results,
                fontsize=11,
                ha="left",
                va="bottom",
                color="white",
                # fontproperties=leg_font,
                transform=ax.transAxes,
                bbox={
                    "boxstyle": "Square",
                    # have control over edge alpha and face alpha
                    "facecolor": colors.to_rgba("black")[:-1] + (0.2,),
                    "linewidth": 0.8,
                    "edgecolor": "white",
                    # "pad": 0.40,
                },
            )
            ax.text(
                0.05,
                0.95,
                "$f_* = {:.2f}$"
                "\n"
                "$\mathrm{{t = {:.2f} \: Myr}}$".format(eff, f3_time),
                ha="left",
                va="top",
                color="white",
                # fontproperties=leg_font,
                transform=ax.transAxes,
                bbox={
                    "boxstyle": "Square",
                    # have control over edge alpha and face alpha
                    "facecolor": colors.to_rgba("black")[:-1] + (0.2,),
                    "linewidth": 0.8,
                    "edgecolor": "white",
                    "pad": 0.40,
                },
            )
            prof.set_facecolor("black")
            prof.set_ylim(0.01, 8e3)
            prof.patch.set_alpha(0.2)
            fig.canvas.draw()
            # x_labels = [i.get_text().replace("10^", "") for i in prof.get_xticklabels()]
            # y_labels = [i.get_text().replace("10^", "") for i in prof.get_yticklabels()]
            # prof.set_xticklabels(x_labels)
            # prof.set_yticklabels(y_labels)

            # color bar
            cbar_ax = fig.add_axes([0.60, 0.77, 0.265, 0.02])
            cbar = fig.colorbar(xy, cax=cbar_ax, pad=0, orientation="horizontal")
            cbar.ax.xaxis.set_tick_params(pad=2, labelsize=8)
            cbar.ax.xaxis.set_ticks_position("top")
            fig.canvas.draw()
            x_labels = [
                i.get_text().replace("10^", "") for i in cbar_ax.get_xticklabels()
            ]
            cbar_ax.set_xticklabels(x_labels)

            cbar_ax.set_title(
                "$\mathrm{\log_{10} \: DM \: Mass \: (M_{\odot})}$",
                # labelpad=2,
                # fontproperties=leg_font,
                fontsize=11,
            )

            plt.show()
