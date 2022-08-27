import sys

sys.path.append("../")
import numpy as np
import os
import glob
from modules.macros import filter_snapshots
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib import colors
import misc_visuals
import yt
from modules.macros import filter_snapshots, ram_fields

yt.enable_parallelism()
strt = 500
end = 500
step = 1
efficiency = 0.70
sim_run = "fs07_refine"
snap_dir = "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/{}".format(sim_run)
snap_dir = os.path.relpath("../../cosm_test_data/refine")
halo_data_directory = r"../halo_data/{}/fof_best".format(sim_run)
pop2_data_directory = r"../particle_data/pop_2_data/{}".format(sim_run)

snapshots = filter_snapshots(snap_dir, strt, end, 1)

sequence_dir = "../rendering/gas/{}/panel_t_series".format(sim_run)


if not os.path.exists(sequence_dir):
    print("# Creating new sequence directory", sequence_dir)
    os.makedirs(sequence_dir)
fs70_ds = np.loadtxt("../sci_plots/fof_time_series/fs07_refine_fof_best_113_1196.txt")[
    ::1, :
]
fail_mask = fs70_ds[:, 3] > 30
# all results are fit filtered
f7_t_sim_myr = fs70_ds[:, 0][fail_mask]
f7_redshift = fs70_ds[:, 1][fail_mask]
f7_lum_field = fs70_ds[:, 5][fail_mask]
f7_lum_bound = fs70_ds[:, 4][fail_mask]
f7_total_lum = f7_lum_field + f7_lum_bound


pop2 = filter_snapshots(pop2_data_directory, strt, end, step)
halo_ds = filter_snapshots(halo_data_directory, strt, end, step)

plt_wdth = 400
star_bins = 2000
pxl_size = (plt_wdth / star_bins) ** 2  # pc
lum_range = (4e33, 4e36)  # (2e32, 5e35)
gas_alpha = 0.5
lum_alpha = 1
cell_fields, epf = ram_fields()

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
        cmap = cm.get_cmap("Set2")
        cmap = cmap(np.linspace(0, 1, 8))

        for idx, (sn, p2, h_ds) in enumerate(zip(snapshots, pop2, halo_ds)):
            output_num_string = h_ds.split("/")[-1].split("_")[-1]
            info_file = os.path.join(sn, "info_{}.txt".format(output_num_string))
            ram_ds = yt.load(info_file, fields=cell_fields, extra_particle_fields=epf)

            # here we will use post-processed particle data from the profiler,
            # in case we want luminosities too.
            code_ctr = np.loadtxt(p2, max_rows=5)[2:5, 6]
            t_myr = np.loadtxt(p2, max_rows=2)[0, 6]
            redshift = np.loadtxt(p2, max_rows=2)[1, 6]
            field_stars = np.loadtxt(os.path.join(h_ds, "field_stars.txt"))
            bound_stars = np.loadtxt(os.path.join(h_ds, "bound_stars.txt"))
            stars = np.vstack((field_stars, bound_stars))
            star_ids = stars[:, 0]
            star_lums = stars[:, 2]
            star_masses = stars[:, 6]
            star_ages = stars[:, 1]  # Myr
            star_bes = t_myr - star_ages
            x = stars[:, 3]
            y = stars[:, 4]
            z = stars[:, 5]

            viewing_angles = [(x, y), (z, x), (y, z)]
            los = ["z", "y", "x"]
            viewing_lums = []
            viewing_gas = []
            for idx, va in enumerate(viewing_angles):
                print("Processing", los[idx])
                gas = yt.ProjectionPlot(
                    ram_ds,
                    los[idx],
                    ("gas", "density"),
                    width=(plt_wdth, "pc"),
                    center=code_ctr,
                )
                gas_frb = gas.data_source.to_frb((plt_wdth, "pc"), star_bins)
                gas_array = np.array(gas_frb["gas", "density"])

                lums, _, _ = np.histogram2d(
                    va[0],
                    va[1],
                    bins=star_bins,
                    weights=star_lums,
                    normed=False,
                    range=[
                        [-plt_wdth / 2, plt_wdth / 2],
                        [-plt_wdth / 2, plt_wdth / 2],
                    ],
                )
                lums = lums.T
                viewing_lums.append(lums)
                viewing_gas.append(gas_array)

            # time series plots
            fig = plt.figure(
                figsize=(12, 7),
                dpi=400,
                facecolor=cm.Greys_r(0),
                # constrained_layout=True,
            )
            fig.subplots_adjust(wspace=0, hspace=-0.016)
            gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.70])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
            ax4 = fig.add_subplot(gs[4:6])

            # clean up edges
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])
            ax3.set_xticklabels([])
            ax1.xaxis.set_ticks_position("none")
            ax2.xaxis.set_ticks_position("none")
            ax3.xaxis.set_ticks_position("none")
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])
            ax3.set_yticklabels([])
            ax1.yaxis.set_ticks_position("none")
            ax2.yaxis.set_ticks_position("none")
            ax3.yaxis.set_ticks_position("none")

            ax4.set_yscale("log")
            ax4.set_xlim(f7_t_sim_myr.min(), f7_t_sim_myr.max())
            ax4.set_ylim(5e33, 8e38)

            time_mask = f7_t_sim_myr <= t_myr

            ax4.plot(
                f7_t_sim_myr[time_mask],
                f7_total_lum[time_mask],
                lw=3,
                alpha=0.9,
                color=cmap[5],
                label="$\mathrm{Total}$",
            )
            ax4.plot(
                f7_t_sim_myr[time_mask],
                f7_lum_bound[time_mask],
                lw=3,
                alpha=0.9,
                color=cmap[1],
                label="$\mathrm{Cluster}$",
            )
            ax4.plot(
                f7_t_sim_myr[time_mask],
                f7_lum_field[time_mask],
                lw=3,
                alpha=0.9,
                color=cmap[0],
                label="$\mathrm{Field}$",
            )
            # after current times
            ax4.plot(
                f7_t_sim_myr[~time_mask],
                f7_total_lum[~time_mask],
                lw=3,
                alpha=0.5,
                color="grey",
            )
            ax4.plot(
                f7_t_sim_myr[~time_mask],
                f7_lum_bound[~time_mask],
                lw=3,
                alpha=0.5,
                color="grey",
            )
            ax4.plot(
                f7_t_sim_myr[~time_mask],
                f7_lum_field[~time_mask],
                lw=3,
                alpha=0.5,
                color="grey",
            )
            ax4.axvline(x=t_myr, ls="--", c="w", lw=2, alpha=0.8)
            ax4.axvspan(t_myr, f7_t_sim_myr.max(), alpha=0.1, color="grey")

            # time series plots set axis params
            ax4.set_xlabel("$\mathrm{time\:(Myr)}$")
            ax4.set_ylabel(
                (
                    r"$\log_{10}\: \mathrm{L}_{\lambda = 1500 \: \mathrm{\AA}} \:$"
                    r"$(\mathrm{erg} \:\mathrm{s}^{-1} \:\mathrm{\AA}^{-1})$"
                ),
                fontsize=12,
            )
            ax4.grid(
                visible=True,
                which="major",
                axis="y",
                ls=":",
                color="white",
                zorder=0.5,
                alpha=0.8,
            )

            # draw the line plot legend and make it square
            line_legend = ax4.legend(loc=(0.805, 0.08), borderpad=0.1)
            line_legend.get_frame().set_boxstyle("Square")
            fig.canvas.draw()
            y_labels = [i.get_text().replace("10^", "") for i in ax4.get_yticklabels()]
            ax4.set_yticklabels(y_labels)

            # three panels gas density
            ax1.imshow(
                viewing_gas[0],
                cmap="cubehelix",
                interpolation="gaussian",
                origin="lower",
                extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
                norm=LogNorm(0.008, 0.32),
                alpha=gas_alpha,
            )
            ax2.imshow(
                viewing_gas[1],
                cmap="cubehelix",
                interpolation="gaussian",
                origin="lower",
                extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
                norm=LogNorm(0.008, 0.32),
                alpha=gas_alpha,
            )
            ax3_gas_im = ax3.imshow(
                viewing_gas[2],
                cmap="cubehelix",
                interpolation="gaussian",
                origin="lower",
                extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
                norm=LogNorm(0.008, 0.32),
                alpha=gas_alpha,
            )

            # luminosity alpha
            ax1.imshow(
                viewing_lums[0] / pxl_size,
                cmap="inferno",
                # interpolation="gaussian",
                origin="lower",
                extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
                norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
                alpha=lum_alpha,
            )
            ax2.imshow(
                viewing_lums[1] / pxl_size,
                cmap="inferno",
                # interpolation="gaussian",
                origin="lower",
                extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
                norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
                alpha=lum_alpha,
            )
            ax3_lum_im = ax3.imshow(
                viewing_lums[2] / pxl_size,
                cmap="inferno",
                # interpolation="gaussian",
                origin="lower",
                extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
                norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
                alpha=lum_alpha,
            )

            # add scale
            scale = patches.Rectangle(
                xy=(plt_wdth / 2 * 0.25, -plt_wdth / 2 * 0.75),
                width=plt_wdth / 2 * 0.5,
                height=0.03 * plt_wdth / 2,
                linewidth=0,
                edgecolor="white",
                facecolor="white",
            )
            ax3.text(
                plt_wdth / 2 * 0.50,
                -plt_wdth / 2 * 0.87,
                r"$\mathrm{{{:.0f} \: pc}}$".format(plt_wdth / 2 * 0.5),
                ha="center",
                va="center",
                color="white",
                # fontproperties=leg_font,
                fontsize=14,
            )
            ax3.add_patch(scale)

            # add the luminosity color bar
            cbar_ax = fig.add_axes([0.130, 0.30, 0.20, 0.025])
            cbar = fig.colorbar(
                ax3_lum_im, cax=cbar_ax, pad=-1, orientation="horizontal"
            )
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")
            cbar.ax.xaxis.set_tick_params(pad=2, labelsize=10)

            cbar_ax.set_title(
                r"$\mathrm{Surface\:Brightness},$"
                r"$\mathrm{\lambda = 1500 \: \AA \:}$"
                "\n"
                r"$\log_{10}\:\mathrm{\left(erg\:\:s^{-1}\:\AA^{-1}\:pc^{-2}\right)}$",
                fontsize=12,
            )
            fig.canvas.draw()
            x_labels = [
                i.get_text().replace("10^", "") for i in cbar_ax.get_xticklabels()
            ]
            cbar_ax.set_xticklabels(x_labels)

            # add the gas color bar
            gas_cbar_gas_ax = fig.add_axes([0.130, 0.26, 0.20, 0.025])
            gas_cbar = fig.colorbar(
                ax3_gas_im, cax=gas_cbar_gas_ax, pad=0, orientation="horizontal"
            )
            gas_cbar.ax.xaxis.set_tick_params(pad=2, labelsize=10)
            gas_cbar.set_label(
                label=r"$\mathrm{Gas\:Density\:\log_{10}\:(g \: cm^{-2})}$",
                fontsize=12,
                labelpad=1,
            )

            fig.canvas.draw()
            x_labels = [
                i.get_text().replace("10^", "")
                for i in gas_cbar_gas_ax.get_xticklabels()
            ]
            gas_cbar_gas_ax.set_xticklabels(x_labels)

            # add time and redshift
            fig.text(
                0.18,
                0.17,
                (
                    "$\quad\:\: f_{{*}} = {:.2f}$"
                    "\n"
                    r"$\mathrm{{z = {:.2f} }}$"
                    "\n"
                    r"$\mathrm{{t = {:.2f} \: Myr}}$"
                ).format(efficiency, redshift, t_myr),
                ha="left",
                va="top",
                color="white",
                fontsize=14,
                bbox={
                    "boxstyle": "Square",
                    # have control over edge alpha and face alpha
                    "facecolor": colors.to_rgba("black")[:-1] + (0.5,),
                    "linewidth": 1,
                    "edgecolor": "white",
                    "pad": 0.42,
                },
            )

            # add fancy axes indicators
            ax1_inset = ax1.inset_axes([0.03, 0.03, 0.15, 0.15])
            misc_visuals.xy_ax_indicator(ax1_inset)
            ax1_inset.set_alpha(0)

            ax2_inset = ax2.inset_axes([0.03, 0.03, 0.15, 0.15])
            misc_visuals.xz_ax_indicator(ax2_inset)
            ax2_inset.set_alpha(0)

            ax3_inset = ax3.inset_axes([0.03, 0.03, 0.15, 0.15])
            misc_visuals.yz_ax_indicator(ax3_inset)
            ax3_inset.set_alpha(0)

            output_path = os.path.join(
                sequence_dir, "lum_series_{}.png".format(output_num_string)
            )
            # plt.show()
            plt.savefig(
                os.path.expanduser(output_path),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.05,
            )

            print(">Saved:", output_path)
