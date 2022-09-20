import sys

sys.path.append("../")
import numpy as np
import os
import glob
from modules.match_t_sims import find_matching_time, get_snapshots
from modules.macros import filter_snapshots
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib import colors
import misc_visuals
import yt
from modules.macros import filter_snapshots, ram_fields
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

yt.enable_parallelism()

# f7_snap_range = (599, 1296)
# f3_snap_range = (179, 1469)

f7_snap_range = (500, 500)
f3_snap_range = (500, 500)

fs070_dir = "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine"
fs035_dir = "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs035_ms10"

# fs070_dir = os.path.relpath("../../cosm_test_data/refine"),
# fs035_dir =  os.path.relpath("../../cosm_test_data/fs035_ms10/"),

fs070_snap_dir = filter_snapshots(
    fs070_dir,
    f7_snap_range[0],
    f7_snap_range[1],
    1,
)
fs035_snap_dir = filter_snapshots(
    fs035_dir,
    f7_snap_range[0],
    f7_snap_range[1],
    1,
)


fs070_pop2_f = filter_snapshots(
    "../particle_data/pop_2_data/fs07_refine", f7_snap_range[0], f7_snap_range[1], 1
)
fs035_pop2_f = filter_snapshots(
    "../particle_data/pop_2_data/fs035_ms10", f3_snap_range[0], f3_snap_range[1], 1
)
fs035_pop2_f, f3_matched_with_f7_nums, f7_nums = find_matching_time(
    sequence=fs070_pop2_f, look_up_sequence=fs035_pop2_f, orig_seq_out_num=True
)

# apply uniqueness cuts.
unique_idxs = np.unique(fs035_pop2_f, return_index=True)[1]
fs070_pop2_f = np.array(fs070_pop2_f)[unique_idxs].tolist()
fs035_pop2_f = np.array(fs035_pop2_f)[unique_idxs].tolist()
f3_nums = f3_matched_with_f7_nums[unique_idxs]
f7_nums = f7_nums[unique_idxs]

fs070_halo_f = get_snapshots(
    snapshot_file_list=filter_snapshots(
        r"../halo_data/fs07_refine/fof_best", f7_snap_range[0], f7_snap_range[1], 1
    ),
    get_list=f7_nums,
)
fs035_halo_f = get_snapshots(
    snapshot_file_list=filter_snapshots(
        r"../halo_data/fs035_ms10/fof_best", f3_snap_range[0], f3_snap_range[1], 1
    ),
    get_list=f3_nums,
)

f7_snap_f = get_snapshots(fs070_snap_dir, get_list=f7_nums)
f3_snap_f = get_snapshots(fs035_snap_dir, get_list=f3_nums)

f7_series = np.loadtxt("../sci_plots/fof_time_series/fs07_refine_fof_best_113_1318.txt")
f7_series = f7_series[f7_series[:, 3] > 30]
f3_series = np.loadtxt("../sci_plots/fof_time_series/fs035_ms10_fof_best_154_1469.txt")
f3_series = f3_series[f3_series[:, 3] > 30]

cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))
plt_wdth = 400

star_bins = 2000
pxl_size = (plt_wdth / star_bins) ** 2  # pc
lum_range = (3e33, 3e36)  # (2e32, 5e35)
gas_range = (0.008, 0.32)
gas_alpha = 0.5
lum_alpha = 1
cell_fields, epf = ram_fields()


sequence_dir = "../rendering/gas_lum/two_panel"
if not os.path.exists(sequence_dir):
    print("# Creating new sequence directory", sequence_dir)
    os.makedirs(sequence_dir)
#%%

for m_i, (f7_gas, f3_gas) in enumerate(zip(f7_snap_f, f3_snap_f)):
    print("Reading", f7_gas)
    print("Reading", f3_gas)
    outnum_f7 = f7_gas.split("/")[-1].split("_")[-1]
    outnum_f3 = f3_gas.split("/")[-1].split("_")[-1]
    # read ramses data
    f7_info_file = os.path.join(f7_gas, "info_{}.txt".format(outnum_f7))
    f3_info_file = os.path.join(f3_gas, "info_{}.txt".format(outnum_f3))
    f7_ram_ds = yt.load(f7_info_file, fields=cell_fields, extra_particle_fields=epf)
    f3_ram_ds = yt.load(f3_info_file, fields=cell_fields, extra_particle_fields=epf)
    # post processed star data
    f7_code_ctr = np.loadtxt(fs070_pop2_f[m_i], max_rows=5)[2:5, 6]
    f7_t_myr = np.loadtxt(fs070_pop2_f[m_i], max_rows=2)[0, 6]
    f7_redshift = np.loadtxt(fs070_pop2_f[m_i], max_rows=2)[1, 6]
    f7_stars = np.vstack(
        (
            np.loadtxt(os.path.join(fs070_halo_f[m_i], "field_stars.txt")),
            np.loadtxt(os.path.join(fs070_halo_f[m_i], "bound_stars.txt")),
        )
    )
    f3_code_ctr = np.loadtxt(fs035_pop2_f[m_i], max_rows=5)[2:5, 6]
    f3_t_myr = np.loadtxt(fs035_pop2_f[m_i], max_rows=2)[0, 6]
    f3_redshift = np.loadtxt(fs035_pop2_f[m_i], max_rows=2)[1, 6]
    f3_stars = np.vstack(
        (
            np.loadtxt(os.path.join(fs035_halo_f[m_i], "field_stars.txt")),
            np.loadtxt(os.path.join(fs035_halo_f[m_i], "bound_stars.txt")),
        )
    )
    f7_star_ids = f7_stars[:, 0]
    f7_star_lums = f7_stars[:, 2]
    f7_x = f7_stars[:, 3]
    f7_y = f7_stars[:, 4]
    f7_z = f7_stars[:, 5]
    f3_star_lums = f3_stars[:, 2]
    f3_x = f3_stars[:, 3]
    f3_y = f3_stars[:, 4]
    f3_z = f3_stars[:, 5]
    # get the projected densities
    print("Integrating Gas")
    f7_gas = yt.ProjectionPlot(
        f7_ram_ds, "z", ("gas", "density"), width=(plt_wdth, "pc"), center=f7_code_ctr
    )
    f7_gas_frb = f7_gas.data_source.to_frb((plt_wdth, "pc"), star_bins)
    f7_gas_array = np.array(f7_gas_frb["gas", "density"])
    f3_gas = yt.ProjectionPlot(
        f3_ram_ds, "z", ("gas", "density"), width=(plt_wdth, "pc"), center=f3_code_ctr
    )
    f3_gas_frb = f3_gas.data_source.to_frb((plt_wdth, "pc"), star_bins)
    f3_gas_array = np.array(f3_gas_frb["gas", "density"])
    print("Integrating Luminosity")
    # get the projected luminosity
    f7_lums, _, _ = np.histogram2d(
        f7_x,
        f7_y,
        bins=star_bins,
        weights=f7_star_lums,
        normed=False,
        range=[
            [-plt_wdth / 2, plt_wdth / 2],
            [-plt_wdth / 2, plt_wdth / 2],
        ],
    )
    f7_lums = f7_lums.T
    f3_lums, _, _ = np.histogram2d(
        f3_x,
        f3_y,
        bins=star_bins,
        weights=f3_star_lums,
        normed=False,
        range=[
            [-plt_wdth / 2, plt_wdth / 2],
            [-plt_wdth / 2, plt_wdth / 2],
        ],
    )
    f3_lums = f3_lums.T
    #%%
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
                ncols=2,
                sharex=True,
                sharey=True,
                figsize=(12, 6),
                dpi=400,
                facecolor=cm.Greys_r(0),
            )
            fig.subplots_adjust(wspace=-0.047, hspace=0)

            # lum rendering
            f7_lum_image = ax[0].imshow(
                f7_lums / pxl_size,
                cmap="inferno",
                # interpolation="gaussian",
                origin="lower",
                extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
                norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
                alpha=lum_alpha,
            )
            ax[1].imshow(
                f3_lums / pxl_size,
                cmap="inferno",
                # interpolation="gaussian",
                origin="lower",
                extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
                norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
                alpha=lum_alpha,
            )

            # gas rendering
            f7_gas_image = ax[0].imshow(
                f7_gas_array,
                cmap="cubehelix",
                interpolation="gaussian",
                origin="lower",
                extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
                norm=LogNorm(gas_range[0], gas_range[1]),
                alpha=gas_alpha,
            )
            ax[1].imshow(
                f3_gas_array,
                cmap="cubehelix",
                interpolation="gaussian",
                origin="lower",
                extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
                norm=LogNorm(gas_range[0], gas_range[1]),
                alpha=gas_alpha,
            )

            ax[0].set(
                xlim=(-plt_wdth / 2, plt_wdth / 2),
                ylim=(-plt_wdth / 2, plt_wdth / 2),
                xticklabels=[],
                yticklabels=[],
            )

            ax[1].xaxis.set_ticks_position("none")
            ax[1].yaxis.set_ticks_position("none")
            ax[0].xaxis.set_ticks_position("none")
            ax[0].yaxis.set_ticks_position("none")

            # add some scale bar
            scale = patches.Rectangle(
                xy=(plt_wdth / 2 * 0.38, -plt_wdth / 2 * 0.78),
                width=plt_wdth / 2 * 0.5,
                height=0.025 * plt_wdth / 2,
                linewidth=0,
                edgecolor="white",
                facecolor="white",
            )
            ax[1].text(
                plt_wdth / 2 * 0.63,
                -plt_wdth / 2 * 0.87,
                r"$\mathrm{{{:.0f} \: pc}}$".format(plt_wdth / 2 * 0.5),
                ha="center",
                va="center",
                color="white",
                # fontproperties=leg_font,
                fontsize=14,
            )
            ax[1].add_patch(scale)

            # time  and redshift
            box_style = {
                "boxstyle": "Square",
                "facecolor": colors.to_rgba("black")[:-1] + (0.5,),
                "linewidth": 1,
                "edgecolor": "grey",
                "pad": 0.42,
            }
            ax[0].text(
                0.05,
                0.95,
                (
                    r"$\mathrm{{t = {:.2f} \: Myr}}$" "\n" r"$\mathrm{{z = {:.2f} }}$"
                ).format(f7_redshift, f7_t_myr),
                ha="left",
                va="top",
                color="white",
                fontsize=14,
                transform=ax[0].transAxes,
                bbox=box_style,
            )
            # efficiency labels
            ax[0].text(
                0.78,
                0.95,
                r"$f_* = 0.70$",
                ha="left",
                va="top",
                color="white",
                fontsize=14,
                transform=ax[0].transAxes,
                bbox=box_style,
            )
            ax[1].text(
                0.78,
                0.95,
                r"$f_* = 0.35$",
                ha="left",
                va="top",
                color="white",
                fontsize=14,
                transform=ax[1].transAxes,
                bbox=box_style,
            )
            # zoom in axes and sfr
            with plt.style.context("dark_background"):
                with plt.rc_context(
                    {
                        "font.family": "serif",
                        "mathtext.fontset": "cm",
                        "xtick.labelsize": 8,
                        "ytick.labelsize": 8,
                        "font.size": 10,
                    }
                ):
                    f7_inset = ax[0].inset_axes([0.05, 0.05, 0.30, 0.30])
                    f3_inset = ax[1].inset_axes([0.05, 0.65, 0.30, 0.30])
                    inset_width = 40
                    f7_inset.set(
                        xlim=(-inset_width / 2, inset_width / 2),
                        ylim=(-inset_width / 2, inset_width / 2),
                        xticklabels=[],
                        yticklabels=[],
                    )
                    f3_inset.set(
                        xlim=(-inset_width / 2, inset_width / 2),
                        ylim=(-inset_width / 2, inset_width / 2),
                        xticklabels=[],
                        yticklabels=[],
                    )
                    f7_inset.xaxis.set_ticks_position("none")
                    f7_inset.yaxis.set_ticks_position("none")
                    f3_inset.xaxis.set_ticks_position("none")
                    f3_inset.yaxis.set_ticks_position("none")

                    # inset luminosity
                    f7_inset.imshow(
                        f7_lums / pxl_size,
                        cmap="inferno",
                        # interpolation="gaussian",
                        origin="lower",
                        extent=[
                            -plt_wdth / 2,
                            plt_wdth / 2,
                            -plt_wdth / 2,
                            plt_wdth / 2,
                        ],
                        norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
                        alpha=lum_alpha,
                    )
                    f3_inset.imshow(
                        f3_lums / pxl_size,
                        cmap="inferno",
                        # interpolation="gaussian",
                        origin="lower",
                        extent=[
                            -plt_wdth / 2,
                            plt_wdth / 2,
                            -plt_wdth / 2,
                            plt_wdth / 2,
                        ],
                        norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
                        alpha=lum_alpha,
                    )
                    # inset gas
                    f7_inset.imshow(
                        f7_gas_array,
                        cmap="cubehelix",
                        interpolation="gaussian",
                        origin="lower",
                        extent=[
                            -plt_wdth / 2,
                            plt_wdth / 2,
                            -plt_wdth / 2,
                            plt_wdth / 2,
                        ],
                        norm=LogNorm(gas_range[0], gas_range[1]),
                        alpha=gas_alpha,
                    )
                    f3_inset.imshow(
                        f3_gas_array,
                        cmap="cubehelix",
                        interpolation="gaussian",
                        origin="lower",
                        extent=[
                            -plt_wdth / 2,
                            plt_wdth / 2,
                            -plt_wdth / 2,
                            plt_wdth / 2,
                        ],
                        norm=LogNorm(gas_range[0], gas_range[1]),
                        alpha=gas_alpha,
                    )

                    scale_ins = patches.Rectangle(
                        xy=(0, 1.1 * (inset_width / 2)),
                        width=inset_width / 2,
                        height=1,
                        linewidth=0,
                        edgecolor="white",
                        facecolor="white",
                        clip_on=False,
                    )
                    f7_inset.text(
                        inset_width / 4,
                        1.3 * (inset_width / 2),
                        r"$\mathrm{{{:.0f} \: pc}}$".format(inset_width / 2),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=12,
                    )
                    f7_inset.add_patch(scale_ins)

                    mark_inset(
                        ax[0], f7_inset, loc1=2, loc2=4, edgecolor="white", alpha=0.4
                    )
                    mark_inset(
                        ax[1], f3_inset, loc1=1, loc2=3, edgecolor="white", alpha=0.4
                    )

                    # time series line plots
                    time_series_ax_f7 = ax[0].inset_axes([0.56, 0.10, 0.40, 0.2])
                    time_series_ax_f3 = ax[1].inset_axes([0.10, 0.10, 0.40, 0.2])
                    time_series_ax_f7.spines["right"].set_visible(False)
                    time_series_ax_f7.spines["top"].set_visible(False)
                    time_series_ax_f3.spines["right"].set_visible(False)
                    time_series_ax_f3.spines["top"].set_visible(False)
                    time_series_ax_f7.patch.set_alpha(0.5)
                    time_series_ax_f3.patch.set_alpha(0.5)

                    f7_lum_field = f7_series[:, 5]
                    f7_lum_bound = f7_series[:, 4]
                    f3_lum_field = f3_series[:, 5]
                    f3_lum_bound = f3_series[:, 4]

                    f7_time_mask = f7_series[:, 0] <= f7_t_myr
                    f3_time_mask = f3_series[:, 0] <= f3_t_myr

                    time_series_ax_f7.plot(
                        f7_series[:, 0][f7_time_mask],
                        (f7_lum_bound + f7_lum_field)[f7_time_mask],
                        color=cmap[0],
                        label="Total",
                    )
                    time_series_ax_f7.plot(
                        f7_series[:, 0][f7_time_mask],
                        f7_lum_bound[f7_time_mask],
                        color=cmap[1],
                        label="Bound",
                        alpha=0.8,
                    )
                    time_series_ax_f3.plot(
                        f3_series[:, 0][f3_time_mask],
                        (f3_lum_bound + f3_lum_field)[f3_time_mask],
                        color=cmap[2],
                        label="Total",
                    )
                    time_series_ax_f3.plot(
                        f3_series[:, 0][f3_time_mask],
                        f3_lum_bound[f3_time_mask],
                        color=cmap[3],
                        label="Bound",
                        alpha=0.8,
                    )
                    # after current times
                    time_series_ax_f7.plot(
                        f7_series[:, 0][~f7_time_mask],
                        (f7_lum_bound + f7_lum_field)[~f7_time_mask],
                        alpha=0.5,
                        color="grey",
                    )
                    time_series_ax_f7.plot(
                        f7_series[:, 0][~f7_time_mask],
                        f7_lum_bound[~f7_time_mask],
                        alpha=0.5,
                        color="grey",
                    )
                    time_series_ax_f3.plot(
                        f3_series[:, 0][~f3_time_mask],
                        (f3_lum_bound + f3_lum_field)[~f3_time_mask],
                        alpha=0.5,
                        color="grey",
                    )
                    time_series_ax_f3.plot(
                        f3_series[:, 0][~f3_time_mask],
                        f3_lum_bound[~f3_time_mask],
                        alpha=0.5,
                        color="grey",
                    )

                    time_series_ax_f7.grid(
                        visible=True,
                        which="major",
                        axis="y",
                        ls=":",
                        color="white",
                        zorder=0.5,
                        alpha=0.8,
                    )

                    time_series_ax_f7.set(
                        yscale="log",
                        xlabel="$\mathrm{time\:(Myr)}$",
                        ylim=(2e35, 1e39),
                        xlim=(f7_series[:, 0].min(), f7_series[:, 0].max()),
                    )
                    time_series_ax_f7.set_title(
                        r"$\mathrm{Surface\: Brightness}\:$"
                        r"$\left(\mathrm{erg} \:\mathrm{s}^{-1} \:\mathrm{\AA}^{-1}\right)$",
                        fontsize=10,
                    )
                    time_series_ax_f3.grid(
                        visible=True,
                        which="major",
                        axis="y",
                        ls=":",
                        color="white",
                        zorder=0.5,
                        alpha=0.8,
                    )
                    time_series_ax_f3.set(
                        yscale="log",
                        xlabel="$\mathrm{time\:(Myr)}$",
                        ylim=(2e35, 1e39),
                        xlim=(f3_series[:, 0].min(), f3_series[:, 0].max()),
                    )
                    time_series_ax_f3.set_title(
                        r"$\mathrm{Surface\: Brightness}\:$"
                        r"$\left(\mathrm{erg} \:\mathrm{s}^{-1} \:\mathrm{\AA}^{-1}\right)$",
                        fontsize=10,
                    )
                    f7_leg = time_series_ax_f7.legend(
                        loc="lower right", ncol=2, fontsize=8
                    )
                    f3_leg = time_series_ax_f3.legend(
                        loc="lower right", ncol=2, fontsize=8
                    )

                    f7_leg.get_frame().set_alpha(0)
                    f3_leg.get_frame().set_alpha(0)

            # declar the color bar axes
            dens_cbar_ax = fig.add_axes([0.5130, 0.095, 0.3777, 0.030])
            lums_cbar_ax = fig.add_axes([0.1345, 0.095, 0.3777, 0.030])

            dens_cbar = fig.colorbar(
                f7_gas_image, cax=dens_cbar_ax, pad=-1, orientation="horizontal"
            )
            dens_cbar.set_label(
                label=r"$\mathrm{Surface \:Density\:(g \: cm^{-2})}$",
                fontsize=15,
                labelpad=5,
            )
            dens_cbar.ax.xaxis.set_tick_params(pad=2, labelsize=12)

            lums_cbar = fig.colorbar(
                f7_lum_image, cax=lums_cbar_ax, pad=-1, orientation="horizontal"
            )
            lums_cbar.set_label(
                label=r"$\mathrm{Surface\:Brightness}$"
                r"$\: \mathrm{\lambda = 1500 \: \AA \:}$"
                r"$\mathrm{\left(erg\:\:s^{-1}\:\AA^{-1}\:pc^{-2}\right)}$",
                fontsize=15,
                labelpad=5,
            )
            lums_cbar.ax.xaxis.set_tick_params(pad=2, labelsize=12)

    output_path = os.path.join(
        sequence_dir,
        "render_{}_{}_{}.png".format(
            outnum_f7, outnum_f3, str(np.round(f3_redshift, 3)).replace(".", "_")
        ),
    )
    plt.savefig(
        os.path.expanduser(output_path),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close("all")
    print(">Saved:", output_path)
