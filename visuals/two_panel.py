import sys

sys.path.append("../")
# sys.path.insert(
#     1,
#     "/scratch/zt1/project/ricotti-prj/user/fgarcia4/master/lib/python3.7/site-packages",
# )
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
from modules.macros import filter_snapshots, ram_fields, t_myr_from_z
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy import interpolate
from scipy.ndimage import gaussian_filter

yt.enable_parallelism()

# f7_snap_range = (599, 1296)
# f3_snap_range = (179, 1469)
# f7_snap_range = (924, 1318)
# f3_snap_range = (1103, 1502)
# f7_snap_range = (500, 500)
# f3_snap_range = (500, 500)
f7_snap_range = (1318, 1318)
f3_snap_range = (1502, 1502)

# fs070_dir = "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine"
# fs035_dir = "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs035_ms10"

master_data_dir = "/scratch/dt2/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/"

fs070_dir = os.path.join(master_data_dir, "fs07_refine")
fs035_dir = os.path.join(master_data_dir, "fs035_ms10")

# fs070_dir = os.path.relpath("../../cosm_test_data/refine")
# fs035_dir = os.path.relpath("../../cosm_test_data/fs035_ms10/")

fs070_snap_dir = filter_snapshots(
    fs070_dir,
    f7_snap_range[0],
    f7_snap_range[1],
    1,
)
fs035_snap_dir = filter_snapshots(
    fs035_dir,
    f3_snap_range[0],
    f3_snap_range[1],
    1,
)

print(fs035_snap_dir)
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

# get SFR data
bin_width_myr = 1  # interpolation equal width
f7_series = np.loadtxt("../sim_log_files/fs07_refine/logSFC")
f3_series = np.loadtxt("../sim_log_files/fs035_ms10/logSFC")
redshift_hi = f7_series[:, 2]
redshift_lo = f3_series[:, 2]
tmyr_hi = t_myr_from_z(redshift_hi)
tmyr_lo = t_myr_from_z(redshift_lo)

hi_tot_m = np.cumsum(f7_series[:, 7])
lo_tot_m = np.cumsum(f3_series[:, 7])

lo_interp_points = np.arange(tmyr_lo.min(), tmyr_lo.max(), bin_width_myr)
lo_interp = interpolate.interp1d(x=tmyr_lo, y=lo_tot_m, kind="previous")

hi_interp_points = np.arange(tmyr_hi.min(), tmyr_hi.max(), bin_width_myr)
hi_interp = interpolate.interp1d(x=tmyr_hi, y=hi_tot_m, kind="previous")

sfr_fs035 = np.gradient(lo_interp(lo_interp_points)) / (bin_width_myr * 1e6)
sfr_fs070 = np.gradient(hi_interp(hi_interp_points)) / (bin_width_myr * 1e6)

lo_tot_m = lo_interp(lo_interp_points)
hi_tot_m = hi_interp(hi_interp_points)

cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))
plt_wdth = 400

star_bins = 2000
pxl_size = (plt_wdth / star_bins) ** 2  # pc
pc2_to_cm2 = 9.52140614e36
lum_range = (3e33 / pc2_to_cm2, 3e36 / pc2_to_cm2)  # (2e32, 5e35)
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
    f7_ram_ds = yt.load(
        f7_info_file,
        fields=cell_fields,
        # extra_particle_fields=epf,
    )
    ad70 = f7_ram_ds.all_data()
    print("70 percent: gas density, surface density ")
    print(
        "mean",
        np.mean(ad70["gas", "density"].to("Msun/pc**3")),
        np.mean(
            ad70["gas", "density"].to("Msun/pc**3") ** (2 / 3)
            * ad70["gas", "mass"].to("Msun") ** (1 / 3)
        ),
    )
    print(
        "median",
        np.median(ad70["gas", "density"].to("Msun/pc**3")),
        np.median(
            ad70["gas", "density"].to("Msun/pc**3") ** (2 / 3)
            * ad70["gas", "mass"].to("Msun") ** (1 / 3)
        ),
    )
    print(
        "max",
        np.max(ad70["gas", "density"].to("Msun/pc**3")),
        np.max(
            ad70["gas", "density"].to("Msun/pc**3") ** (2 / 3)
            * ad70["gas", "mass"].to("Msun") ** (1 / 3)
        ),
    )
    print(
        "min",
        np.min(ad70["gas", "density"].to("Msun/pc**3")),
        np.min(
            ad70["gas", "density"].to("Msun/pc**3") ** (2 / 3)
            * ad70["gas", "mass"].to("Msun") ** (1 / 3)
        ),
    )
    print(
        "std",
        np.std(ad70["gas", "density"].to("Msun/pc**3")),
        np.std(
            ad70["gas", "density"].to("Msun/pc**3") ** (2 / 3)
            * ad70["gas", "mass"].to("Msun") ** (1 / 3)
        ),
    )
    print(
        "75 percentile",
        np.percentile(ad70["gas", "density"].to("Msun/pc**3"), 75),
        np.percentile(
            ad70["gas", "density"].to("Msun/pc**3") ** (2 / 3)
            * ad70["gas", "mass"].to("Msun") ** (1 / 3),
            75,
        ),
    )

    f3_ram_ds = yt.load(
        f3_info_file,
        fields=cell_fields,
        # extra_particle_fields=epf,
    )
    ad35 = f3_ram_ds.all_data()
    print("35 percent: gas density, surface density ")
    print(
        "mean",
        np.mean(ad35["gas", "density"].to("Msun/pc**3")),
        np.mean(
            ad35["gas", "density"].to("Msun/pc**3") ** (2 / 3)
            * ad35["gas", "mass"].to("Msun") ** (1 / 3)
        ),
    )
    print(
        "median",
        np.median(ad35["gas", "density"].to("Msun/pc**3")),
        np.median(
            ad35["gas", "density"].to("Msun/pc**3") ** (2 / 3)
            * ad35["gas", "mass"].to("Msun") ** (1 / 3)
        ),
    )
    print(
        "max",
        np.max(ad35["gas", "density"].to("Msun/pc**3")),
        np.max(
            ad35["gas", "density"].to("Msun/pc**3") ** (2 / 3)
            * ad35["gas", "mass"].to("Msun") ** (1 / 3)
        ),
    )
    print(
        "min",
        np.min(ad35["gas", "density"].to("Msun/pc**3")),
        np.min(
            ad35["gas", "density"].to("Msun/pc**3") ** (2 / 3)
            * ad35["gas", "mass"].to("Msun") ** (1 / 3)
        ),
    )
    print(
        "std",
        np.std(ad35["gas", "density"].to("Msun/pc**3")),
        np.std(
            ad35["gas", "density"].to("Msun/pc**3") ** (2 / 3)
            * ad35["gas", "mass"].to("Msun") ** (1 / 3)
        ),
    )
    print(
        "75 percentile",
        np.percentile(ad35["gas", "density"].to("Msun/pc**3"), 75),
        np.percentile(
            ad35["gas", "density"].to("Msun/pc**3") ** (2 / 3)
            * ad35["gas", "mass"].to("Msun") ** (1 / 3),
            75,
        ),
    )
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
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "font.size": 10,
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
            fig.subplots_adjust(wspace=-0.0475, hspace=0)

            # lum rendering
            f7_lum_image = ax[0].imshow(
                f7_lums / (pxl_size * pc2_to_cm2),
                cmap="inferno",
                interpolation="gaussian",
                origin="lower",
                extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
                norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
                alpha=lum_alpha,
            )
            f3_lum_image = ax[1].imshow(
                f3_lums / (pxl_size * pc2_to_cm2),
                cmap="inferno",
                interpolation="gaussian",
                origin="lower",
                extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
                norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
                alpha=lum_alpha,
            )
            gas_sigma = 10
            # gas rendering
            f7_gas_image = ax[0].imshow(
                gaussian_filter(f7_gas_array, sigma=gas_sigma),
                cmap="cubehelix",
                # interpolation="gaussian",
                origin="lower",
                extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
                norm=LogNorm(gas_range[0], gas_range[1]),
                alpha=gas_alpha,
            )
            f3_gas_image = ax[1].imshow(
                gaussian_filter(f3_gas_array, sigma=gas_sigma),
                cmap="cubehelix",
                # interpolation="gaussian",
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
                xy=(plt_wdth / 2 * 0.38, -plt_wdth / 2 * 0.80),
                width=plt_wdth / 2 * 0.5,
                height=0.020 * plt_wdth / 2,
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
                ).format(f7_t_myr, f7_redshift),
                ha="left",
                va="top",
                color="white",
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
                transform=ax[1].transAxes,
                bbox=box_style,
            )
            # zoom in axes and sfr
            # with plt.style.context("dark_background"):
            #     with plt.rc_context(
            #         {
            #             "font.family": "serif",
            #             "mathtext.fontset": "cm",
            #             "xtick.labelsize": 8,
            #             "ytick.labelsize": 8,
            #             "font.size": 10,
            #         }
            #     ):
            #         f7_inset = ax[0].inset_axes([0.05, 0.05, 0.30, 0.30])
            #         f3_inset = ax[1].inset_axes([0.05, 0.65, 0.30, 0.30])
            #         inset_width = 40
            #         f7_inset.set(
            #             xlim=(-inset_width / 2, inset_width / 2),
            #             ylim=(-inset_width / 2, inset_width / 2),
            #             xticklabels=[],
            #             yticklabels=[],
            #         )
            #         f3_inset.set(
            #             xlim=(-inset_width / 2, inset_width / 2),
            #             ylim=(-inset_width / 2, inset_width / 2),
            #             xticklabels=[],
            #             yticklabels=[],
            #         )
            #         f7_inset.xaxis.set_ticks_position("none")
            #         f7_inset.yaxis.set_ticks_position("none")
            #         f3_inset.xaxis.set_ticks_position("none")
            #         f3_inset.yaxis.set_ticks_position("none")

            #         # inset luminosity
            #         f7_inset.imshow(
            #             f7_lums / pxl_size,
            #             cmap="inferno",
            #             interpolation="gaussian",
            #             origin="lower",
            #             extent=[
            #                 -plt_wdth / 2,
            #                 plt_wdth / 2,
            #                 -plt_wdth / 2,
            #                 plt_wdth / 2,
            #             ],
            #             norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
            #             alpha=lum_alpha,
            #         )
            #         f3_inset.imshow(
            #             f3_lums / pxl_size,
            #             cmap="inferno",
            #             interpolation="gaussian",
            #             origin="lower",
            #             extent=[
            #                 -plt_wdth / 2,
            #                 plt_wdth / 2,
            #                 -plt_wdth / 2,
            #                 plt_wdth / 2,
            #             ],
            #             norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
            #             alpha=lum_alpha,
            #         )
            #         # inset gas
            #         f7_inset.imshow(
            #             f7_gas_array,
            #             cmap="cubehelix",
            #             interpolation="gaussian",
            #             origin="lower",
            #             extent=[
            #                 -plt_wdth / 2,
            #                 plt_wdth / 2,
            #                 -plt_wdth / 2,
            #                 plt_wdth / 2,
            #             ],
            #             norm=LogNorm(gas_range[0], gas_range[1]),
            #             alpha=gas_alpha,
            #         )
            #         f3_inset.imshow(
            #             f3_gas_array,
            #             cmap="cubehelix",
            #             interpolation="gaussian",
            #             origin="lower",
            #             extent=[
            #                 -plt_wdth / 2,
            #                 plt_wdth / 2,
            #                 -plt_wdth / 2,
            #                 plt_wdth / 2,
            #             ],
            #             norm=LogNorm(gas_range[0], gas_range[1]),
            #             alpha=gas_alpha,
            #         )

            #         scale_ins = patches.Rectangle(
            #             xy=(0, 1.1 * (inset_width / 2)),
            #             width=inset_width / 2,
            #             height=1,
            #             linewidth=0,
            #             edgecolor="white",
            #             facecolor="white",
            #             clip_on=False,
            #         )
            #         f7_inset.text(
            #             inset_width / 4,
            #             1.3 * (inset_width / 2),
            #             r"$\mathrm{{{:.0f} \: pc}}$".format(inset_width / 2),
            #             ha="center",
            #             va="center",
            #             color="white",
            #         )

            #         f7_inset.add_patch(scale_ins)

            #         # mark_inset(
            #         #     ax[0], f7_inset, loc1=2, loc2=4, edgecolor="white", alpha=0.4
            #         # )
            #         # mark_inset(
            #         #     ax[1], f3_inset, loc1=2, loc2=4, edgecolor="white", alpha=0.4
            #         # )

            #         # time series line plots
            #         totm_series = ax[0].inset_axes([0.56, 0.10, 0.40, 0.2])
            #         sfr_series = ax[1].inset_axes([0.10, 0.10, 0.40, 0.2])
            #         totm_series.spines["right"].set_visible(False)
            #         totm_series.spines["top"].set_visible(False)
            #         sfr_series.spines["right"].set_visible(False)
            #         sfr_series.spines["top"].set_visible(False)
            #         totm_series.patch.set_alpha(0.5)
            #         sfr_series.patch.set_alpha(0.5)

            #         interp_mask_hi = hi_interp_points <= f7_t_myr
            #         interp_mask_lo = lo_interp_points <= f3_t_myr

            #         # plot total masses
            #         totm_series.plot(
            #             hi_interp_points[interp_mask_hi],
            #             np.log10(hi_tot_m[interp_mask_hi]),
            #             color=cmap[2],
            #             alpha=0.8,
            #             label=r"$70 \%$",
            #         )
            #         # after current time
            #         totm_series.plot(
            #             hi_interp_points[~interp_mask_hi],
            #             np.log10(hi_tot_m[~interp_mask_hi]),
            #             alpha=0.3,
            #             color="grey",
            #         )

            #         totm_series.plot(
            #             lo_interp_points[interp_mask_lo],
            #             np.log10(lo_tot_m[interp_mask_lo]),
            #             color=cmap[1],
            #             alpha=0.8,
            #             label=r"$35 \%$",
            #         )
            #         # after current time
            #         totm_series.plot(
            #             lo_interp_points[~interp_mask_lo],
            #             np.log10(lo_tot_m[~interp_mask_lo]),
            #             color="grey",
            #             alpha=0.3,
            #         )

            #         # plot star formation rates
            #         sfr_series.plot(
            #             hi_interp_points[interp_mask_hi],
            #             sfr_fs070[interp_mask_hi],
            #             color=cmap[2],
            #             alpha=0.8,
            #             lw=1,
            #         )
            #         sfr_series.plot(
            #             lo_interp_points[interp_mask_lo],
            #             sfr_fs035[interp_mask_lo],
            #             color=cmap[1],
            #             alpha=0.8,
            #             lw=1,
            #         )

            #         sfr_series.plot(
            #             hi_interp_points[~interp_mask_hi],
            #             sfr_fs070[~interp_mask_hi],
            #             color="grey",
            #             alpha=0.3,
            #             lw=1,
            #         )
            #         sfr_series.plot(
            #             lo_interp_points[~interp_mask_lo],
            #             sfr_fs035[~interp_mask_lo],
            #             color="grey",
            #             alpha=0.3,
            #             lw=1,
            #         )

            #         totm_series.grid(
            #             visible=True,
            #             which="major",
            #             axis="y",
            #             ls=":",
            #             color="white",
            #             zorder=0.5,
            #             alpha=0.8,
            #         )

            #         totm_series.set(
            #             xlabel="$\mathrm{time\:(Myr)}$",
            #             ylim=(3, 6.5),
            #             xlim=(hi_interp_points.min(), hi_interp_points.max()),
            #         )
            #         totm_series.set_title(
            #             r"$\mathrm{\log M_{\star, total} \: (M_{\odot})}\: $",
            #             fontsize=10,
            #         )
            #         sfr_series.grid(
            #             visible=True,
            #             which="major",
            #             axis="y",
            #             ls=":",
            #             color="white",
            #             zorder=0.5,
            #             alpha=0.8,
            #         )
            #         sfr_series.set(
            #             xlabel="$\mathrm{time\:(Myr)}$",
            #             ylim=(0, 0.05),
            #             xlim=(hi_interp_points.min(), hi_interp_points.max()),
            #         )
            #         sfr_series.set_title(
            #             r"$\mathrm{SFR\:\left(M_{\odot}\:yr^{-1}\right)}$",
            #             fontsize=10,
            #         )
            #         f7_leg = totm_series.legend(loc="lower right", fontsize=8)
            #         # f3_leg = sfr_series.legend(loc="lower right",  fontsize=8)

            #         f7_leg.get_frame().set_alpha(0)
            # f3_leg.get_frame().set_alpha(0)

            # declar the color bar axes
            dens_cbar_ax = ax[1].inset_axes([0.05, 0.05, 0.45, 0.040])
            lums_cbar_ax = ax[0].inset_axes([0.05, 0.05, 0.45, 0.040])
            dens_cbar_ax.tick_params(
                axis="both", labeltop="on", direction="in", which="both"
            )
            lums_cbar_ax.tick_params(axis="both", direction="in", which="both")

            dens_cbar = fig.colorbar(
                f7_gas_image, cax=dens_cbar_ax, pad=-1, orientation="horizontal"
            )
            dens_cbar.set_label(
                label=r"$\log \: \mathrm{Surface \:Density\:(g \: cm^{-2})}$" "\n",
                fontsize=10,
                labelpad=-3,
            )
            # dens_cbar.ax.xaxis.set_tick_params(pad=2)

            lums_cbar = fig.colorbar(
                f7_lum_image, cax=lums_cbar_ax, pad=-1, orientation="horizontal"
            )
            lums_cbar.set_label(
                label=r"$\log \: \mathrm{\lambda = 1500 \: \AA \:}$"
                # r"$\mathrm{Surface\:Brightness}$"
                # r"$\mathrm{Integrated\:Brightness}$"
                r"$\mathrm{\left(erg\:\:s^{-1}\:\AA^{-1}\:pc^{-2}\right)}$" "\n",
                fontsize=10,
                labelpad=-3,
            )
            # lums_cbar.ax.xaxis.set_tick_params(pad=2)
            fig.canvas.draw()

            x_labels = [
                i.get_text().replace("10^", "") for i in dens_cbar_ax.get_xticklabels()
            ]
            dens_cbar_ax.set_xticklabels(x_labels)
            x_labels = [
                i.get_text().replace("10^", "") for i in lums_cbar_ax.get_xticklabels()
            ]
            lums_cbar_ax.set_xticklabels(x_labels)

            dens_cbar_ax.xaxis.set_ticks_position("top")
            dens_cbar_ax.xaxis.set_label_position("top")
            lums_cbar_ax.xaxis.set_ticks_position("top")
            lums_cbar_ax.xaxis.set_label_position("top")

    output_path = os.path.join(
        sequence_dir,
        "render_{}_{}_{}.png".format(
            outnum_f7, outnum_f3, str(np.round(f3_redshift, 3)).replace(".", "_")
        ),
    )
    plt.savefig(
        os.path.expanduser(output_path),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    # plt.close("all")
    print(">Saved:", output_path)
