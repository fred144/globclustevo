import sys


sys.path.append("../")
import numpy as np
import os
import glob
from modules.luminosity.lum_functions import lum_look_up_table
from modules.match_t_sims import find_matching_time, get_snapshots
from modules.macros import filter_snapshots
from scipy.ndimage import gaussian_filter

sys.path.insert(1, "/home/fgarcia4/.local/lib/python3.8/site-packages")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib import colors
import misc_visuals
import yt
from modules.macros import filter_snapshots, ram_fields, t_myr_from_z, code_age_to_myr
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib as mpl


yt.enable_parallelism()
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "font.size": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
    }
)
plt.style.use("dark_background")


f7_snap_range = (118, 1318)
f3_snap_range = (154, 1502)

# f7_snap_range = (500, 500)
# f3_snap_range = (500, 500)

# fs070_dir = os.path.relpath("../../cosm_test_data/refine")
# fs035_dir = os.path.relpath("../../cosm_test_data/fs035_ms10/")

master_data_dir = (
    "/afs/shell.umd.edu/project/ricotti-prj/user/fgarcia4/dwarf/data/cluster_evolution/"
)

fs070_dir = os.path.join(master_data_dir, "fs07_refine")
fs035_dir = os.path.join(master_data_dir, "fs035_ms10")


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
f3_series = np.loadtxt("../sci_plots/fof_time_series/fs035_ms10_fof_best_154_1502.txt")
f3_series = f3_series[f3_series[:, 3] > 30]

cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))
plt_wdth = 400

star_bins = 2000
pxl_size = (plt_wdth / star_bins) ** 2  # pc
lum_range = (2e33, 3e36)  # (2e32, 5e35)
gas_range = (0.008, 0.30)
gas_alpha = 0.5
lum_alpha = 1
cell_fields, epf = ram_fields()

# cvals = [0.5, 1.7, 1.9, 2.1, 2.75, 4]
# cmapcolors = ["crimson", "magenta", "hotpink", "gold", "lime", "cyan"]
# norm = plt.Normalize(min(cvals), max(cvals))
# tuples = list(zip(map(norm, cvals), cmapcolors))
# star_cmap = mpl.colors.LinearSegmentedColormap.from_list("", tuples)

star_map = cm.get_cmap("viridis")
star_t_range = (355, 675)
# .5 Myr intervals
evenly_spaced_times = np.arange(star_t_range[0], star_t_range[1], 0.25)
star_cmap = star_map(np.linspace(0, 1, evenly_spaced_times.size))
# dense_norm = (0.007, 0.35)
temp_norm = (6e2, 6e4)

sequence_dir = "../rendering/gas_lum/multi_panel"
if not os.path.exists(sequence_dir):
    print("# Creating new sequence directory", sequence_dir)
    os.makedirs(sequence_dir)
#%%

for m_i, (f7_gas, f3_gas) in enumerate(zip(f7_snap_f, f3_snap_f)):

    print("Reading", f7_gas)
    print("Reading", f3_gas)
    outnum_f7 = f7_gas.split("/")[-1].split("_")[-1]
    outnum_f3 = f3_gas.split("/")[-1].split("_")[-1]
    if int(outnum_f3) == 1102:
        continue
    # read ramses data
    f7_info_file = os.path.join(f7_gas, "info_{}.txt".format(outnum_f7))
    f3_info_file = os.path.join(f3_gas, "info_{}.txt".format(outnum_f3))
    f7_ram_ds = yt.load(f7_info_file, fields=cell_fields, extra_particle_fields=epf)
    f3_ram_ds = yt.load(f3_info_file, fields=cell_fields, extra_particle_fields=epf)
    # post processed star data
    f7_code_ctr = np.loadtxt(fs070_pop2_f[m_i], max_rows=5)[2:5, 6]
    f3_code_ctr = np.loadtxt(fs035_pop2_f[m_i], max_rows=5)[2:5, 6]
    f7_t_myr = np.loadtxt(fs070_pop2_f[m_i], max_rows=2)[0, 6]
    f3_t_myr = np.loadtxt(fs035_pop2_f[m_i], max_rows=2)[0, 6]
    f7_redshift = np.loadtxt(fs070_pop2_f[m_i], max_rows=2)[1, 6]
    try:
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

        f7_current_ages = f7_stars[:, 1]
        f3_current_ages = f3_stars[:, 1]
        f7_star_bes = f7_t_myr - f7_current_ages
        f3_star_bes = f3_t_myr - f3_current_ages
        f7_rounded_times = np.round_(f7_current_ages, 1)
        f3_rounded_times = np.round_(f3_current_ages, 1)
        f7_unique_birth_times = np.unique(f7_rounded_times)
        f3_unique_birth_times = np.unique(f3_rounded_times)
    except:
        print("does not exist, creating luminosity tables")
        f7_current_hubble = f7_ram_ds.hubble_constant
        f7_ad = f7_ram_ds.all_data()
        f7_t_myr = float(f7_ram_ds.current_time.in_units("Myr"))
        f7_redshift = float(f7_ram_ds.current_redshift)
        f3_current_hubble = f3_ram_ds.hubble_constant
        f3_ad = f3_ram_ds.all_data()
        f3_t_myr = float(f3_ram_ds.current_time.in_units("Myr"))
        f3_redshift = float(f3_ram_ds.current_redshift)

        # find CoM of the system, starting from the most dense gas coord
        f7_sphere = f7_ram_ds.sphere("max", (plt_wdth / 2, "pc"))
        f3_sphere = f3_ram_ds.sphere("max", (plt_wdth / 2, "pc"))
        # return CoM in code units
        f7_com = f7_sphere.quantities.center_of_mass(
            use_gas=True, use_particles=True, particle_type="star"
        )

        f3_com = f3_sphere.quantities.center_of_mass(
            use_gas=True, use_particles=True, particle_type="star"
        )

        # recenter the stars based on the CoM
        f7_x = np.array((f7_ad["star", "particle_position_x"] - f7_com[0]).to("pc"))
        f7_y = np.array((f7_ad["star", "particle_position_y"] - f7_com[1]).to("pc"))
        f7_z = np.array((f7_ad["star", "particle_position_z"] - f7_com[2]).to("pc"))
        f7_be_star = f7_ad["star", "particle_birth_epoch"]

        f7_unique_birth_epochs = code_age_to_myr(
            f7_ad["star", "particle_birth_epoch"], f7_current_hubble, unique_age=True
        )
        # calculate the age of the universe when the first star was born
        # using the logSFC as a reference point for redshift when the first star
        # was born. Every age is relative to this. Due to our mods of ramses.
        f7_birth_start = np.round_(
            float(f7_ram_ds.cosmology.t_from_z(f7_series[0, 2]).in_units("Myr")), 0
        )
        # all the birth epochs of the stars
        f7_converted_unfiltered = code_age_to_myr(
            f7_ad["star", "particle_birth_epoch"], f7_current_hubble, unique_age=False
        )
        f7_abs_birth_epochs = np.round(f7_converted_unfiltered + f7_birth_start, 3)  #!
        f7_current_ages = np.round(f7_t_myr, 3) - np.round(f7_abs_birth_epochs, 3)
        f7_star_lums = (
            lum_look_up_table(
                stellar_ages=f7_current_ages,
                table_link="../particle_data/luminosity_look_up_tables/l1500_inst_e.txt",
                column_idx=1,
                log=True,
            )
            * 1e-5
        )

        ###
        f3_x = (f3_ad["star", "particle_position_x"] - f3_com[0]).to("pc")
        f3_y = (f3_ad["star", "particle_position_y"] - f3_com[1]).to("pc")
        f3_z = (f3_ad["star", "particle_position_z"] - f3_com[2]).to("pc")
        f3_be_star = f3_ad["star", "particle_birth_epoch"]
        f3_unique_birth_epochs = code_age_to_myr(
            f3_ad["star", "particle_birth_epoch"], f3_current_hubble, unique_age=True
        )
        f3_birth_start = np.round_(
            float(f3_ram_ds.cosmology.t_from_z(f3_series[0, 2]).in_units("Myr")), 0
        )
        f3_converted_unfiltered = code_age_to_myr(
            f3_ad["star", "particle_birth_epoch"], f3_current_hubble, unique_age=False
        )
        f3_abs_birth_epochs = np.round(f3_converted_unfiltered + f3_birth_start, 3)  #!
        f3_current_ages = np.round(f3_t_myr, 3) - np.round(f3_abs_birth_epochs, 3)
        f3_star_lums = (
            lum_look_up_table(
                stellar_ages=f3_current_ages,
                table_link="../particle_data/luminosity_look_up_tables/l1500_inst_e.txt",
                column_idx=1,
                log=True,
            )
            * 1e-5
        )

        f7_rounded_times = np.round_(f7_current_ages, 1)
        f3_rounded_times = np.round_(f3_current_ages, 1)
        f7_unique_birth_times = np.unique(f7_rounded_times)
        f3_unique_birth_times = np.unique(f3_rounded_times)

    # get the projected densities
    #!!!
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
    #!!!
    print("Getting Temp")
    f7_t = yt.ProjectionPlot(
        f7_ram_ds,
        "z",
        ("gas", "temperature"),
        width=(plt_wdth, "pc"),
        center=f7_code_ctr,
        weight_field=("gas", "density"),
    )
    f7_temp_frb = f7_t.data_source.to_frb((plt_wdth, "pc"), star_bins)
    f7_temp_array = np.array(f7_temp_frb["gas", "temperature"])
    f3_t = yt.ProjectionPlot(
        f3_ram_ds,
        "z",
        ("gas", "temperature"),
        width=(plt_wdth, "pc"),
        center=f3_code_ctr,
        weight_field=("gas", "density"),
    )
    f3_temp_frb = f3_t.data_source.to_frb((plt_wdth, "pc"), star_bins)
    f3_temp_array = np.array(f3_temp_frb["gas", "temperature"])
    #!!!
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
    inset_width = 80
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        sharex=True,
        sharey=True,
        figsize=(6, 6),
        dpi=400,
        facecolor=cm.Greys_r(0),
    )
    fig.subplots_adjust(wspace=-0.0475, hspace=0)
    ax1 = ax.inset_axes([1.8, 0, 1, 1])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    # zoom in axes and sfr
    f7_temp = ax.inset_axes([1.0, 0.0, 0.40, 0.40])
    f3_temp = ax1.inset_axes([-0.4, 0.0, 0.40, 0.40])
    f7_inset = ax.inset_axes([1.0, 0.4, 0.40, 0.40])
    f3_inset = ax1.inset_axes([-0.4, 0.4, 0.40, 0.40])

    mark_inset(ax, f7_inset, loc1=2, loc2=2, edgecolor="white", alpha=0.1, lw=0.8)
    mark_inset(ax1, f3_inset, loc1=1, loc2=1, edgecolor="white", alpha=0.1, lw=0.8)

    mark_inset(ax, f7_temp, loc1=3, loc2=3, edgecolor="white", alpha=0.1, lw=0.8)
    mark_inset(ax1, f3_temp, loc1=4, loc2=4, edgecolor="white", alpha=0.1, lw=0.8)

    # f7_inset.xaxis.set_ticks_position("none")
    # f7_inset.yaxis.set_ticks_position("none")
    # f3_inset.xaxis.set_ticks_position("none")
    # f3_inset.yaxis.set_ticks_position("none")
    # f7_temp.xaxis.set_ticks_position("none")
    # f7_temp.yaxis.set_ticks_position("none")
    # f3_temp.xaxis.set_ticks_position("none")
    # f3_temp.yaxis.set_ticks_position("none")

    f7_lum_image = ax.imshow(
        np.log10(
            f7_lums / pxl_size,
            where=(f7_lums != 0),
            out=np.full_like(f7_lums, np.log10(lum_range[0])),
        ),
        cmap="inferno",
        origin="lower",
        extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
        vmin=np.log10(lum_range[0]),
        vmax=np.log10(lum_range[1]),
        alpha=lum_alpha,
    )
    ax1.imshow(
        np.log10(
            f3_lums / pxl_size,
            where=(f3_lums != 0),
            out=np.full_like(f3_lums, np.log10(lum_range[0])),
        ),
        cmap="inferno",
        origin="lower",
        extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
        vmin=np.log10(lum_range[0]),
        vmax=np.log10(lum_range[1]),
        alpha=lum_alpha,
    )

    f7_gas_image = ax.imshow(
        gaussian_filter(
            np.log10(
                f7_gas_array,
                where=(f7_gas_array != 0),
                out=np.full_like(f3_lums, np.log10(gas_range[0])),
            ),
            5,
        ),
        cmap="cubehelix",
        origin="lower",
        extent=[
            -plt_wdth / 2,
            plt_wdth / 2,
            -plt_wdth / 2,
            plt_wdth / 2,
        ],
        vmin=np.log10(gas_range[0]),
        vmax=np.log10(gas_range[1]),
        alpha=gas_alpha,
    )
    ax1.imshow(
        gaussian_filter(
            np.log10(
                f3_gas_array,
                where=(f3_gas_array != 0),
                out=np.full_like(f3_lums, np.log10(gas_range[0])),
            ),
            5,
        ),
        cmap="cubehelix",
        origin="lower",
        extent=[
            -plt_wdth / 2,
            plt_wdth / 2,
            -plt_wdth / 2,
            plt_wdth / 2,
        ],
        vmin=np.log10(gas_range[0]),
        vmax=np.log10(gas_range[1]),
        alpha=gas_alpha,
    )

    ax.set(
        xlim=(-plt_wdth / 2, plt_wdth / 2),
        ylim=(-plt_wdth / 2, plt_wdth / 2),
        xticklabels=[],
        yticklabels=[],
    )

    ax1.xaxis.set_ticks_position("none")
    ax1.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    # add some scale bar
    scale = patches.Rectangle(
        xy=(-plt_wdth / 2 * 0.25, -plt_wdth / 2 * 0.88),
        width=plt_wdth / 2 * 0.5,
        height=0.010 * plt_wdth / 2,
        linewidth=0,
        edgecolor="white",
        facecolor="white",
    )
    ax.text(
        0,
        -plt_wdth / 2 * 0.92,
        r"$\mathrm{{{:.0f} \: pc}}$".format(plt_wdth / 2 * 0.5),
        ha="center",
        va="center",
        color="white",
        # fontproperties=leg_font,
    )
    ax.add_patch(scale)

    # time  and redshift

    ax.text(
        0.05,
        0.95,
        (r"$\mathrm{{t = {:.2f} \: Myr}}$" "\n" r"$\mathrm{{z = {:.2f} }}$").format(
            f7_t_myr, f7_redshift
        ),
        ha="left",
        va="top",
        color="white",
        transform=ax.transAxes,
    )
    # efficiency labels
    ax.text(
        0.5,
        0.95,
        r"$\mathrm{high-SFE\:(70\%)}$",
        ha="center",
        va="top",
        color="white",
        transform=ax.transAxes,
    )
    ax1.text(
        0.5,
        0.95,
        r"$\mathrm{low-SFE\:(35\%)}$",
        ha="center",
        va="top",
        color="white",
        transform=ax1.transAxes,
    )

    # inset luminosity
    f7_inset.imshow(
        np.log10(
            f7_lums / pxl_size,
            where=(f7_lums != 0),
            out=np.full_like(f7_lums, np.log10(lum_range[0])),
        ),
        cmap="inferno",
        interpolation="gaussian",
        origin="lower",
        extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
        vmin=np.log10(lum_range[0]),
        vmax=np.log10(lum_range[1]),
        alpha=lum_alpha,
    )
    f7_inset.set(
        xlim=(-inset_width / 2, inset_width / 2),
        ylim=(-inset_width / 2, inset_width / 2),
        xticklabels=[],
        yticklabels=[],
    )
    f3_inset.imshow(
        np.log10(
            f3_lums / pxl_size,
            where=(f3_lums != 0),
            out=np.full_like(f3_lums, np.log10(lum_range[0])),
        ),
        cmap="inferno",
        origin="lower",
        extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
        vmin=np.log10(lum_range[0]),
        vmax=np.log10(lum_range[1]),
        alpha=lum_alpha,
    )
    f3_inset.set(
        xlim=(-inset_width / 2, inset_width / 2),
        ylim=(-inset_width / 2, inset_width / 2),
        xticklabels=[],
        yticklabels=[],
    )

    # inset gas
    f7_inset.imshow(
        gaussian_filter(
            np.log10(
                f7_gas_array,
                where=(f7_gas_array != 0),
                out=np.full_like(f3_lums, np.log10(gas_range[0])),
            ),
            5,
        ),
        cmap="cubehelix",
        origin="lower",
        extent=[
            -plt_wdth / 2,
            plt_wdth / 2,
            -plt_wdth / 2,
            plt_wdth / 2,
        ],
        # norm=LogNorm(gas_range[0], gas_range[1]),
        vmin=np.log10(gas_range[0]),
        vmax=np.log10(gas_range[1]),
        alpha=gas_alpha,
    )
    f3_inset.imshow(
        gaussian_filter(
            np.log10(
                f3_gas_array,
                where=(f3_gas_array != 0),
                out=np.full_like(f3_lums, np.log10(gas_range[0])),
            ),
            5,
        ),
        cmap="cubehelix",
        origin="lower",
        extent=[
            -plt_wdth / 2,
            plt_wdth / 2,
            -plt_wdth / 2,
            plt_wdth / 2,
        ],
        # norm=LogNorm(gas_range[0], gas_range[1]),
        vmin=np.log10(gas_range[0]),
        vmax=np.log10(gas_range[1]),
        alpha=gas_alpha,
    )

    f7_temp_image = f7_temp.imshow(
        gaussian_filter(
            np.log10(
                f7_temp_array,
                where=(f7_temp_array != 0),
                out=np.full_like(f7_temp_array, 0),
            ),
            5,
        ),
        cmap="gist_heat",
        origin="lower",
        extent=[
            -plt_wdth / 2,
            plt_wdth / 2,
            -plt_wdth / 2,
            plt_wdth / 2,
        ],
        vmin=np.log10(temp_norm[0]),
        vmax=np.log10(temp_norm[1]),
    )
    f3_temp.imshow(
        gaussian_filter(
            np.log10(
                f3_temp_array,
                where=(f3_temp_array != 0),
                out=np.full_like(f3_temp_array, 0),
            ),
            5,
        ),
        cmap="gist_heat",
        origin="lower",
        extent=[
            -plt_wdth / 2,
            plt_wdth / 2,
            -plt_wdth / 2,
            plt_wdth / 2,
        ],
        vmin=np.log10(temp_norm[0]),
        vmax=np.log10(temp_norm[1]),
    )

    # annotate star particles by age
    for unique_age in f7_unique_birth_times:
        mask = np.array(f7_rounded_times) == unique_age
        filtered_x = f7_x[mask]
        filtered_y = f7_y[mask]
        idx_of_nearest_c = np.argmin(np.abs(evenly_spaced_times - unique_age))
        color = cmap[idx_of_nearest_c]
        color = color.reshape(1, -1)
        f7_temp.scatter(
            filtered_x,
            filtered_y,
            marker=".",
            c=color,
            s=0.8,
            edgecolors=None,
            alpha=0.05,
        )
    for unique_age in f3_unique_birth_times:
        mask = np.array(f3_rounded_times) == unique_age
        filtered_x = f3_x[mask]
        filtered_y = f3_y[mask]
        idx_of_nearest_c = np.argmin(np.abs(evenly_spaced_times - unique_age))
        color = cmap[idx_of_nearest_c]
        color = color.reshape(1, -1)
        f3_temp.scatter(
            filtered_x,
            filtered_y,
            marker=".",
            c=color,
            s=0.8,
            edgecolors=None,
            alpha=0.05,
        )
    f7_temp.set(
        xlim=(-inset_width / 2, inset_width / 2),
        ylim=(-inset_width / 2, inset_width / 2),
        xticklabels=[],
        yticklabels=[],
    )
    f3_temp.set(
        xlim=(-inset_width / 2, inset_width / 2),
        ylim=(-inset_width / 2, inset_width / 2),
        xticklabels=[],
        yticklabels=[],
    )

    # declare the color bar axes
    lum_cbar_ax = ax.inset_axes([1.01, 0.89, 0.38, 0.04])
    lum_cbar = fig.colorbar(
        f7_lum_image, cax=lum_cbar_ax, pad=0, orientation="horizontal"
    )
    lum_cbar.ax.xaxis.set_tick_params(pad=-9)
    lum_cbar_ax.set_title(
        r"$\mathrm{\log\:Surface\:Brightness},$"
        "\n"
        r"$\mathrm{\lambda = 1500 \: \AA \:}$"
        r"$\:\mathrm{\left(erg\:\:s^{-1}\:\AA^{-1}\:pc^{-2}\right)}$",
        fontsize=7,
        pad=4,
    )

    star_cbar_ax = ax.inset_axes([1.41, 0.89, 0.38, 0.04])
    star_cb = mpl.colorbar.ColorbarBase(
        star_cbar_ax,
        norm=mpl.colors.Normalize(star_t_range[0], star_t_range[1]),
        # ticks = [340,405,470],
        orientation="horizontal",
        cmap="viridis",
    )
    star_cb.ax.xaxis.set_tick_params(pad=-9)
    star_cbar_ax.set_title(
        r"$\mathrm{Population\:II\:}$"
        "\n"
        "$\mathrm{Time\:of\:Birth}\:\mathrm{(Myr)}$",
        fontsize=7,
        pad=4,
    )

    gas_cbar_ax = ax.inset_axes([1.01, 0.840, 0.38, 0.04])
    gas_cbar = fig.colorbar(
        f7_gas_image, cax=gas_cbar_ax, pad=0, orientation="horizontal"
    )
    gas_cbar.ax.xaxis.set_tick_params(pad=-9)
    gas_cbar.set_label(
        label=r"$\mathrm{\log\:Gas\:Density\:\left(g \: cm^{-2}\right)}$", labelpad=1.5
    )

    temp_cbar_ax = ax.inset_axes([1.41, 0.840, 0.38, 0.04])
    temp_cbar = fig.colorbar(
        f7_temp_image, cax=temp_cbar_ax, pad=0, orientation="horizontal"
    )
    temp_cbar.ax.xaxis.set_tick_params(pad=-9)
    temp_cbar.set_label(label=r"$\mathrm{\log\:Temperature\:(K)}$", labelpad=1.5)

    output_path = os.path.join(
        sequence_dir,
        "render_{}_{}_{}.png".format(
            outnum_f7, outnum_f3, str(np.round(f3_redshift, 3)).replace(".", "_")
        ),
    )
    plt.savefig(
        os.path.expanduser(output_path), dpi=400, bbox_inches="tight", pad_inches=0.05
    )
    # plt.close("all")
    print(">Saved:", output_path)
