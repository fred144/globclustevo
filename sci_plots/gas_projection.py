import sys

sys.path.append("../")
import numpy as np
import os
from modules.macros import (
    filter_snapshots,
    characterisitc_mass,
    sci_notation,
    ram_fields,
)
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
import yt

f7_strt = 113
f7_end = 1000
f3_strt = 154
f3_end = 1177
step = 1

f7_pop2 = filter_snapshots(
    r"../particle_data/pop_2_data/fs07_refine", f7_strt, f7_end, step
)
# matched snapshots
f3_pop2 = filter_snapshots(
    r"../particle_data/pop_2_data/fs035_ms10", f3_strt, f3_end, step
)

# optional if you want to match them
# _, f3_matched_nums = find_matching_time(sequence=f7_pop2, look_up_sequence=f3_pop2)
# f3_pop2_matched = get_snapshots(f3_pop2, get_list=f3_matched_nums)

f7_sn_dir = "../../cosm_test_data/refine"
f3_sn_dir = "../../cosm_test_data/fs035_ms10"

f7_halo_dir = r"../halo_data/fs07_refine/fof_best"
f3_halo_dir = r"../halo_data/fs035_ms10/fof_best"

f7_halos = filter_snapshots(os.path.relpath(f7_halo_dir), f7_strt, f7_end)
f3_halos = filter_snapshots(os.path.relpath(f3_halo_dir), f3_strt, f3_end)

#!!! to change for actual plot
f7_sn = filter_snapshots(os.path.relpath(f7_sn_dir), f7_strt, 500)
f3_sn = filter_snapshots(os.path.relpath(f3_sn_dir), 500, 1102)

# dictate which snapshots will be plotted
f7_sn_list = np.array([250, 373, 500])
f3_sn_list = np.array([500, 500, 500])

f7_plt_sn = get_snapshots(f7_sn, get_list=f7_sn_list)
f3_plt_sn = get_snapshots(f3_sn, get_list=f3_sn_list)

f7_plt_p2 = get_snapshots(f7_pop2, get_list=f7_sn_list)
f3_plt_p2 = get_snapshots(f3_pop2, get_list=f3_sn_list)

f7_plt_halo = get_snapshots(f7_halos, get_list=f7_sn_list)
f3_plt_halo = get_snapshots(f3_halos, get_list=f3_sn_list)

slice_axis = "z"
width = (400, "pc")

# first entry is for .70, second is for .35
time = []
redshift = []
gas_list = []
star_coords_list = []
star_lums_list = []
star_t_birth_list = []


# for i, (f7_sn, f3_sn, f7_p2, f3_p2) in enumerate(
#     zip(f7_plt_sn, f3_plt_sn, f7_plt_p2, f3_plt_p2)
# ):

#     cell_fields, epf = ram_fields()
#     f7_info_file = os.path.join(f7_sn, "info_{}.txt".format(f7_sn.split("_")[-1]))
#     f3_info_file = os.path.join(f3_sn, "info_{}.txt".format(f3_sn.split("_")[-1]))

#     f7_ds = yt.load(f7_info_file, fields=cell_fields, extra_particle_fields=epf)
#     f3_ds = yt.load(f3_info_file, fields=cell_fields, extra_particle_fields=epf)
#     print(i, "Loaded Snapshot Data")
#     # get pre processed data from pop2 data sets
#     f7_t_myr = np.loadtxt(f7_p2, max_rows=2)[0, 6]
#     f7_redshift = np.loadtxt(f7_p2, max_rows=2)[1, 6]
#     f3_t_myr = np.loadtxt(f3_p2, max_rows=2)[0, 6]
#     f3_redshift = np.loadtxt(f3_p2, max_rows=2)[1, 6]

#     f7_code_ctr = np.loadtxt(f7_p2, max_rows=5)[2:5, 6]
#     f3_code_ctr = np.loadtxt(f3_p2, max_rows=5)[2:5, 6]

#     f7_field_stars = np.loadtxt(os.path.join(f7_plt_halo[i], "field_stars.txt"))
#     f7_bound_stars = np.loadtxt(os.path.join(f7_plt_halo[i], "bound_stars.txt"))
#     f7_stars = np.vstack((f7_field_stars, f7_bound_stars))
#     f7_star_ids = f7_stars[:, 0]
#     f7_star_lums = f7_stars[:, 2]
#     # f7_star_masses = f7_stars[:, 6]
#     f7_star_ages = f7_stars[:, 1]  # Myr
#     f7_star_bes = f7_t_myr - f7_star_ages
#     f7_pos_pc = f7_stars[:, 3:6]

#     f3_field_stars = np.loadtxt(os.path.join(f3_plt_halo[i], "field_stars.txt"))
#     f3_bound_stars = np.loadtxt(os.path.join(f3_plt_halo[i], "bound_stars.txt"))
#     f3_stars = np.vstack((f3_field_stars, f3_bound_stars))
#     # f3_star_ids = f3_stars[:, 0]
#     f3_star_lums = f3_stars[:, 2]
#     # f3_star_masses = f3_stars[:, 6]
#     f3_star_ages = f3_stars[:, 1]  # Myr
#     f3_star_bes = f3_t_myr - f3_star_ages
#     f3_pos_pc = f3_stars[:, 3:6]

#     # start getting gas data
#     f7_p = yt.ProjectionPlot(
#         f7_ds, slice_axis, ("gas", "density"), width=width, center=f7_code_ctr
#     )
#     f7_frb = f7_p.data_source.to_frb(width, 3000)
#     f7_gas = np.array(f7_frb["gas", "density"])

#     f3_p = yt.ProjectionPlot(
#         f3_ds, slice_axis, ("gas", "density"), width=width, center=f3_code_ctr
#     )
#     f3_frb = f3_p.data_source.to_frb(width, 3000)
#     f3_gas = np.array(f3_frb["gas", "density"])
#     print(i, "Integrated Gas")

#     time.append([f7_t_myr, f3_t_myr])
#     redshift.append([f7_redshift, f3_redshift])
#     gas_list.append([f7_gas, f3_gas])
#     star_coords_list.append([f7_pos_pc, f3_pos_pc])
#     star_lums_list.append([f7_star_lums, f3_star_lums])
#     star_t_birth_list.append([f7_star_bes, f3_star_bes])
#%%


rows = 3
cols = 5
star_lum_bin = 1500
pxl_size = width[0] / star_lum_bin
proj_r = width[0] / 2
axlims = (-100, 100)
star_map = cm.get_cmap("YlOrRd_r")
cmap = star_map(np.linspace(0, 1, 300))
evenly_spaced_times = np.arange(320, 600)  # 1 Myr intervals

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
        nrows=rows,
        ncols=cols,
        gridspec_kw={"width_ratios": [1, 1, 0.15, 1, 1]},
        figsize=(7, 5.25),
        dpi=400,
        sharex=True,
        sharey=True,
    )

    # time = []
    # redshift = []
    # gas_list = []
    # star_coords_list = []
    # star_lums_list = []
    # star_t_birth_list = []

    for idx, (t, z, gas, coord, lums, birth) in enumerate(
        zip(
            time,
            redshift,
            gas_list,
            star_coords_list,
            star_lums_list,
            star_t_birth_list,
        )
    ):
        f7_t = t[0]
        f7_z = z[0]
        f7_gas = gas[0]
        f7_coord = coord[0]
        f7_lums = lums[0]
        f7_birth = birth[0]

        f3_t = t[1]
        f3_z = z[1]
        f3_gas = gas[1]
        f3_coord = coord[1]
        f3_lums = lums[1]
        f3_birth = birth[1]

        ax[idx, 2].set_visible(False)

        # show projection plot
        f7_gas = ax[idx, 0].imshow(
            f7_gas,
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            cmap="cubehelix",
            norm=LogNorm(0.008, 0.35),
            origin="lower",
        )
        f3_gas = ax[idx, 3].imshow(
            f3_gas,
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            cmap="cubehelix",
            norm=LogNorm(0.008, 0.35),
            origin="lower",
        )

        # annotatet projection plot with stars
        # gets the clump ages, treats all clumps within 1 Myr as the from same
        f7_rounded_times = np.round_(f7_birth, 0)
        f7_unique_birth_times = np.unique(f7_rounded_times)
        for i, unique_age in enumerate(f7_unique_birth_times):

            mask = np.array(f7_rounded_times) == unique_age
            f7_filtered_x = f7_coord[:, 0][mask]
            f7_filtered_y = f7_coord[:, 1][mask]
            idx_of_nearest_c = np.argmin(np.abs(evenly_spaced_times - unique_age))
            color = cmap[idx_of_nearest_c]
            color = color.reshape(1, -1)
            ax[idx, 0].scatter(
                f7_filtered_x,
                f7_filtered_y,
                marker=".",
                c=color,
                s=0.5,
                edgecolors=None,
                alpha=0.02,
            )
        f3_rounded_times = np.round_(f3_birth, 0)
        f3_unique_birth_times = np.unique(f3_rounded_times)
        for i, unique_age in enumerate(f3_unique_birth_times):
            mask = np.array(f3_rounded_times) == unique_age
            f3_filtered_x = f3_coord[:, 0][mask]
            f3_filtered_y = f3_coord[:, 1][mask]
            idx_of_nearest_c = np.argmin(np.abs(evenly_spaced_times - unique_age))
            color = cmap[idx_of_nearest_c]
            color = color.reshape(1, -1)
            ax[idx, 3].scatter(
                f3_filtered_x,
                f3_filtered_y,
                marker=".",
                c=color,
                s=0.5,
                edgecolors=None,
                alpha=0.02,
            )

        # make the light projection plots
        f7_y = f7_coord[:, 1]
        f7_xy_lums, _, _ = np.histogram2d(
            f7_coord[:, 0],
            f7_coord[:, 1],
            bins=star_lum_bin,
            weights=f7_lums,
            normed=False,
            range=[[-proj_r, proj_r], [-proj_r, proj_r]],
        )
        f7_xy_lums = f7_xy_lums.T
        f3_xy_lums, _, _ = np.histogram2d(
            f3_coord[:, 0],
            f3_coord[:, 1],
            bins=star_lum_bin,
            weights=f3_lums,
            normed=False,
            range=[[-proj_r, proj_r], [-proj_r, proj_r]],
        )
        f3_xy_lums = f3_xy_lums.T
        f7_lums = ax[idx, 1].imshow(
            f7_xy_lums,
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            cmap="inferno",
            norm=LogNorm(vmin=2e32, vmax=5e35),
            origin="lower",
        )
        ax[idx, 1].set_facecolor(cm.Greys_r(0))
        f3_lums = ax[idx, 4].imshow(
            f3_xy_lums,
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            cmap="inferno",
            norm=LogNorm(vmin=2e32, vmax=5e35),
            origin="lower",
        )
        ax[idx, 4].set_facecolor(cm.Greys_r(0))

        for t in range(cols):
            ax[idx, t].spines["top"].set_color("white")
            ax[idx, t].spines["bottom"].set_color("white")
            ax[idx, t].spines["right"].set_color("white")
            ax[idx, t].spines["left"].set_color("white")
            ax[idx, t].tick_params(colors="white")
            ax[idx, t].set_yticklabels([])
            ax[idx, t].set_xticklabels([])
            ax[idx, t].xaxis.set_ticks_position("none")
            ax[idx, t].yaxis.set_ticks_position("none")

        ax[idx, 0].set_xlim(axlims)
        ax[idx, 0].set_ylim(axlims)
plt.subplots_adjust(hspace=0, wspace=-0.1)
