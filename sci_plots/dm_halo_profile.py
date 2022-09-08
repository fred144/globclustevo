import sys

sys.path.append("..")
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
from modules.macros import filter_snapshots, t_myr_from_z
import matplotlib.lines as mlines
from modules.match_t_sims import find_matching_time, get_snapshots
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from modules.profiles.profile_functions import projected_surf_densities

fs070 = filter_snapshots("../particle_data/pop_2_data/fs07_refine", 115, 1195, 1)
fs035 = filter_snapshots("../particle_data/pop_2_data/fs035_ms10", 154, 1364, 1)
# find matching fs = 0.35 snapshots in terms of time to fs = 0.70
# smaller goes fist
# in general, use the simulation with more snapshots as a lookup table and match
# the ones with less
_, f7_matched_nums = find_matching_time(sequence=fs035, look_up_sequence=fs070)

fs070_dat_dir = r"../dm/fs07_refine/dm_hop"
fs035_dat_dir = r"../dm/fs035_ms10/dm_hop"

fs035_matched = filter_snapshots(fs035_dat_dir, 154, 1364, 1)[245:246:600]

fs070_matched = get_snapshots(
    snapshot_file_list=filter_snapshots(fs070_dat_dir, 115, 1195, 1),
    get_list=f7_matched_nums,
    verbose=False,
)[245:246:600]

bins = 800
plt_rad = 2000  # pc
pxl_size = (plt_rad * 2 / bins) ** 2
profile_plot_bins = 100
for i, f3 in enumerate(fs035_matched):
    f3_dm_part_data = np.loadtxt(os.path.join(f3, "dm_data.txt"))

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
            ax.scatter([0], [0])
            ax.set_facecolor(cm.viridis(0))
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # ax.xaxis.set_ticks_position("none")
            # ax.yaxis.set_ticks_position("none")

            with plt.rc_context(
                {
                    "xtick.labelsize": 7,
                    "ytick.labelsize": 7,
                    "font.size": 10,
                }
            ):
                prof = ax.inset_axes([0.10, 0.10, 0.3, 0.3])

            r, rho, err, _, _, _, half_r = projected_surf_densities(
                x_coord=f3_x,
                y_coord=f3_y,
                lums=np.zeros_like(f3_x),
                masses=f3_mass,
                radius=halo_radius,
                num_bins=profile_plot_bins,
                log_bins=True,
                dr=None,
                calc_half_r=True,
            )
