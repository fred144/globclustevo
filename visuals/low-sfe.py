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
import matplotlib
import matplotlib.patheffects as patheffects
from scipy.spatial.transform import Rotation as R
from yt.visualization.volume_rendering.api import Scene
from scipy.ndimage import gaussian_filter

yt.enable_parallelism()
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "font.size": 6,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)
plt.style.use("dark_background")


def draw_frame(gas_array, luminosity, ax, fig, wdth, t_myr, redshift, star_bins=2000):
    lum_range = (3e33, 3e36)
    pxl_size = (wdth / star_bins) ** 2
    lum_alpha = 1
    gas_alpha = 0.5

    # clean up edges
    ax.set_xticklabels([])
    ax.xaxis.set_ticks_position("none")
    ax.set_yticklabels([])
    ax.yaxis.set_ticks_position("none")
    # luminosity
    lum = ax.imshow(
        luminosity / pxl_size,
        cmap="inferno",
        interpolation="gaussian",
        origin="lower",
        extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
        norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
        alpha=lum_alpha,
    )

    # three panels gas density
    gas = ax.imshow(
        gaussian_filter(gas_array, sigma=5),
        cmap="cubehelix",
        # interpolation="gaussian",
        origin="lower",
        extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
        norm=LogNorm(0.008, 0.32),
        alpha=gas_alpha,
    )

    # add scale
    scale = patches.Rectangle(
        xy=(wdth / 2 * 0.35, -wdth / 2 * 0.85),
        width=wdth / 2 * 0.5,
        height=0.015 * wdth / 2,
        linewidth=0,
        edgecolor="white",
        facecolor="white",
        clip_on=False,
        alpha=0.8,
    )
    ax.text(
        wdth / 2 * 0.61,
        -wdth / 2 * 0.90,
        r"$\mathrm{{{:.0f}\:pc}}$".format(wdth / 2 * 0.5),
        ha="center",
        va="center",
        color="white",
        fontsize=8,
        alpha=0.8,
        # fontproperties=leg_font,
        # fontsize=14,
    )
    ax.add_patch(scale)

    # add the luminosity color bar
    lum_cbar_ax = ax.inset_axes([0.05, 0.10, 0.36, 0.027])
    lum_cbar = fig.colorbar(lum, cax=lum_cbar_ax, pad=-1, orientation="horizontal")
    lum_cbar.ax.xaxis.set_ticks_position("bottom")
    lum_cbar.ax.xaxis.set_label_position("bottom")
    lum_cbar.ax.xaxis.set_tick_params(pad=-8)
    lum_cbar_ax.set_title(
        r"$\mathrm{\log\:Surface\:Brightness},$"
        # "\n"
        r"$\mathrm{\lambda = 1500 \: \AA \:}$"
        r"$\:\mathrm{\left(erg\:\:s^{-1}\:\AA^{-1}\:pc^{-2}\right)}$",
        fontsize=6,
    )
    fig.canvas.draw()
    x_labels = [i.get_text().replace("10^", "") for i in lum_cbar_ax.get_xticklabels()]
    lum_cbar_ax.set_xticklabels(
        x_labels,
        path_effects=[patheffects.withStroke(linewidth=1, foreground="black")],
    )
    # add the gas color bar
    gas_cbar_ax = ax.inset_axes([0.05, 0.06, 0.36, 0.027])
    gas_cbar = fig.colorbar(gas, cax=gas_cbar_ax, pad=0, orientation="horizontal")
    gas_cbar.ax.xaxis.set_tick_params(pad=-8)
    gas_cbar.set_label(
        label=r"$\mathrm{\log\:Gas\:Density\:(g \: cm^{-2})}$",
        labelpad=1,
    )
    fig.canvas.draw()
    x_labels = [i.get_text().replace("10^", "") for i in gas_cbar_ax.get_xticklabels()]
    gas_cbar_ax.set_xticklabels(
        x_labels,
        path_effects=[patheffects.withStroke(linewidth=1, foreground="black")],
    )

    # add time and redshift
    ax.text(
        0.05,
        0.96,
        (
            "$\mathrm{{low-SFE\: (35 \%)}}$"
            "\n"
            r"$\mathrm{{t = {:.2f} \: Myr}}$"
            "\n"
            r"$\mathrm{{z = {:.2f} }}$"
        ).format(
            t_myr,
            redshift,
        ),
        ha="left",
        va="top",
        color="white",
        transform=ax.transAxes,
        fontsize=8,
        # bbox={
        #     "boxstyle": "Square",
        #     # have control over edge alpha and face alpha
        #     "facecolor": colors.to_rgba("black")[:-1] + (0.5,),
        #     "linewidth": 1,
        #     "edgecolor": "white",
        #     "pad": 0.5,
        # },
    )

    # add fancy axes indicators
    # ax_inset = ax.inset_axes([0.03, 0.03, 0.15, 0.15])
    # misc_visuals.xy_ax_indicator(ax_inset)
    # ax_inset.set_alpha(0)


# fig, ax = plt.subplots(
#     figsize=(7, 7),
#     dpi=400,
#     facecolor=cm.Greys_r(0),
# )
# draw_frame(gas_array, lums, ax, fig, plt_wdth, t_myr, redshift)
# output_path = os.path.join(
#     sequence_dir,
#     "lowsfe_{}_{}.png".format(output_num_string, str(rot_idx).zfill(3)),
# )
# plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)

#%%
strt = 450
end = 450
step = 1
efficiency = 0.35
sim_run = "fs035_ms10"
master_data_dir = (
    "/afs/shell.umd.edu/project/ricotti-prj/user/fgarcia4/dwarf/data/cluster_evolution/"
)
snap_dir = os.path.join(master_data_dir, sim_run)
# snap_dir = os.path.relpath("../../cosm_test_data/fs035_ms10/")
halo_data_directory = r"../halo_data/{}/fof_best".format(sim_run)
pop2_data_directory = r"../particle_data/pop_2_data/{}".format(sim_run)
snapshots = filter_snapshots(snap_dir, strt, end, 1)

sequence_dir = "../rendering/gas_lum/{}/lowsfe_static_00450".format(sim_run)
if not os.path.exists(sequence_dir):
    print("# Creating new sequence directory", sequence_dir)
    os.makedirs(sequence_dir)


pop2 = filter_snapshots(pop2_data_directory, strt, end, step)
halo_ds = filter_snapshots(halo_data_directory, strt, end, step)

static_plt_wdth = 460
star_bins = 2000
pxl_size = (static_plt_wdth / star_bins) ** 2  # pc
lum_range = (3e33, 3e36)  # (2e32, 5e35)
gas_alpha = 0.5
lum_alpha = 1
cell_fields, epf = ram_fields()
cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))

total_pan_frames = 800
num_rots = 4
rotation_interval = (
    np.linspace(0, 2 * num_rots, total_pan_frames, endpoint=False) * np.pi
)
zoom_interval = np.concatenate(
    [
        static_plt_wdth * np.ones(int(total_pan_frames / 4)),
        np.linspace(static_plt_wdth, 150, int(total_pan_frames / 4)),
        150 * np.ones(int(total_pan_frames / 4)),
        np.linspace(150, static_plt_wdth, int(total_pan_frames / 4)),
    ]
)
pause_and_rotate = [
    450,
]
#%%

for idx, (sn, p2, h_ds) in enumerate(zip(snapshots, pop2, halo_ds)):
    output_num_string = h_ds.split("/")[-1].split("_")[-1]
    if int(output_num_string) == 1102:
        continue
    info_file = os.path.join(sn, "info_{}.txt".format(output_num_string))
    ram_ds = yt.load(info_file, fields=cell_fields, extra_particle_fields=epf)

    # here we will use post-processed particle data from the profiler,
    # in case we want luminosities too.
    code_ctr = np.loadtxt(p2, max_rows=5)[2:5, 6]
    t_myr = np.loadtxt(p2, max_rows=2)[0, 6]
    redshift = np.loadtxt(p2, max_rows=2)[1, 6]
    print(t_myr, redshift)
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

    if int(output_num_string) in pause_and_rotate:
        # reset the star positions every loop
        print("Rotating View")
        for rot_idx, rotation_angle in enumerate(rotation_interval[254:], start=253):
            plt_wdth = zoom_interval[rot_idx]
            star_positions = stars[:, 3:6]
            # along (x,y,z) axis
            r = R.from_rotvec(rotation_angle * np.array([0, 1, 0]))
            rotation_matrix = r.as_matrix()
            # rotation_vector = r.as_rotvec()
            print("rotation angle", rotation_angle, "of", rotation_interval[-1])
            # rotate stars
            rotated_star_positions = np.dot(star_positions, rotation_matrix)
            lums, _, _ = np.histogram2d(
                rotated_star_positions[:, 0],
                rotated_star_positions[:, 1],
                bins=star_bins,
                weights=star_lums,
                normed=False,
                range=[
                    [-plt_wdth / 2, plt_wdth / 2],
                    [-plt_wdth / 2, plt_wdth / 2],
                ],
            )
            lums = lums.T

            # cam = sc.add_camera()
            # # rotate the camera by pi / 4 radians:
            # cam.rotate(np.pi / 4.0)
            # rotate the camera about the y-axis instead of cam.north_vector:
            # cam.rotate(rotation_angle, np.array([0.0, 1.0, 0.0]))
            # sc = yt.create_scene(ram_ds)
            # cam = sc.add_camera(ram_ds, lens_type="parallel")
            # cam.focus = code_ctr
            # cam.rotate(rotation_angle, np.array([0.0, 1.0, 0.0]))
            # # for i in cam.iter_rotate(np.pi, 10):
            # #     im = sc.render()
            # #     sc.save("rotation_%04i.png" % i)
            # print(rotation_vector)
            gas = yt.OffAxisProjectionPlot(
                ram_ds,
                np.dot(np.array([0.0, 0.0, 1.0]), rotation_matrix),
                fields=("gas", "density"),
                center=code_ctr,
                north_vector=np.array([0.0, 1.0, 0.0]),
                width=(plt_wdth, "pc"),
                # resolution=2000,
            )
            # gas = yt.ProjectionPlot(
            #     ram_ds,
            #     np.dot(np.array([0.0, 0.0, 1.0]), rotation_matrix),
            #     ("gas", "density"),
            #     width=(plt_wdth, "pc"),
            #     center=code_ctr,
            # north_vector=np.array([0.0, 1.0, 0.0]),
            # normal_vector=np.dot(np.array([0.0, 0.0, 1.0]), rotation_matrix),
            # )
            gas_frb = gas.to_fits_data()
            # gas_frb = gas.data_source.to_frb((plt_wdth, "pc"), star_bins)
            # gas_array = np.array(gas_frb["gas", "density"])

            #     ram_ds,
            #     "z",
            #     ("gas", "density"),
            #     width=(plt_wdth, "pc"),
            #     center=code_ctr,
            # )

            # gas_frb = gas.data_source.to_frb((plt_wdth, "pc"), star_bins)
            gas_array = np.array(gas_frb[0].data)  # .T

            fig, ax = plt.subplots(
                figsize=(7, 7),
                dpi=400,
                facecolor=cm.Greys_r(0),
            )
            draw_frame(gas_array, lums, ax, fig, plt_wdth, t_myr, redshift)
            output_path = os.path.join(
                sequence_dir,
                "lowsfe_{}_{}.png".format(output_num_string, str(rot_idx).zfill(3)),
            )
            plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
            print(">Saved:", output_path)
            plt.close()
    else:
        plt_wdth = static_plt_wdth
        lums, _, _ = np.histogram2d(
            x,
            y,
            bins=star_bins,
            weights=star_lums,
            normed=False,
            range=[
                [-plt_wdth / 2, plt_wdth / 2],
                [-plt_wdth / 2, plt_wdth / 2],
            ],
        )
        lums = lums.T

        gas = yt.ProjectionPlot(
            ram_ds,
            "z",
            ("gas", "density"),
            width=(plt_wdth, "pc"),
            center=code_ctr,
        )
        gas_frb = gas.data_source.to_frb((plt_wdth, "pc"), star_bins)
        gas_array = np.array(gas_frb["gas", "density"])
        fig, ax = plt.subplots(
            figsize=(7, 7),
            dpi=400,
            facecolor=cm.Greys_r(0),
        )
        draw_frame(gas_array, lums, ax, fig, plt_wdth, t_myr, redshift)

        output_path = os.path.join(
            sequence_dir, "lowsfe_{}.png".format(output_num_string)
        )
        # plt.show()
        plt.savefig(
            os.path.expanduser(output_path),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )

        print(">Saved:", output_path)
        plt.close()
