"""
Script that gets the data as well as (projected) gas density. 
Must be ran in the direcotry.
"""

import sys

sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/master/lib/python3.7/site-packages"
)
sys.path.append("..")  # makes sure that importing the modules work

from modules.macros import code_age_to_myr
import matplotlib as mpl
from matplotlib import cm
from yt.funcs import mylog
import numpy as np
import yt
import os
import warnings

#yt.enable_parallelism()
#mylog.setLevel(40)
#warnings.simplefilter(action="ignore", category=RuntimeWarning)

# ==============================================================================
# TODO: edit for rendering runs
simulation_run_name = "fs07_refine"
latest_sim_stats = np.loadtxt(
    "../sim_log_files/{}/latest_sim_stats.txt".format(simulation_run_name)
)
# ===================================local test=================================
# datadir = os.path.relpath("../../cosm_test_data/refine")
# parent_folder = "../rendering"
# sequence_folder = "test_frames"
# ===================================dt2 paths=================================
datadir = os.path.expanduser(
    "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/{}"  # lustre data path
).format(simulation_run_name)
# save path
parent_folder = "../rendering/gas/{}".format(simulation_run_name)

# TODO: edit for rendering runs
sequence_folder = "lin_gas_projected_density_x"
# ===================================save path=================================

pop_2_save = "../particle_data/pop_2_data/{}".format(simulation_run_name)
sfc_save = "../particle_data/sfc_data/{}".format(simulation_run_name)
psc_save = "../particle_data/psc_data/{}".format(simulation_run_name)


newpath = os.path.join(parent_folder, sequence_folder)
if not os.path.exists(newpath):
    print("# Creating new sequence directory", newpath)
    os.makedirs(newpath)
if not os.path.exists(pop_2_save):
    print("# Creating new sequence directory", pop_2_save)
    os.makedirs(pop_2_save)
if not os.path.exists(sfc_save):
    print("# Creating new sequence directory", sfc_save)
    os.makedirs(sfc_save)
if not os.path.exists(psc_save):
    print("# Creating new sequence directory", psc_save)
    os.makedirs(psc_save)
# ===================================plot params=================================

# TODO: edit for rendering runs
sequence_title = "x_gas_lin"
slice_axis = "x"
width = (400, "pc")
start_step = 897  # fs07:113, fs035:154
end_step = 918
# cosmetics
mpl.rc("font", family="serif")
clrmap = "BuGn_r"  # for the pop II ages
density_cmap = "inferno"  # "cmyt.dusk"
z_scale = "lin"
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# https://yt-project.org/doc/visualizing/colormaps/index.html
star_map = cm.get_cmap(clrmap)

# snapshot 115 to 452 roughly spans 340 to 470 myr
# ===================================MAIN=================================
for loop_num, output_num in enumerate(range(start_step, end_step + 1)):
    print("#________________________________________________________________")
    infofile = os.path.abspath(
        datadir + f"/output_{output_num:05}/info_{output_num:05}.txt"
    )
    print("# reading in", infofile)

    # read fields explicitly, not recognized by YT from this ver of RAMSES
    cell_fields = [
        "Density",
        "x-velocity",
        "y-velocity",
        "z-velocity",
        "Pressure",
        "Metallicity",
        # "dark_matter_density",
        "xHI",
        "xHII",
        "xHeII",
        "xHeIII",
    ]
    epf = [
        ("particle_family", "b"),
        ("particle_tag", "b"),
        ("particle_birth_epoch", "d"),
        ("particle_metallicity", "d"),
    ]

    # read in RAMSES data set
    ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)

    # get time-dependent params.
    redshft = ds.current_redshift
    current_hubble = ds.hubble_constant
    current_time = float(ds.current_time.in_units("Myr"))

    # extract all data fields
    ad = ds.all_data()

    # get SFC/PSC positions and other important fields,
    # need to modify definitions to get these sinks
    pos_sfcs = np.array(ad["SFC", "particle_position"])
    pos_pscs = np.array(ad["PSC", "particle_position"])

    # read POPII star info
    star_id = np.array(ad["star", "particle_identity"])
    be_star = ad["star", "particle_birth_epoch"]
    x_pos = np.array(ad["star", "particle_position_x"])
    y_pos = np.array(ad["star", "particle_position_y"])
    z_pos = np.array(ad["star", "particle_position_z"])

    # center based on star position distribution
    x_center = np.mean(x_pos)
    y_center = np.mean(y_pos)
    z_center = np.mean(z_pos)
    plt_ctr = np.array([x_center, y_center, z_center])
    plt_ctr_in_pc = np.array(ds.arr(plt_ctr, "code_length").to("pc"))

    # translate points to center
    x_pos = x_pos - plt_ctr[0]
    y_pos = y_pos - plt_ctr[1]
    z_pos = z_pos - plt_ctr[2]

    pos_sfcs_recentered = pos_sfcs - plt_ctr
    pos_pscs_recentered = pos_pscs - plt_ctr

    p = yt.ProjectionPlot(
        ds, slice_axis, ("gas", "density"), width=width, center=plt_ctr
    )

    # aesthetics
    p.set_font(
        {
            "family": "serif",
            "size": 14,
        }
    )
    p.annotate_timestamp(
        corner="upper_left",
        time_format="t = {time:.2f} {units}",
        time_unit="Myr",
        redshift=True,
    )
    p.annotate_scale(
        corner="lower_right",
        coeff=width[0] / 4,
        unit="pc",
        text_args={"size": 12, "family": "serif"},
    )
    p.set_cmap(field=("gas", "density"), cmap=density_cmap)

    if z_scale == "log":
        # log scale
        p.set_zlim("density", 0.008, 0.35)
        p.set_log(("gas", "density"), True)
    elif z_scale == "lin":
        # linear scale
        p.set_zlim("density", 0.005, 0.34)  # default
        p.set_log(("gas", "density"), False)
    else:
        print("* scale not supported!")

    p.set_colorbar_label(("gas", "density"), r"Projected Gas Density (g cm$^{-2}$)")
    p.hide_axes(draw_frame=True)

    print("> annotating", np.array(be_star).size, "star particles")

    # particle clumps by age; converts code age to relative ages
    unique_birth_epochs = code_age_to_myr(
        ad["star", "particle_birth_epoch"], current_hubble, unique_age=True
    )
    # calculate the age of the universe when the first star was born
    # using the logSFC as a reference point for redshift when the first star
    # was born. Every age is relative to this. Due to our mods of ramses.
    log_sfc = np.loadtxt("../sim_log_files/{}/logSFC".format(simulation_run_name))
    birth_start = np.round_(
        float(ds.cosmology.t_from_z(log_sfc[0, 2]).in_units("Myr")), 0
    )

    # pop II birth color bar
    time_range = [int(birth_start), int(np.ceil(latest_sim_stats[1]))]  # Myr
    print("> star age range:", time_range, "Myr")
    evenly_spaced_times = np.arange(time_range[0], time_range[1] + 5)  # Myr
    cmap = star_map(np.linspace(0, 1, time_range[1] - time_range[0]))

    # gets the clump ages, treats all clumps within 1 Myr as the from same
    unique_birth_epochs = np.unique(np.round_(unique_birth_epochs, 0))

    # all the birth epochs of the stars
    converted_unfiltered = code_age_to_myr(
        ad["star", "particle_birth_epoch"], current_hubble, unique_age=False
    )
    # treats all clusters within 1 Myr birth epoch as same birth epoch
    # the first output with star in it was t ~ 339.562 for fs07
    # have yet to figure out how to calculate absolute times
    # just relative for now
    converted_unfiltered_rounded = np.round_(converted_unfiltered, 0) + birth_start

    # pop II annotate loop
    for i, unique_age in enumerate(unique_birth_epochs + birth_start):

        mask = np.array(converted_unfiltered_rounded) == unique_age
        filtered_x = ds.arr(x_pos, "code_length").to("pc")[mask]
        filtered_y = ds.arr(y_pos, "code_length").to("pc")[mask]
        filtered_z = ds.arr(z_pos, "code_length").to("pc")[mask]

        idx_of_nearest_c = np.argmin(np.abs(evenly_spaced_times - unique_age))
        # print(idx_of_nearest_c)
        color = cmap[idx_of_nearest_c]
        color = color.reshape(1, -1)

        if slice_axis == "z":
            p["gas", "density"].axes.scatter(
                filtered_x,
                filtered_y,
                marker=".",
                c=color,
                s=1,
                edgecolors=None,
                alpha=0.03,
            )
        elif slice_axis == "x":
            p["gas", "density"].axes.scatter(
                filtered_y,
                filtered_z,
                marker=".",
                c=color,
                s=1,
                edgecolors=None,
                alpha=0.03,
            )
        elif slice_axis == "y":
            p["gas", "density"].axes.scatter(
                filtered_z,
                filtered_x,
                marker=".",
                c=color,
                s=1,
                edgecolors=None,
                alpha=0.03,
            )
        else:
            print("Invalid slice axis.")
    # cbar_fig.style.use('dark_background')

    # pop II birth epoch color bar
    cbar_fig = p.plots[("gas", "density")].figure
    ax = cbar_fig.add_axes([0.31, 0.91, 0.3, 0.015])
    cb = mpl.colorbar.ColorbarBase(
        ax,
        norm=mpl.colors.Normalize(time_range[0], time_range[1]),
        # ticks = [340,405,470],
        orientation="horizontal",
        cmap=clrmap,
        # label='Birth Epoch (Myr)'
    )
    cb.ax.tick_params(colors="white", labelsize=6)
    for t in cb.ax.xaxis.get_ticklabels():
        t.set_family("serif")
    ax.set_title(
        "Pop II Birth Time (Myr) | Count: {:.2e}".format(np.size(be_star)),
        c="white",
        fontsize=9,
        fontfamily="serif",
    )

    # axes guides
    p_ax = p.plots[("gas", "density")].axes
    if slice_axis == "z":
        p_ax.text(
            -width[0] * 0.375,
            -width[0] * 0.4625,
            "X",
            size=7,
            ha="center",
            va="center",
            color="white",
        )
        p_ax.text(
            -width[0] * 0.4625,
            -width[0] * 0.375,
            "Y",
            size=7,
            ha="center",
            va="center",
            color="white",
        )
    elif slice_axis == "x":
        p_ax.text(
            -width[0] * 0.375,
            -width[0] * 0.4625,
            "Y",
            size=7,
            ha="center",
            va="center",
            color="white",
        )
        p_ax.text(
            -width[0] * 0.4625,
            -width[0] * 0.375,
            "Z",
            size=7,
            ha="center",
            va="center",
            color="white",
        )
    elif slice_axis == "y":
        p_ax.text(
            -width[0] * 0.375,
            -width[0] * 0.4625,
            "Z",
            size=7,
            ha="center",
            va="center",
            color="white",
        )
        p_ax.text(
            -width[0] * 0.4625,
            -width[0] * 0.375,
            "X",
            size=8,
            ha="center",
            va="center",
            color="white",
        )
    else:
        print("Invalid slice axis.")
    p_ax.arrow(
        -width[0] * 0.4625,
        -width[0] * 0.4625,
        width[0] * 0.075,
        0,
        head_width=width[0] * 0.0075,
        head_length=width[0] * 0.0075,
        linewidth=width[0] * 0.00125,
        color="w",
        length_includes_head=True,
    )
    p_ax.arrow(
        -width[0] * 0.4625,
        -width[0] * 0.4625,
        0,
        width[0] * 0.075,
        head_width=width[0] * 0.0075,
        head_length=width[0] * 0.0075,
        linewidth=width[0] * 0.00125,
        color="w",
        length_includes_head=True,
    )

    # ==========================luminosity mappping data extraction==============

    # get popII star positons
    abs_birth_epochs = np.round(converted_unfiltered + birth_start, 3)  #!
    current_ages = np.round(current_time, 3) - np.round(abs_birth_epochs, 3)
    extra_info = np.array(
        [np.concatenate((np.array([current_time, redshft]), plt_ctr, plt_ctr_in_pc))]
    )
    extra_info.resize(np.size(current_ages))
    star_info = np.array(
        [
            star_id,
            current_ages,
            ds.arr(x_pos, "code_length").to("pc"),
            ds.arr(y_pos, "code_length").to("pc"),
            ds.arr(z_pos, "code_length").to("pc"),
            ds.arr(ad["star", "particle_mass"], "code_mass").to("msun"),
            extra_info,
        ]
    )

    # =========================star positions save=================================

    star_info = np.array(star_info).T
    save_time = str(format(current_time, ".2f")).replace(".", "_")
    save_name = "{}/pos_{:05d}_{}_myr.txt".format(pop_2_save, output_num, save_time)
    header = (
        "\t\tID"
        "\t\tCurrentAges[MYr]"
        "\t\tX[pc]"
        "\t\tY[pc]\t\t"
        "Z[pc]\t\t"
        "mass[Msun]"
        "\t\tt_sim[Myr], z, ctr(code), ctr(pc)"
    )
    np.savetxt(save_name, X=star_info, header=header)
    print("# saved:", save_name)

    # =========================== psc sfc save==================================

    psc_kazu_radii = np.abs(
        ds.arr(ad["PSC", "particle_metallicity"], "code_length").to("pc")
    )
    sfc_kazu_radii = np.abs(
        ds.arr(ad["SFC", "particle_metallicity"], "code_length").to("pc")
    )
    pos_pscs = ds.arr(pos_pscs_recentered, "code_length").to("pc")
    pos_sfcs = ds.arr(pos_sfcs_recentered, "code_length").to("pc")

    # particle tags, see if unique
    psc_tag = np.array(ad["PSC", "particle_index"])
    sfc_tag = np.array(ad["SFC", "particle_index"])

    # save paths
    psc_path = "{}/psc_{:05d}_{}_myr.txt".format(psc_save, output_num, save_time)
    sfc_path = "{}/sfc_{:05d}_{}_myr.txt".format(sfc_save, output_num, save_time)
    # x(pc), y(pc), z(pc),radii at birth (pc), particle tag
    psc_save_data = np.concatenate(
        (pos_pscs, psc_kazu_radii[:, None], psc_tag[:, None]), axis=1
    )
    sfc_save_data = np.concatenate(
        (pos_sfcs, sfc_kazu_radii[:, None], sfc_tag[:, None]), axis=1
    )
    test_particale_header = "x(pc), y(pc), z(pc),radii at birth (pc), particle tag"
    print("# saved:", psc_path)
    print("# saved:", sfc_path)
    np.savetxt(psc_path, X=psc_save_data, header=test_particale_header)
    np.savetxt(sfc_path, X=sfc_save_data, header=test_particale_header)

    # save the frame
    save_path = str(
        "{}/{}/out-{}-z-{}-t-{}-{}.png".format(
            parent_folder,
            sequence_folder,
            str(output_num).zfill(5),
            str(format(redshft, ".2f")).replace(".", "_"),
            str(format(current_time, ".2f")).replace(".", "_"),
            sequence_title.replace(" ", "-"),
        )
    )
    p.save(
        save_path,
        mpl_kwargs={
            "bbox_inches": "tight",
            "dpi": 300,
            "pad_inches": 0.1
            # 'facecolor': 'black'
        },
    )
    # p.show()
    print("# saved:", save_path)
