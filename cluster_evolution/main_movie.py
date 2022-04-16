import sys

sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/cosmology-clean/lib/python3.7/site-packages"
)

from macros import code_age_to_yr  # , succ_distance
import matplotlib as mpl
from matplotlib import cm
from yt.funcs import mylog
import numpy as np
import yt
import os
import warnings


mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# ==============================================================================
simulation_run_name = "fs07_refine"
# ===================================local test=================================
# datadir = os.path.relpath("../../cosm_test_data/refine")
# # parent_folder = 'C:/Users/144/Desktop/AstroSimulationResearch/cluster_evolution'
# parent_folder = "."
# sequence_folder = "test_frames"
# ===================================DT2 Paths=================================
datadir = os.path.expanduser(
    "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/{}"  # lustre data path
).format(simulation_run_name)
# save path
sequence_folder = "gas_projected_density_z"
parent_folder = "../rendering/gas/{}".format(simulation_run_name)

pop_2_save = "../pop_2_data/{}".format(simulation_run_name)

newpath = os.path.join(parent_folder, sequence_folder)
if not os.path.exists(newpath):
    print("# Creating new sequence directory", newpath)
    os.makedirs(newpath)
if not os.path.exists(pop_2_save):
    print("# Creating new sequence directory", pop_2_save)
    os.makedirs(pop_2_save)
# ===================================plot params=================================
sequence_title = "z_gas"
width = (400, "pc")
slice_axis = "z"
start_step = 800
end_step = 810

# ctr_shift_thresh = 0.00060 #code length
# ctr_shift_thresh =  0.000001 #code length
# max_density_coords = []

# cosmetics
clrmap = "BuGn_r"  # for the pop II ages
mpl.rc("font", family="serif")
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
star_map = cm.get_cmap(clrmap)

# snapshot 115 to 452 roughly spans 340 to 470 myr

# pop II birth color bar kwargs
birth_start = 339.562  # Myr, need to figure out how to got to absolute ages
time_range = (300, 570)  # Myr
evenly_spaced_times = np.arange(time_range[0], time_range[1] + 1)
cmap = star_map(np.linspace(0, 1, time_range[1] - time_range[0]))

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
    p.set_cmap("density", "inferno")
    # p.set_zlim('density', 0.005, .34)                   # default
    p.set_zlim("density", 0.008, 0.35)
    p.set_log(("gas", "density"), True)
    p.set_colorbar_label(("gas", "density"), r"Projected Gas Density (g cm$^{-2}$)")
    p.hide_axes(draw_frame=True)

    print("> annotating", np.array(be_star).size, "star particles")

    # particle clumps by age; converts code age to relative ages
    unique_birth_epochs = code_age_to_yr(
        ad["star", "particle_birth_epoch"], current_hubble
    )

    # gets the clump ages, treats all clumps within 1 Myr as the from same
    unique_birth_epochs = np.unique(np.round_(unique_birth_epochs, 0))

    # all the birth epochs of the stars
    converted_unfiltered = code_age_to_yr(
        ad["star", "particle_birth_epoch"], current_hubble, unique=False
    )
    # treats all clusters within 1 Myr birth epoch as same birth epoch
    # the first output with star in it was t = 339.562 for fs07
    # have yet to figure out how to calculate absolute times
    # just relative for now
    converted_unfiltered_rounded = np.round_(converted_unfiltered, 0) + birth_start

    # pop II annotate loop
    for i, unique_age in enumerate(unique_birth_epochs + birth_start):

        print(unique_age)

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
                filtered_x, filtered_y, marker=".", c=color, s=0.0005, alpha=1
            )
        elif slice_axis == "x":
            p["gas", "density"].axes.scatter(
                filtered_y, filtered_z, marker=".", c=color, s=0.0005, alpha=1
            )
        elif slice_axis == "y":
            p["gas", "density"].axes.scatter(
                filtered_z, filtered_x, marker=".", c=color, s=0.0005, alpha=1
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

    # AXES GUIDE
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

    # =============================luminosity mappping data extraction==================

    # get star positons
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

    # =============================================================================
    # star positions save
    star_info = np.array(star_info).T
    save_time = str(format(current_time, ".2f")).replace(".", "_")
    save_name = "../pop_2_data/{}/pos_{:05d}_{}_myr.txt".format(
        simulation_run_name, output_num, save_time
    )
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

    # =============================================================================
    # psc sfc save

    # psc_kazu_radii = np.abs(
    #     ds.arr(ad["PSC", "particle_metallicity"], "code_length").to("pc")
    # )
    # sfc_kazu_radii = np.abs(
    #     ds.arr(ad["SFC", "particle_metallicity"], "code_length").to("pc")
    # )
    # pos_pscs = ds.arr(pos_pscs_recentered, "code_length").to("pc")
    # pos_sfcs = ds.arr(pos_sfcs_recentered, "code_length").to("pc")

    # # particle tags, see if unique
    # psc_tag = np.array(ad["PSC", "particle_index"])
    # sfc_tag = np.array(ad["SFC", "particle_index"])

    # # save paths
    # psc_path = "../luminosity_mapping/psc_data/psc_{:05d}_{}_myr.txt".format(
    #     output_num, save_time
    # )
    # sfc_path = "../luminosity_mapping/sfc_data/sfc_{:05d}_{}_myr.txt".format(
    #     output_num, save_time
    # )
    # # x,y,z,radii at birth (pc), particle tag
    # psc_save_data = np.concatenate(
    #     (pos_pscs, psc_kazu_radii[:, None], psc_tag[:, None]), axis=1
    # )
    # sfc_save_data = np.concatenate(
    #     (pos_sfcs, sfc_kazu_radii[:, None], sfc_tag[:, None]), axis=1
    # )
    # print("# saved:", psc_path)
    # print("# saved:", sfc_path)
    # np.savetxt(psc_path, X=psc_save_data)
    # np.savetxt(sfc_path, X=sfc_save_data)
    # =============================================================================

    # from yt.extensions.astro_analysis.halo_analysis import HaloCatalog

    # hc = HaloCatalog(
    #     ds,
    #     finder_method="hop",
    #     finder_kwargs={
    #         "ptype": "star",
    #         "padding": 0.1,
    #         "link": 0.2,
    #         "dm_only": False,
    #     },
    #     output_dir="../halo_data/",
    # )

    # hc.create()

    # p.annotate_particles(width=width, ptype="star", p_size=20.0, marker=".", col="r")
    # p.annotate_halos(hc, width=width)

    # =============================================================================
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
            "dpi": 200,
            "pad_inches": 0.1
            # 'facecolor': 'black'
        },
    )
    # p.show()
    print("# saved:", save_path)
