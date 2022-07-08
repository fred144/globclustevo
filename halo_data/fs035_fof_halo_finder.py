import sys

sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/clean-install/lib/python3.7/site-packages"
)
sys.path.append("..")  # makes sure that importing the modules in the main folder work

from modules.macros import filter_snapshots, common_filter_snapshots
import glob
import h5py as h5
import warnings
import os
import numpy as np
import yt

yt.enable_parallelism()
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog
from yt.funcs import mylog


mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

simulation_run = "fs035_ms10"
profiler_run = "fof_best"
processor_number = 0

# local_snapshots = filter_snapshots(r"../../cosm_test_data/fs035_ms10", 500, 500, 1)

datadir = os.path.expanduser(
    "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/{}"  # lustre data path
).format(simulation_run)
local_snapshots = filter_snapshots(datadir, 1125, 1177, 1)


# if post processing isn't done alongside catalogue
# fof_run_snapshots = filter_snapshots("../halo_data/fs07_refine/fof_best", 113, 918, 1)
# pop_2_dataset = filter_snapshots(
#     "../particle_data/pop_2_data/fs07_refine", 113, 918, 1
# )
# reduced_h5_list = common_filter_snapshots(fof_run_snapshots, local_snapshots)
# reduced_pop2_list = common_filter_snapshots(pop_2_dataset, reduced_h5_list)
# reduced_local_list = common_filter_snapshots(local_snapshots, reduced_h5_list)


cell_fields = [
    "Density",
    "x-velocity",
    "y-velocity",
    "z-velocity",
    "Pressure",
    "Metallicity",
    "dark_matter_density",
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
    ("particle_birth_time", "d"),
    ("star_age", "d"),
]


for i, file_name in enumerate(local_snapshots):
    snapshot_num_string = file_name.split("_")[-1]
    ds_file_name = os.path.join(file_name, "info_{}.txt".format(snapshot_num_string))
    print("> Reading", file_name)

    ds = yt.load(ds_file_name, fields=cell_fields, extra_particle_fields=epf)
    ad = ds.all_data()
    hc = HaloCatalog(
        data_ds=ds,
        finder_method="fof",
        finder_kwargs={
            "ptype": "star",
            "padding": 0.0001,
            "link": 0.00001,  # "best"
            # "link": 0.0000025, # "fof"
            "dm_only": False,
        },
        output_dir="../halo_data/{}/{}/".format(simulation_run, profiler_run),
    )

    hc.create()

    # read POPII star info
    x_pos = np.array(ad["star", "particle_position_x"])
    y_pos = np.array(ad["star", "particle_position_y"])
    z_pos = np.array(ad["star", "particle_position_z"])
    star_id = np.array(ad["star", "particle_identity"])

    # center based on star position distribution
    x_center = np.mean(x_pos)
    y_center = np.mean(y_pos)
    z_center = np.mean(z_pos)
    plt_ctr = np.array([x_center, y_center, z_center])

    # =============================================================================
    # post processing
    # =============================================================================

    # read in the newly created hdf5 catalogue

    fof_catalogue = "../halo_data/{}/{}/info_{}/info_{}.{}.h5".format(
        simulation_run,
        profiler_run,
        snapshot_num_string,
        snapshot_num_string,
        processor_number,
    )
    cata_h5 = h5.File(fof_catalogue, "r")

    # need to read in using yt for virial radius for some reason unknown units in catalogue
    cata_yt = yt.load(fof_catalogue)

    # make a halo catalogue for yt overplot
    halo_cat_plotting = HaloCatalog(halos_ds=cata_yt)
    halo_cat_plotting.load()

    cata_yt = cata_yt.all_data()

    pop2_data = np.loadtxt(
        glob.glob(
            "../particle_data/pop_2_data/{}/pos_{}_*.txt".format(
                simulation_run, snapshot_num_string
            )
        )[0]
    )
    ctr_at_pc = pop2_data[5:8, 6]
    ctr_at_code_length = pop2_data[2:5, 6]
    star_ids = pop2_data[:, 0]
    x_pos = pop2_data[:, 2]
    y_pos = pop2_data[:, 3]
    z_pos = pop2_data[:, 4]

    # get the halo centers

    halo_id = np.array(cata_h5["particle_identifier"])

    halo_x = np.array(cata_h5["particle_position_x"])
    halo_y = np.array(cata_h5["particle_position_y"])
    halo_z = np.array(cata_h5["particle_position_z"])

    halo_x = np.array(ds.arr(halo_x, "code_length").to("pc")) - ctr_at_pc[0]
    halo_y = np.array(ds.arr(halo_y, "code_length").to("pc")) - ctr_at_pc[1]
    halo_z = np.array(ds.arr(halo_z, "code_length").to("pc")) - ctr_at_pc[2]

    # get halo virial radii
    halo_vir_rad = np.array(ds.arr(cata_yt["all", "virial_radius"], "cm").to("pc"))

    cat_pc = np.vstack((halo_id, halo_x, halo_y, halo_z, halo_vir_rad)).T
    cat_save_name = "./{}/{}/info_{}/catalogue_{}.txt".format(
        simulation_run, profiler_run, snapshot_num_string, snapshot_num_string
    )
    header = "halo_id \t x_coord [pc] \t y_coord [pc] \t z_coord [pc] \t vir_rad [pc]"
    np.savetxt(cat_save_name, X=cat_pc, header=header)

    # get particles belonging to each halo
    num_stars_in_halo = np.array(cata_h5["particle_number"])
    start_of_new_halo = np.array(cata_h5["particle_index_start"])
    halo_star_ids = np.array(cata_h5["particles/ids"])

    cata_h5.close()
    for i, (new_h, h_id) in enumerate(zip(start_of_new_halo, halo_id), start=1):

        if i == np.size(
            start_of_new_halo
        ):  # cheeky over ride once it reaches end of list
            star_ids_inside = halo_star_ids[new_h:]
        else:
            star_ids_inside = halo_star_ids[new_h : start_of_new_halo[i]]

        # translate stars, taking into account centers
        gc_mask = np.isin(star_ids, star_ids_inside)
        gc_x = x_pos[gc_mask]
        gc_y = y_pos[gc_mask]
        gc_z = z_pos[gc_mask]

        # take the x,y,z of individual clusters and subtract center of halo cluster
        # this makes them all centered at the origin (0,0,0)
        gc_stars = np.vstack((gc_x, gc_y, gc_z)).T - cat_pc[:, 1:-1][i - 1]
        gc_stars = np.column_stack((star_ids_inside, gc_stars))
        header = (
            "star id \t star_x_coords [pc] \t star_y_coords [pc] \t star_z_coords [pc] "
        )

        save_name = "../halo_data/{}/{}/info_{}/gc_vir_{}.txt".format(
            simulation_run, profiler_run, snapshot_num_string, str(int(h_id)).zfill(3)
        )
        np.savetxt(save_name, X=gc_stars, header=header)

    # optionally plot the halod finder results and put in the catalogue directory
    width = (400, "pc")
    # p = yt.ProjectionPlot(ds, "z", "density", width=width, center=ctr_at_code_length)
    # p.annotate_particles(
    #     width=width, ptype="star", alpha=0.5, p_size=0.2, marker=".", col="red"
    # )
    # p.annotate_halos(
    #     halo_cat_plotting,
    #     width=width,
    # )
    # p.set_cmap("density", "copper")
    # p["gas", "density"].axes.scatter(
    #     halo_x,
    #     halo_y,
    #     color="green",
    #     alpha=1,
    #     marker="x",
    #     linewidths=0.1,
    #     s=2,
    # )

    # # p.set_figure_size(5)
    # p.save(
    #     "../halo_data/{}/{}/info_{}/annotated.png".format(
    #         simulation_run, profiler_run, snapshot_num_string
    #     ),
    #     mpl_kwargs={"bbox_inches": "tight", "dpi": 300, "pad_inches": 0.1},
    # )
