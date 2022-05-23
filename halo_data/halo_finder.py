import sys

sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/clean-install/lib/python3.7/site-packages"
)
# sys.path.insert(
#     1, "/homes/fgarcia4/py-virtual-envs/old-yt-361/lib/python3.7/site-packages"
# )

import warnings
import os

# import pathlib
import yt
import h5py as h5

# yt.enable_parallelism()
from yt.funcs import mylog
import numpy as np

from yt.extensions.astro_analysis.halo_analysis import HaloCatalog

# from yt.analysis_modules.halo_analysis.api import HaloCatalog


mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


data_directory = r"../../cosm_test_data/refine"

# =============================================================================
# lustre data path
# data_directory = os.path.expanduser(
#     "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine"
# )
# # save path
# sequence_folder = "newcmap_gas_projected_density_z"
# parent_folder = "/homes/fgarcia4/analysis/cluster_evolution/sequences/new_refine"
# newpath = parent_folder + "/" + sequence_folder
# if not os.path.exists(newpath):
#     print("# Creating new sequence directory", newpath)
#     os.makedirs(newpath)
# =============================================================================

# enable discrete selection of time range based on snapshot number
strt_snapshot = "00250"
end_snapshot = "00250"
files = sorted(os.listdir(data_directory))  # [-2:-1]  [300:400:2]
strt_idx = [i for i, s in enumerate(files) if strt_snapshot in s][0]
end_idx = [i for i, s in enumerate(files) if end_snapshot in s][0]
filtered_files = files[strt_idx : end_idx + 1 : 1]
filtered_files = [os.path.join(data_directory, file) for file in filtered_files]

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

# yt_data_series = yt.DatasetSeries(
#     filtered_files, parallel=True, fields=cell_fields, extra_particle_fields=epf
# )


# for ds in yt_data_series:
#     print(ds.current_time)

for file_name in filtered_files:
    output_num_string = file_name[-5:]
    file_name = file_name + "/info_{}.txt".format(output_num_string)

    print("Reading", file_name)

    ds = yt.load(file_name, fields=cell_fields, extra_particle_fields=epf)
    ad = ds.all_data()

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

    hc = HaloCatalog(
        data_ds=ds,
        finder_method="fof",
        finder_kwargs={
            "ptype": "star",
            "padding": 0.0001,
            "link": 0.00001,
            "dm_only": False,
        },
        output_dir="../halo_data/fs07_refine/test_run",
    )

    hc.create()

    # load halo data
    cata_h5 = h5.File(
        "../halo_data/fs07_refine/test_run/info_00250/info_00250.0.h5", "r"
    )

    # need to read in using yt for virial radius for some reason unknown units in catalogue
    cata_yt = yt.load("../halo_data/fs07_refine/test_run/info_00250/info_00250.0.h5")

    # make a halo catalogue for yt overplot
    halo_cat_plotting = HaloCatalog(halos_ds=cata_yt)
    halo_cat_plotting.load()

    cata_yt = cata_yt.all_data()

    pop2_data = np.loadtxt("../pop_2_data/fs07_refine/pos_00250_426_61_myr.txt")
    ctr_at = pop2_data[5:8, 6]

    # get the halo centers

    halo_id = np.array(cata_h5["particle_identifier"])

    halo_x = np.array(cata_h5["particle_position_x"])
    halo_y = np.array(cata_h5["particle_position_y"])
    halo_z = np.array(cata_h5["particle_position_z"])

    halo_x = np.array(ds.arr(halo_x, "code_length").to("pc")) - ctr_at[0]
    halo_y = np.array(ds.arr(halo_y, "code_length").to("pc")) - ctr_at[1]
    halo_z = np.array(ds.arr(halo_z, "code_length").to("pc")) - ctr_at[2]

    # get halo virial radii
    halo_vir_rad = np.array(ds.arr(cata_yt["all", "virial_radius"], "cm").to("pc"))

    cat_pc = np.vstack((halo_id, halo_x, halo_y, halo_z, halo_vir_rad)).T
    header = "halo_id \t x_coord [pc] \t y_coord [pc] \t z_coord [pc] \t vir_rad [pc]"
    # TODO: remove hardcode
    hard = "../halo_data/fs07_refine/test_run/"
    save_name = hard + "info_{}/catalogue_{}.txt".format(
        output_num_string, output_num_string
    )
    np.savetxt(save_name, X=cat_pc, header=header)

    # get particles belonging to each halo
    num_stars_in_halo = np.array(cata_h5["particle_number"])
    start_of_new_halo = np.array(cata_h5["particle_index_start"])
    halo_star_ids = np.array(cata_h5["particles/ids"])

    cata_h5.close()
    for i, (new_h, h_id) in enumerate(zip(start_of_new_halo, halo_id), start=1):

        if i == np.size(
            start_of_new_halo
        ):  # cheecky over ride once it reaches end of list
            star_ids_inside = halo_star_ids[new_h:]
        else:
            star_ids_inside = halo_star_ids[new_h : start_of_new_halo[i]]

        # translate stars, taking into account centers
        gc_mask = np.isin(star_id, star_ids_inside)
        gc_x = np.array(ds.arr(x_pos - x_center, "code_length").to("pc"))[gc_mask]
        gc_y = np.array(ds.arr(y_pos - y_center, "code_length").to("pc"))[gc_mask]
        gc_z = np.array(ds.arr(z_pos - z_center, "code_length").to("pc"))[gc_mask]
        gc_stars = np.vstack((gc_x, gc_y, gc_z)).T
        header = "star_x_coords [pc] \t star_y_coords [pc] \t star_z_coords [pc] "
        # TODO: remove hardcode
        hard = "../halo_data/fs07_refine/test_run/"
        save_name = hard + "info_{}/gc_vir_{}".format(
            output_num_string, str(int(h_id)).zfill(3)
        )
        np.savetxt(save_name + ".txt", X=gc_stars, header=header)

    # optionally plot the halod finder results and put in the catalogue directory
    width = (400, "pc")
    p = yt.ProjectionPlot(ds, "z", "density", width=width, center=plt_ctr)
    p.annotate_particles(
        width=width, ptype="star", alpha=0.1, p_size=0.2, marker=".", col="red"
    )
    p.annotate_halos(
        halo_cat_plotting,
        width=width,
    )
    p.set_cmap("density", "copper")
    p["gas", "density"].axes.scatter(
        halo_x,
        halo_y,
        color="green",
        alpha=1,
        marker="x",
        linewidths=0.1,
        s=2,
    )

    # p.set_figure_size(5)
    p.save(
        hard + "info_{}/annotated.png".format(output_num_string),
        mpl_kwargs={"bbox_inches": "tight", "dpi": 300, "pad_inches": 0.1},
    )
