import sys

sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/cosmology-clean/lib/python3.7/site-packages"
)
# sys.path.insert(
#     1, "/homes/fgarcia4/py-virtual-envs/old-yt-361/lib/python3.7/site-packages"
# )

import warnings
import os

# import pathlib
import yt

from yt.funcs import mylog
import numpy as np

from yt.extensions.astro_analysis.halo_analysis import HaloCatalog

# from yt.analysis_modules.halo_analysis.api import HaloCatalog


# yt.enable_parallelism()
mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


# data_directory = r"../../cosm_test_data/refine"

# ---------------------------------DT2 Paths------------------------------------
# lustre data path
data_directory = os.path.expanduser(
    "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine"
)
# # save path
# sequence_folder = "newcmap_gas_projected_density_z"
# parent_folder = "/homes/fgarcia4/analysis/cluster_evolution/sequences/new_refine"
# newpath = parent_folder + "/" + sequence_folder
# if not os.path.exists(newpath):
#     print("# Creating new sequence directory", newpath)
#     os.makedirs(newpath)
# =============================================================================

# enable discrete selection of time range based on snapshot number
strt_snapshot = "00113"
end_snapshot = "00810"
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
]

# yt_data_series = yt.DatasetSeries(
#     filtered_files, parallel=True, fields=cell_fields, extra_particle_fields=epf
# )


# for ds in yt_data_series:
#     print(ds.current_time)

for file_name in filtered_files:
    file_name = file_name + "/info_{}.txt".format(file_name[-5:])
    print("Reading", file_name)
    ds = yt.load(file_name, fields=cell_fields, extra_particle_fields=epf)
    ad = ds.all_data()

    # read POPII star info

    x_pos = np.array(ad["star", "particle_position_x"])
    y_pos = np.array(ad["star", "particle_position_y"])
    z_pos = np.array(ad["star", "particle_position_z"])

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
            "link": 0.00000148,
            "dm_only": False,
        },
        output_dir="../halo_data/",
    )

    hc.create()

    # if runnig old halo finder

    # hc_ad = hc.halos_ds.all_data()

    # new yt
    # halos = yt.load("../halo_data/info_00500/info_00500.0.h5")
    # cata = HaloCatalog(halos_ds=halos)
    # cata.load()

    # # optionally plot it
    # width = (400, "pc")
    # p = yt.ProjectionPlot(ds, "z", "density", width=width, center=plt_ctr)
    # p.annotate_particles(width=width, ptype="star", p_size=0.2, marker=".", col="r")
    # p.annotate_halos(
    #     cata,
    #     width=width,
    # )
    # p.set_cmap("density", "copper")
    # # p.set_figure_size(5)
    # p.save(
    #     "./",
    #     mpl_kwargs={"bbox_inches": "tight", "dpi": 500, "pad_inches": 0.1},
    # )
