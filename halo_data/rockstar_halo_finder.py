import sys

sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/clean-install/lib/python3.7/site-packages"
)

import h5py as h5
import warnings
import os
import numpy as np
import yt

yt.enable_parallelism()
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog
from yt.funcs import mylog

# from yt.analysis_modules.halo_analysis.api import HaloCatalog


# mylog.setLevel(40)
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

    hc = HaloCatalog(
        data_ds=ds,
        finder_method="rockstar",
        finder_kwargs={"particle_type": "star", "num_readers": 1, "num_writers": 1},
        output_dir="../halo_data/fs07_refine/rockstar",
    )

    # hc.create()
