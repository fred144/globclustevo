import sys

sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/cosmology-clean/lib/python3.7/site-packages"
)
import warnings
import os

# import pathlib
import yt

from yt.funcs import mylog
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog

yt.enable_parallelism()
mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# ---------------------------------local test-----------------------------------

data_directory = r"../../cosm_test_data/refine"
parent_folder = "."
sequence_folder = "test_frames"

# ---------------------------------DT2 Paths------------------------------------
# lustre data path
# datadir = os.path.expanduser(
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
strt_snapshot = "00500"
end_snapshot = "00500"
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

yt_data_series = yt.DatasetSeries(
    filtered_files, parallel=True, fields=cell_fields, extra_particle_fields=epf
)


for ds in yt_data_series:
    print(ds.current_time)
    p = yt.ProjectionPlot(ds, "z", ("gas", "density"), width=(400, "pc"))
    p.annotate_particles(width=(400, "pc"), p_size=0.2, ptype="star")
    p.annotate_particles(width=(400, "pc"), ptype="SFC", p_size=10, marker="x", col="b")
    p.show()


# if pos_sfcs.size > 0:
#     p.annotate_particles(width=width, ptype="SFC", p_size=10, marker="x", col="b")
# if pos_pscs.size > 0:
#     p.annotate_particles(width=width, ptype="PSC", p_size=10, marker="x", col="r")

# from yt.analysis_modules.halo_analysis.api import HaloCatalog

# hc = HaloCatalog(data_ds=ds, finder_method='hop',
#                   finder_kwargs={"threshold": 100, #default: 160
#                                   "ptype":'DM',
#                                   "dm_only":False})

# hc = HaloCatalog(data_ds=ds, finder_method='fof',
#                   finder_kwargs={"ptype": 'DM',
#                                 "link": 0.2,
#                                 "dm_only":False})

# hc.create()
# hc_ad = hc.halos_ds.all_data()
# p.annotate_halos(hc,
#                   width=width,
#                   factor = 0.03)
