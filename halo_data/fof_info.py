"""
make an info file similar to the one produced by gc_profiler but based soley on the
fof results. to be used for things like the mass function or bubble plot.
"""

import sys

sys.path.append("..")
import numpy as np
import os
import glob
from modules.macros import filter_snapshots
from modules.luminosity.lum_functions import unpack_pop_ii_data

from scipy import stats

strt = 694
end = 918
step = 1

halo_data_directory = r"./fs07_refine/fof_best"
pop2_data_directory = r"../particle_data/pop_2_data/fs07_refine"
pop2 = filter_snapshots(pop2_data_directory, strt, end, step)
halo_ds = filter_snapshots(halo_data_directory, strt, end, step)

# TODO: make a an independent profiler and lum map three panel. etc.
for p2, ds in zip(pop2, halo_ds):
    output_num = int(ds.split("/")[-1].split("_")[-1])

    _, scaled_stellar_lums, masses, ages, tz, ids = unpack_pop_ii_data(
        p2, return_ids=True, return_z=True
    )
    t_myr = tz[0]
    redshift = tz[1]

    halos = sorted(glob.glob(os.path.join(ds, "gc_vir_*.txt")))

    gc_ages_per_snapshot = []
    gc_masses_per_snapshot = []
    gc_lums_per_snapshot = []

    for halo in halos:

        halo_data = np.loadtxt(halo)

        halo_num = int(halo.split("/")[-1].split("_")[-1].split(".")[0])
        gc_mask = np.isin(ids, halo_data[:, 0])
        gc_lums = scaled_stellar_lums[gc_mask]
        gc_mass = masses[gc_mask]
        gc_ages = ages[gc_mask] / 1e6

        gc_masses_per_snapshot.append(np.sum(gc_mass))
        gc_lums_per_snapshot.append(np.sum(gc_lums))
        gc_ages_per_snapshot.append(float(stats.mode(gc_ages)[0]))

    gc_ages_per_snapshot = np.array(gc_ages_per_snapshot)
    gc_masses_per_snapshot = np.array(gc_masses_per_snapshot)
    gc_lums_per_snapshot = np.array(gc_lums_per_snapshot)

    t_myr = np.array([t_myr])
    redshift = np.array([redshift])
    t_myr.resize(np.size(gc_lums_per_snapshot))
    redshift.resize(np.size(gc_lums_per_snapshot))
    output = np.vstack(
        (
            t_myr,
            redshift,
            gc_ages_per_snapshot,
            gc_masses_per_snapshot,
            gc_lums_per_snapshot,
        )
    ).T

    header = (
        "t_myr, z, gc_ages_per_snapshot, gc_masses_per_snapshot, gc_lums_per_snapshot"
    )
    info_save_path = os.path.join(ds, "fof_info.txt")
    np.savetxt(fname=info_save_path, X=output, header=header)
    print("> saved", info_save_path)
