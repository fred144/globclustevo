import sys

sys.path.append("..")
import numpy as np
import os
import glob
from modules.macros import filter_snapshots

halo_data_directory = r"../halo_data/fs07_refine/fof_best"
halo_ds = filter_snapshots(halo_data_directory, 404, 918, 1)
# TODO: make a an independent profiler and lum map three panel. etc.
for ds in halo_ds:
    output_num = int(ds.split("/")[-1].split("_")[-1])
    halos = sorted(glob.glob(os.path.join(ds, "gc*.txt")))
    for halo in halos:

        halo_data = np.loadtxt(halo)

        if 33387 in halo_data[:, 0]:  # the star id number was found via trial and error
            halo_num = int(halo.split("/")[-1].split("_")[-1].split(".")[0])
            print(output_num, halo_num)
        else:
            pass
