import sys

sys.path.append("../")
import numpy as np
import os
import glob
from modules.match_t_sims import find_matching_time, get_snapshots
from modules.macros import filter_snapshots
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib import colors
import misc_visuals
import yt
from modules.macros import filter_snapshots, ram_fields


yt.enable_parallelism()

f7_snap_range = (126, 1296)
f3_snap_range = (179, 1469)

fs070_pop2_f = filter_snapshots(
    "../particle_data/pop_2_data/fs07_refine", f7_snap_range[0], f7_snap_range[1], 1
)
fs035_pop2_f = filter_snapshots(
    "../particle_data/pop_2_data/fs035_ms10", f3_snap_range[0], f3_snap_range[1], 1
)

fs035_pop2_f, f3_matched_with_f7_nums, f7_nums = find_matching_time(
    sequence=fs070_pop2_f, look_up_sequence=fs035_pop2_f, orig_seq_out_num=True
)

# apply uniqueness cuts.
unique_idxs = np.unique(fs035_pop2_f, return_index=True)[1]
fs070_pop2_f = np.array(fs070_pop2_f)[unique_idxs].tolist()
fs035_pop2_f = np.array(fs035_pop2_f)[unique_idxs].tolist()
f3_nums = f3_matched_with_f7_nums[unique_idxs]
f7_nums = f7_nums[unique_idxs]

fs070_halo_f = get_snapshots(
    snapshot_file_list=filter_snapshots(
        r"../halo_data/fs07_refine/fof_best", f7_snap_range[0], f7_snap_range[1], 1
    ),
    get_list=f7_nums,
    verbose=False,
)
fs035_halo_f = get_snapshots(
    snapshot_file_list=filter_snapshots(
        r"../halo_data/fs035_ms10/fof_best", f3_snap_range[0], f3_snap_range[1], 1
    ),
    get_list=f3_nums,
    verbose=False,
)

fs070_snap_dir = "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine"
fs035_snap_dir = "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs035_ms10"

f7_snap_f = filter_snapshots(fs070_snap_dir, get_list=f7_nums)
f3_snap_f = filter_snapshots(fs035_snap_dir, get_list=f3_nums)
