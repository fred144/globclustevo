import sys

sys.path.append("../")
import numpy as np
import os
from modules.macros import filter_snapshots, ram_fields
import yt

yt.enable_parallelism()
simulation_name = "fs035_ms10"
f3_sn_dir = os.path.abspath(
    "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/{}".format(simulation_name)
)
ft_p2_dir = "../particle_data/pop_2_data/{}".format(simulation_name)

slice_axis = "z"
width = (400, "pc")
res = 1000

f3_strt = 154
f3_end = 1177
step = 1

f3_pop2 = filter_snapshots(ft_p2_dir, f3_strt, f3_end, step)
f3_sn = filter_snapshots(f3_sn_dir, f3_strt, f3_end)

parent_folder = "../gas_data/{}".format(simulation_name)
dens_sequence_folder = os.path.abspath(os.path.join(parent_folder, "gas_density"))
temp_sequence_folder = os.path.abspath(os.path.join(parent_folder, "temperature"))


if not os.path.exists(dens_sequence_folder):
    print("# Creating new sequence directory", dens_sequence_folder)
    os.makedirs(dens_sequence_folder)

if not os.path.exists(temp_sequence_folder):
    print("# Creating new sequence directory", temp_sequence_folder)
    os.makedirs(temp_sequence_folder)

skip_frames = [1102]  # corrupted frames

for i, (f3_sn, f3_p2) in enumerate(zip(f3_sn, f3_pop2)):
    print("> Reading {}".format(f3_sn))
    output_num = f3_sn.split("_")[-1]
    if output_num in skip_frames:
        continue
    cell_fields, epf = ram_fields()
    f3_info_file = os.path.join(f3_sn, "info_{}.txt".format(output_num))

    f3_ds = yt.load(f3_info_file, fields=cell_fields, extra_particle_fields=epf)

    print("Loaded Snapshot Data", output_num)
    # get pre processed data from pop2 data sets

    f3_code_ctr = np.loadtxt(f3_p2, max_rows=5)[2:5, 6]

    # start getting gas data
    f3_p_gas = yt.ProjectionPlot(
        f3_ds, slice_axis, ("gas", "density"), width=width, center=f3_code_ctr
    )
    f3_p_temp = yt.ProjectionPlot(
        f3_ds, slice_axis, ("gas", "temperature"), width=width, center=f3_code_ctr
    )

    f3_gas_frb = f3_p_gas.data_source.to_frb(width, res)
    f3_temp_frb = f3_p_temp.data_source.to_frb(width, res)

    f3_gas = np.array(f3_gas_frb["gas", "density"])
    f3_temp = np.array(f3_temp_frb["gas", "temperature"])

    gas_save = os.path.join(dens_sequence_folder, "dens_{}.txt".format(output_num))
    np.savetxt(gas_save, X=f3_gas)
    print("> saved: {}".format(gas_save))

    temp_save = os.path.join(temp_sequence_folder, "temp_{}.txt".format(output_num))
    np.savetxt(temp_save, X=f3_temp)
    print("> saved: {}".format(temp_save))
