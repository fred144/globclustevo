import sys

sys.path.append("../")
import numpy as np
import os
from modules.macros import filter_snapshots, ram_fields
import yt

yt.enable_parallelism()
simulation_name = "fs07_refine"
f7_sn_dir = os.path.abspath(
    "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine"
)
ft_p2_dir = "../particle_data/pop_2_data/fs07_refine"

slice_axis = "z"
width = (400, "pc")
res = 1000

f7_strt = 1110
f7_end = 1110
step = 25

f7_pop2 = filter_snapshots(ft_p2_dir, f7_strt, f7_end, step)
f7_snap = filter_snapshots(f7_sn_dir, f7_strt, f7_end, step)
print(f7_pop2)

parent_folder = "../gas_data/{}".format(simulation_name)
dens_sequence_folder = os.path.abspath(os.path.join(parent_folder, "gas_density"))
temp_sequence_folder = os.path.abspath(
    os.path.join(parent_folder, "weighted_temperature")
)


if not os.path.exists(dens_sequence_folder):
    print("# Creating new sequence directory", dens_sequence_folder)
    os.makedirs(dens_sequence_folder)

if not os.path.exists(temp_sequence_folder):
    print("# Creating new sequence directory", temp_sequence_folder)
    os.makedirs(temp_sequence_folder)

for i, (f7_sn, f7_p2) in enumerate(zip(f7_snap, f7_pop2)):
    print("> Reading {}".format(f7_sn))
    output_num = f7_sn.split("_")[-1]

    cell_fields, epf = ram_fields()
    f7_info_file = os.path.join(f7_sn, "info_{}.txt".format(output_num))

    f7_ds = yt.load(f7_info_file, fields=cell_fields, extra_particle_fields=epf)

    print("Loaded Snapshot Data", output_num)
    # get pre processed data from pop2 data sets

    f7_code_ctr = np.loadtxt(f7_p2, max_rows=5)[2:5, 6]

    # start getting gas data
    f7_p_gas = yt.ProjectionPlot(
        f7_ds, slice_axis, ("gas", "density"), width=width, center=f7_code_ctr
    )
    # f7_p_temp = yt.ProjectionPlot(
    #     f7_ds,
    #     slice_axis,
    #     ("gas", "temperature"),
    #     width=width,
    #     center=f7_code_ctr,
    #     weight_field=("gas", "density"),
    # )

    # f7_gas_frb = f7_p_gas.data_source.to_frb(width, res)
    f7_temp_frb = f7_p_temp.data_source.to_frb(width, res)

    # f7_gas = np.array(f7_gas_frb["gas", "density"])
    f7_temp = np.array(f7_temp_frb["gas", "temperature"])

    # gas_save = os.path.join(dens_sequence_folder, "dens_{}.txt".format(output_num))
    # np.savetxt(gas_save, X=f7_gas)
    # print("> saved: {}".format(gas_save))

    temp_save = os.path.join(
        temp_sequence_folder, "weight_temp_{}.txt".format(output_num)
    )
    np.savetxt(temp_save, X=f7_temp)
    print("> saved: {}".format(temp_save))
