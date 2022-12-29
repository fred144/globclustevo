import sys

# sys.path.insert(1, "/homes/fgarcia4/py-virtual-envs/master/lib/python3.7/site-packages")
sys.path.append("..")  # makes sure that importing the modules work

from modules.luminosity.lum_functions import unpack_pop_ii_data
from modules.macros import code_age_to_myr, common_filter_snapshots
from gc_profiles.snapshot_profiler import run_profiler
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import yt
import os
import glob
from modules.macros import filter_snapshots, ram_fields
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog
from scipy import stats


cell_fields, epf = ram_fields()

simulation_run = "fs035_ms10"
finder_profiler_run = "dm_hop"
processor_number = 0  # 0 means one process unparalleled
start_step = 1365
end_step = 1503
step = 1

print("=============================================================================")
print("RUNING DM HALO FINDER")
print("=============================================================================")
datadir = os.path.expanduser(
    "/scratch/dt2/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/{}"  # lustre data path
).format(simulation_run)

# datadir = os.path.expanduser(
#     "/home/fabg/cosm_test_data/refine"  # lustre data path
# ).format(simulation_run)

local_snapshots = filter_snapshots(datadir, start_step, end_step, step)

# if post processing isn't done alongside catalogue
# fof_run_snapshots = filter_snapshots("../halo_data/fs07_refine/fof_best", 113, 918, 1)
# pop_2_dataset = filter_snapshots(
#     "../particle_data/pop_2_data/fs07_refine", 113, 918, 1
# )
# reduced_h5_list = common_filter_snapshots(fof_run_snapshots, local_snapshots)
# reduced_pop2_list = common_filter_snapshots(pop_2_dataset, reduced_h5_list)
# reduced_local_list = common_filter_snapshots(local_snapshots, reduced_h5_list)

m_vir = []
r_vir = []
tot_m_star = []
t = []
z = []
snap = []
#%%
for i, file_name in enumerate(local_snapshots):
    snapshot_num_string = file_name.split("_")[-1]
    ds_file_name = os.path.join(file_name, "info_{}.txt".format(snapshot_num_string))
    print("> Reading", file_name)
    ds = yt.load(ds_file_name, fields=cell_fields, extra_particle_fields=epf)
    ad = ds.all_data()

    hc = HaloCatalog(
        data_ds=ds,
        finder_method="hop",
        finder_kwargs={
            "ptype": "DM",
            # "dm_only": True,
            # "threshold": 160,
        },
        output_dir="./{}/{}/".format(simulation_run, finder_profiler_run),
    )

    hc.create()

    # # read DM info
    x_pos = np.array(ad["DM", "particle_position_x"])
    y_pos = np.array(ad["DM", "particle_position_y"])
    z_pos = np.array(ad["DM", "particle_position_z"])
    dm_id = np.array(ad["DM", "particle_identity"])

    # # center based on star position distribution
    # x_center = np.mean(x_pos)
    # y_center = np.mean(y_pos)
    # z_center = np.mean(z_pos)
    # plt_ctr = np.array([x_center, y_center, z_center])

    # some post processing

    # read in the newly created hdf5 catalogue
    #%%
    hop_catalogue = "./{}/{}/info_{}/info_{}.{}.h5".format(
        simulation_run,
        finder_profiler_run,
        snapshot_num_string,
        snapshot_num_string,
        processor_number,
    )

    with h5.File(hop_catalogue, "r") as f:
        star_ids = np.array(f["particles/ids"])

    # need to read in using yt for virial radius for some reason unknown units in catalogue
    cata_yt = yt.load(hop_catalogue)

    # make a halo catalogue for yt overplot
    # halo_cat_plotting = HaloCatalog(halos_ds=cata_yt)
    # halo_cat_plotting.load()

    cata_yt = cata_yt.all_data()
    dm_halo_m = np.max(np.array(ds.arr(cata_yt["all", "particle_mass"]).to("Msun")))
    haloidx = np.argmax(np.array(ds.arr(cata_yt["all", "particle_mass"]).to("Msun")))
    vir_rad = np.array(ds.arr(cata_yt["all", "virial_radius"]).to("pc"))[haloidx]

    # useful info
    current_time = float(ds.current_time.in_units("Myr"))
    redshft = ds.current_redshift
    t.append(current_time)
    z.append(redshft)
    m_vir.append(dm_halo_m)
    r_vir.append(vir_rad)
    tot_m_star.append(np.sum(np.array(ad["star", "particle_mass"].to("Msun"))))
    snap.append(int(snapshot_num_string))
    # center of the halo
    halo_x = np.array(ds.arr(cata_yt["all", "particle_position_x"]).to("pc"))[haloidx]
    halo_y = np.array(ds.arr(cata_yt["all", "particle_position_y"]).to("pc"))[haloidx]
    halo_z = np.array(ds.arr(cata_yt["all", "particle_position_z"]).to("pc"))[haloidx]

    halo_end = np.array(cata_yt["all", "particle_index_start"])[1]

    halo_p_ids = star_ids[0 : int(halo_end) - 1]
    halo_id_mask = np.isin(dm_id, halo_p_ids)
    x_pos = np.array(ad["DM", "particle_position_x"].to("pc"))[halo_id_mask] - halo_x
    y_pos = np.array(ad["DM", "particle_position_y"].to("pc"))[halo_id_mask] - halo_y
    z_pos = np.array(ad["DM", "particle_position_z"].to("pc"))[halo_id_mask] - halo_z
    halo_dm_mass = np.array(ad["DM", "particle_mass"].to("Msun"))[halo_id_mask]

    dm_part_data = np.vstack((halo_dm_mass, x_pos, y_pos, z_pos)).T
    header = "particle_mass[msun]\tparticle_x[pc]\tparticle_y[pc]\tparticle_z[pc]"
    np.savetxt(
        "./{}/{}/info_{}/dm_data.txt".format(
            simulation_run,
            finder_profiler_run,
            snapshot_num_string,
        ),
        X=dm_part_data,
        header=header,
    )

    dm_total_data = np.vstack((snap, t, z, m_vir, r_vir, tot_m_star)).T
    np.savetxt(
        "./{}/{}/dm_halo_evo.txt".format(
            simulation_run,
            finder_profiler_run,
        ),
        X=dm_total_data,
        header="snap_num\tt[myr]\tredshift\tmvir[msun]\trvir[pc]\tm_star[msun]",
    )
    print(dm_total_data)
