import sys

sys.path.append("..")
sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/clean-install/lib/python3.7/site-packages"
)
import glob
import os
from modules.macros import filter_snapshots, common_filter_snapshots
from snapshot_profiler import run_profiler
import matplotlib.pyplot as plt
import numpy as np
import yt
from yt.funcs import mylog
import warnings


yt.enable_parallelism()
mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


plt.rcParams.update({"figure.max_open_warning": 0})
# mpl.rc('font', family='serif')
# mpl.rc('text', usetex=True)
# plt.style.use("dark_background")

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })


def fof_profiler(pop2_data_set, halo_data_directory, run_save_path, gc_radii):
    # declare lists that will be filled as the profiler is ran,
    # one data point per snpashot-- time series over an entire sim.
    gc_counts = []
    gc_masses = []
    gc_core_masses = []
    total_counts = []
    total_masses = []
    time_myr = []
    redshift = []

    # loop over snapshots
    for pop_data, hc_data in zip(pop2_data_set, halo_data_directory):
        snapshot_num = pop_data.split("/")[-1].split("_")[0]
        pop2_data = np.loadtxt(pop_data)

        # load the processed catalogue, recentered and in pc
        catalogue_text_file = os.path.join(hc_data, sorted(os.listdir(hc_data))[1])
        halo_data = np.loadtxt(catalogue_text_file)
        # centers with label in the first columns
        try:
            ctrs_pc = halo_data[:, :-1]
            if gc_radii == "vir_rad":
                profile_gcs_up_to = halo_data[:, -1]
            elif isinstance(gc_radii, int):
                profile_gcs_up_to = gc_radii
            else:
                print("> gc radius not supported")
                break
        except:
            # triggered if there is only one in the catalogue
            ctrs_pc = halo_data[:-1]
            if gc_radii == "vir_rad":
                profile_gcs_up_to = halo_data[-1]
            elif isinstance(gc_radii, int):
                profile_gcs_up_to = gc_radii
            else:
                print("> gc radius not supported")
                break

        # put all verbose output into a text file
        # orig_stdout = sys.stdout
        # sys.stdout = open(os.path.join(run_save_path, "log.txt"), "w")

        (
            gc_full_masses,
            core_radii,
            core_masses,
            r_trunc,
            ages,
            time,
            particles_in_this_snapshot,
            pop_2_ids_in_gcs,
        ) = run_profiler(
            star_file_path=pop_data,
            parent_save_path=save_path,
            proj_width=400,  # pc
            gc_radii=profile_gcs_up_to,  # radii to be used to extract clusters
            lum_map_bins=4000,  # bad resolution so that you can see it better
            centers=ctrs_pc,
            particle_filter=hc_data,
        )
        mask = np.isin(pop2_data[:, 0], pop_2_ids_in_gcs)
        pop2_data_field = pop2_data[~mask]

        gc_counts.append(np.sum(particles_in_this_snapshot))
        gc_masses.append(np.sum(gc_full_masses))  # masses within the vir_radius, full
        gc_core_masses.append(np.sum(gc_core_masses))  # Msun

        total_masses.append(np.sum(pop2_data[:, 5]))  # Msun
        total_counts.append(pop2_data.shape[0])

        time_myr.append(time)
        redshift.append(pop2_data[2, 6])

        header = (
            "\t\tID"
            "\t\tCurrentAges[MYr]"
            "\t\tX[pc]"
            "\t\tY[pc]\t\t"
            "Z[pc]\t\t"
            "mass[Msun]"
            "\t\tt_sim[Myr], z, ctr(code), ctr(pc)"
        )
        field_save_path = glob.glob(
            os.path.join(save_path, "snapshot_{}*".format(snapshot_num))
        )
        print(field_save_path)
        np.savetxt(
            fname=os.path.join(field_save_path, "field_stars.txt"),
            X=pop2_data_field,
            header=header,
        )
        # print(os.path.join(hc_data, "field_stars.txt"))
        # run_save_path


if __name__ == "__main__":

    pop2_data_directory = r"../particle_data/pop_2_data/fs07_refine"
    halo_data_directory = r"../halo_data/fs07_refine/fof_best"

    save_path = "./profile_runs/fs07_refine/test_run/fof_best"

    strt = 113
    end = 113
    step = 1
    pop2 = filter_snapshots(pop2_data_directory, strt, end, step)
    halos = filter_snapshots(halo_data_directory, strt, end, step)

    # pop2 = common_filter_snapshots(pop2, halos)
    # fof_profiler(
    #     pop2_data_set=pop2,
    #     halo_data_directory=halos,
    #     run_save_path=save_path,
    #     gc_radii="vir_rad",
    # )
    # declare lists that will be filled as the profiler is ran,
    # one data point per snpashot-- time series over an entire sim.
    gc_counts = []
    gc_masses = []
    gc_core_masses = []
    total_counts = []
    total_masses = []
    time_myr = []
    redshift = []
    gc_radii = "vir_rad"
    # loop over snapshots
    for pop_data, hc_data in zip(pop2, halos):
        snapshot_num = pop_data.split("/")[-1].split("_")[0]
        pop2_data = np.loadtxt(pop_data)

        # load the processed catalogue, recentered and in pc
        catalogue_text_file = os.path.join(hc_data, sorted(os.listdir(hc_data))[1])
        halo_data = np.loadtxt(catalogue_text_file)
        # centers with label in the first columns
        try:
            ctrs_pc = halo_data[:, :-1]
            if gc_radii == "vir_rad":
                profile_gcs_up_to = halo_data[:, -1]
            elif isinstance(gc_radii, int):
                profile_gcs_up_to = gc_radii
            else:
                print("> gc radius not supported")
                break
        except:
            # triggered if there is only one in the catalogue
            ctrs_pc = halo_data[:-1]
            if gc_radii == "vir_rad":
                profile_gcs_up_to = halo_data[-1]
            elif isinstance(gc_radii, int):
                profile_gcs_up_to = gc_radii
            else:
                print("> gc radius not supported")
                break

        # put all verbose output into a text file
        # orig_stdout = sys.stdout
        # sys.stdout = open(os.path.join(run_save_path, "log.txt"), "w")

        (
            gc_full_masses,
            core_radii,
            core_masses,
            r_trunc,
            ages,
            time,
            particles_in_this_snapshot,
            pop_2_ids_in_gcs,
        ) = run_profiler(
            star_file_path=pop_data,
            parent_save_path=save_path,
            proj_width=400,  # pc
            gc_radii=profile_gcs_up_to,  # radii to be used to extract clusters
            lum_map_bins=4000,  # bad resolution so that you can see it better
            centers=ctrs_pc,
            particle_filter=hc_data,
        )
        mask = np.isin(pop2_data[:, 0], pop_2_ids_in_gcs)
        pop2_data_field = pop2_data[~mask]

        gc_counts.append(np.sum(particles_in_this_snapshot))
        gc_masses.append(np.sum(gc_full_masses))  # masses within the vir_radius, full
        gc_core_masses.append(np.sum(gc_core_masses))  # Msun

        total_masses.append(np.sum(pop2_data[:, 5]))  # Msun
        total_counts.append(pop2_data.shape[0])

        time_myr.append(time)
        redshift.append(pop2_data[2, 6])

        header = (
            "\t\tID"
            "\t\tCurrentAges[MYr]"
            "\t\tX[pc]"
            "\t\tY[pc]\t\t"
            "Z[pc]\t\t"
            "mass[Msun]"
            "\t\tt_sim[Myr], z, ctr(code), ctr(pc)"
        )  # TODO figure this out
        field_save_path = glob.glob(
            os.path.join(save_path, r"snapshot_{}_**".format(snapshot_num))
        )
        print(field_save_path)
        np.savetxt(
            fname=os.path.join(field_save_path, "field_stars.txt"),
            X=pop2_data_field,
            header=header,
        )


#%%
# plt.figure(figsize=(5, 4), dpi=200)
# plt.plot(time_myr, total_counts, label="total counts")
# plt.plot(time_myr, gc_counts, label="inside globular clusters")
# plt.legend()
