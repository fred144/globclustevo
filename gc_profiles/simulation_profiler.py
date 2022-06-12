import sys

sys.path.append("..")
sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/clean-install/lib/python3.7/site-packages"
)
import glob
import os
from modules.macros import filter_snapshots, common_filter_snapshots
from modules.luminosity.lum_functions import lum_look_up_table
from snapshot_profiler import run_profiler
import matplotlib.pyplot as plt
import numpy as np
import yt
import warnings

yt.enable_parallelism()
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

    if not os.path.exists(run_save_path):
        print("# Creating new profiler run directory", run_save_path)
        os.makedirs(run_save_path)

    snapshot = []
    gc_counts_in_snap = []
    gc_masses_in_snap = []
    gc_lums_in_snap = []
    gc_core_masses_in_snap = []
    total_counts_in_snap = []
    total_masses_in_snap = []
    total_lums_in_snap = []
    time_myr_in_snap = []
    redshift_in_snap = []

    # loop over snapshots
    for pop_data, hc_data in zip(pop2_data_set, halo_data_directory):
        pop_data = pop_data.replace("\\", "/")

        snapshot_num = pop_data.split("/")[-1].split("_")[1]

        pop2_data = np.loadtxt(pop_data)

        # load the processed catalogue, recentered and in pc
        catalogue_text_file = os.path.join(
            hc_data, "catalogue_{}.txt".format(snapshot_num)
        )
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
            gc_lums,
            core_radii,
            core_masses,
            r_trunc,
            ages,
            time,
            particles_in_this_snapshot,
            pop_2_ids_in_gcs,
            pop_2_ids_in_fitted_gcs,  # profiler run dictated
            scaled_stellar_lums,  # for all stars, UV [erg s^-1 Angstrom-^1]
        ) = run_profiler(
            star_file_path=pop_data,
            parent_save_path=save_path,
            proj_width=400,  # pc
            gc_radii=profile_gcs_up_to,  # radii to be used to extract clusters
            lum_map_bins=4000,  # bad resolution so that you can see it better
            centers=ctrs_pc,
            particle_filter=hc_data,
        )
        # modify pop2 by tacking on luminosities
        pop2_data = np.insert(pop2_data, 2, scaled_stellar_lums, axis=1)

        mask = np.isin(pop2_data[:, 0], pop_2_ids_in_gcs)
        pop2_data_field = pop2_data[~mask]
        pop2_data_bound = pop2_data[mask]

        mask_only_fitted = np.isin(pop2_data[:, 0], pop_2_ids_in_fitted_gcs)
        fitted_pop2_data_field = pop2_data[~mask_only_fitted]
        fitted_pop2_data_bound = pop2_data[mask_only_fitted]

        gc_counts_in_snap.append(np.sum(particles_in_this_snapshot))
        # masses within the vir_radius, full cluster
        gc_masses_in_snap.append(np.sum(gc_full_masses))
        gc_core_masses_in_snap.append(np.sum(core_masses))  # Msun
        # total light in gcs, profiler determined
        gc_lums_in_snap.append(np.sum(gc_lums))

        total_masses_in_snap.append(np.sum(pop2_data[:, 5]))  # Msun
        total_lums_in_snap.append(np.sum(scaled_stellar_lums))
        total_counts_in_snap.append(pop2_data.shape[0])
        snapshot.append(int(snapshot_num))
        time_myr_in_snap.append(time)
        redshift_in_snap.append(pop2_data[1, 6])

        header = (
            "\t\tID"
            "\t\tCurrentAges[Myr]"
            "\t CurrentLums[erg s^-1 Angstrom-^1]"
            "\t\tX[pc] "
            "Y[pc] "
            "Z[pc] "
            "\t\tmass[Msun]"
            "\t\tt_sim[Myr], z, ctr(code), ctr(pc)"
        )
        # save, regardless of profiler outcome
        field_save_path = os.path.join(hc_data, "field_stars.txt")
        clust_save_path = os.path.join(hc_data, "bound_stars.txt")
        # save only those that weer succesfully fitted
        fitted_field_save_path = os.path.join(hc_data, "fitted_field_stars.txt")
        fitted_clust_save_path = os.path.join(hc_data, "fitted_bound_stars.txt")
        # save the feild / bound stars as seen by the halo finder,
        # doesn't care if fitted or not
        np.savetxt(
            fname=field_save_path,
            X=pop2_data_field,
            header=header,
        )
        np.savetxt(
            fname=clust_save_path,
            X=pop2_data_bound,
            header=header,
        )

        np.savetxt(
            fname=fitted_field_save_path,
            X=fitted_pop2_data_field,
            header=header,
        )
        np.savetxt(
            fname=fitted_clust_save_path,
            X=fitted_pop2_data_bound,
            header=header,
        )

        # undo put all verbose output into a text file
        # sys.stdout.close()
        # sys.stdout = orig_stdout

        # profiler run stats, one row per snapshot, save per loop for crashes
        save_gc_counts = np.array(gc_counts_in_snap)
        save_gc_masses = np.array(gc_masses_in_snap)
        save_gc_lums = np.array(gc_lums_in_snap)
        save_gc_core_masses = np.array(gc_core_masses_in_snap)
        save_total_counts = np.array(total_counts_in_snap)
        save_total_masses = np.array(total_masses_in_snap)
        save_total_lums = np.array(total_lums_in_snap)
        save_snapshot = np.array(snapshot)
        save_time_myr = np.array(time_myr_in_snap)
        save_redshift = np.array(redshift_in_snap)

        header = (
            "\t Snapshotnum"
            "\t Time[MYR]"
            "\t redshift"
            "\t total_counts"
            "\t gc_counts"
            "\t total_masses[Msun]"
            "\t gc_masses[Msun]"
            "\t gc_core_masses[Msun]"
            "\t total_lums[erg s^-1 Angstrom-^1]"
            "\t gc_lums[erg s^-1 Angstrom-^1]"
        )

        output = np.vstack(
            (
                save_snapshot,
                save_time_myr,
                save_redshift,
                save_total_counts,
                save_gc_counts,
                save_total_masses,
                save_gc_masses,
                save_gc_core_masses,
                save_total_lums,
                save_gc_lums,
            )
        ).T
        save_name = os.path.join(run_save_path, "time_series_run_stats.txt")

        np.savetxt(fname=save_name, X=output, header=header)


if __name__ == "__main__":

    pop2_data_directory = r"../particle_data/pop_2_data/fs07_refine"
    halo_data_directory = r"../halo_data/fs07_refine/fof_best"

    save_path = "./profile_runs/fs07_refine/fof_best_v3"

    strt = 113
    end = 918
    step = 1
    pop2 = filter_snapshots(pop2_data_directory, strt, end, step)
    halos = filter_snapshots(halo_data_directory, strt, end, step)

    pop2 = common_filter_snapshots(pop2, halos)

    fof_profiler(
        pop2_data_set=pop2,
        halo_data_directory=halos,
        run_save_path=save_path,
        gc_radii="vir_rad",
    )


# plt.figure(figsize=(5, 4), dpi=200)
# plt.plot(time_myr, total_counts, label="total counts")
# plt.plot(time_myr, gc_counts, label="inside globular clusters")
# plt.legend()
