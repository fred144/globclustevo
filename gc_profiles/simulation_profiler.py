import os
import sys

sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/clean-install/lib/python3.7/site-packages"
)
sys.path.append("..")  # makes sure that importing the modules in the main folder work

from modules.macros import filter_snapshots
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


def fof_profiler(pop2_data_set, halo_data_directory, run_save_path, gc_radii=None):
    gc_counts = []
    total_counts = []
    time_myr = []
    # loop over snapshots
    for pop_data, hc_data in zip(pop2_data_set, halo_data_directory):

        pop2_data = np.loadtxt(pop_data)

        # load the processed catalogue, recentered and in pc
        catalogue_text_file = os.path.join(hc_data, sorted(os.listdir(hc_data))[1])
        halo_data = np.loadtxt(catalogue_text_file)
        ctrs_pc = halo_data[:, :-1]  # centers with label in the first columns
        vir_radii = halo_data[:, -1]
        # put all verbose output into a text file
        # orig_stdout = sys.stdout
        # sys.stdout = open(os.path.join(run_save_path, "log.txt"), "w")

        (
            masses,
            core_radii,
            core_masses,
            r_trunc,
            ages,
            time,
            particles_in_this_snapshot,
        ) = run_profiler(
            star_file_path=pop_data,
            parent_save_path=run_save_path,
            proj_width=400,  # pc
            gc_radii=vir_radii,  # uniform radii to be used to extract clusters
            lum_map_bins=4000,  # bad resolution so that you can see it better
            centers=ctrs_pc,
        )
        print(particles_in_this_snapshot)
        gc_counts.append(np.sum(particles_in_this_snapshot))
        total_counts.append(pop2_data.shape[0])
        time_myr.append(time)

        # undo put all verbose output into a text file
        # sys.stdout.close()
        # sys.stdout = orig_stdout


if __name__ == "__main__":

    pop2_data_directory = r"../particle_data/pop_2_data/fs07_refine"
    halo_data_directory = r"../halo_data/fs07_refine/test_run"

    save_path = "./profile_runs/fs07_refine/test_run/10_pc_axvline"

    strt = 250
    end = 250
    step = 1
    pop2 = filter_snapshots(pop2_data_directory, strt, end, step)
    halos = filter_snapshots(halo_data_directory, strt, end, step)

    fof_profiler(
        pop2_data_set=pop2, halo_data_directory=halos, run_save_path=save_path
    )


#%%
# plt.figure(figsize=(5, 4), dpi=200)
# plt.plot(time_myr, total_counts, label="total counts")
# plt.plot(time_myr, gc_counts, label="inside globular clusters")
# plt.legend()
