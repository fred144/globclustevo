import os
import sys
import yt

sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/cosmology-clean/lib/python3.7/site-packages"
)
import numpy as np
import matplotlib.pyplot as plt
from lum_funcs import unpack_pop_ii_data
from lum_plotting_lib import king_profiler, star_luminosity_plot
from yt.funcs import mylog
import warnings


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


def filter_snapshots(folder_path, start_snap: int, end_snap: int, sampling=1):
    r"""Given a directory of outputs, return a list of absolute file
    paths given a range of snapshot values. Enables discrete selection
    of time range based on snapshot number.

    """
    strt_string = str(start_snap).zfill(5)
    end_string = str(end_snap).zfill(5)

    files = sorted(os.listdir(folder_path))

    strt_idx = [i for i, s in enumerate(files) if strt_string in s][0]
    end_idx = [i for i, s in enumerate(files) if end_string in s][0]

    filtered_files = files[strt_idx : end_idx + 1 : sampling]

    abs_paths = [os.path.join(folder_path, file) for file in filtered_files]

    return abs_paths


def run_profiler(file_name, proj_width, gc_radii, lum_map_bins, centers=None, **kwargs):
    """
    Runs the profiler for a given snapshot. Finds centers and loops over centers.
    Makes profiles and returns properties of GCs found.
    """

    print("# read in:", file_name)
    # TODO: need to change all these, relatively arbiray
    time_str = file_name[36:42].replace("_", ".")  # in myr
    time = float(time_str)
    snapshot_num = int(file_name.split("_")[4])

    save_name = "./gc_profiles/snapshot_{}_{}/".format(
        str(snapshot_num).zfill(5),
        str(time).ljust(6, "0").replace(".", "_"),
    )

    print("> snapshot time", time, "Myr")
    print("> snapshot number", snapshot_num)
    print("> uniform radius of", gc_radii, "pc")

    star_positions, scaled_stellar_lums, masses, ages, t_myr = unpack_pop_ii_data(
        file_name
    )
    if centers is None:
        # get center x and y coordinates if centers are not provided.
        peak_x, peak_y, gc_labels = star_luminosity_plot(
            proj_width=proj_width,
            star_positions=star_positions,
            scaled_stellar_lums=scaled_stellar_lums,
            time=t_myr,
            snapshot_num=snapshot_num,
            pi_multiple=0,
            plt_bins=lum_map_bins,
            lum_scale=("dynamic", 0, 0),
            get_ctr=(True, "potential", 0.04, True),
            masses=masses,
            **kwargs
        )
        gc_ctrs = np.array([peak_x, peak_y]).T
    else:
        gc_ctrs = np.array(centers[:, 0:3])
        gc_labels = np.array(centers[:, 3]).astype(int)
        # make a master plot of all the gcs
        star_luminosity_plot(
            proj_width=proj_width,
            star_positions=star_positions,
            scaled_stellar_lums=scaled_stellar_lums,
            time=t_myr,
            snapshot_num=snapshot_num,
            pi_multiple=0,
            lum_scale=("dynamic", 0, 0),
            plt_bins=lum_map_bins,
            get_ctr=None,
            plt_ctrs=centers,
            masses=masses,
            **kwargs
        )

    if not os.path.exists(save_name):
        print("# Creating new sequence directory", save_name)
        os.makedirs(save_name)

    # save the figure with annotated found GCs
    plt.savefig(
        save_name + "annotated_gcs.png", dpi=300, bbox_inches="tight", pad_inches=0.05
    )

    # loop over the centers, make profiles, and get data on a cluster basis.
    gc_out_masses = []  # can either be trunc mass or all mass within gc_radii
    gc_r_core = []
    gc_err_rc = []
    gc_m_core = []
    gc_r_trunc = []
    gc_char_age = []
    gc_alpha = []
    gc_err_alpha = []
    gc_sigma0 = []
    gc_err_sigma_0 = []
    gc_sigmabg = []
    gc_err_sigma_bg = []
    # print(gc_labels)
    # iterate over x,y maximas and plot
    for ctr, label in zip(gc_ctrs, gc_labels):

        # print(ctr)
        # print(label)
        (
            _,
            _,
            _,
            m_tot,
            r_c,
            err_rc,
            m_r_c,
            r_trunc,
            char_age,
            alpha,
            err_alpha,
            sigma_0,
            err_sigma_0,
            sigma_bg,
            err_sigma_bg,
        ) = king_profiler(
            star_pos=star_positions,
            lums=scaled_stellar_lums,
            masses=masses,
            ages=ages,
            gc_ctr=ctr,
            gc_rad=gc_radii,
            gc_label=label,
            bins=25,
        )
        # print(char_age )
        gc_out_masses.append(m_tot)
        gc_r_core.append(r_c)
        gc_err_rc.append(err_rc)
        gc_m_core.append(m_r_c)
        gc_r_trunc.append(r_trunc)
        gc_char_age.append(char_age)
        gc_alpha.append(alpha)
        gc_err_alpha.append(err_alpha)
        gc_sigma0.append(sigma_0)
        gc_err_sigma_0.append(err_sigma_0)
        gc_sigmabg.append(sigma_bg)
        gc_err_sigma_bg.append(err_sigma_bg)

        if m_tot > 0:
            plt.savefig(
                save_name + "gc_{}.png".format(str(label).zfill(3)),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.05,
            )
        elif m_tot == -2:
            # save the failed fits
            plt.savefig(
                save_name + "no_fit_gc_{}.png".format(str(label).zfill(3)),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.05,
            )

        plt.close()

    # turn into arrays so we can index them and then clean up
    gc_out_masses = np.array(gc_out_masses)
    gc_r_core = np.array(gc_r_core)
    gc_err_rc = np.array(gc_err_rc)
    gc_m_core = np.array(gc_m_core)
    gc_r_trunc = np.array(gc_r_trunc)
    gc_char_age = np.array(gc_char_age)
    gc_alpha = np.array(gc_alpha)
    gc_err_alpha = np.array(gc_err_alpha)
    gc_sigma0 = np.array(gc_sigma0)
    gc_err_sigma_0 = np.array(gc_err_sigma_0)
    gc_sigmabg = np.array(gc_sigmabg)
    gc_err_sigma_bg = np.array(gc_err_sigma_bg)

    # mask out invalid values
    mask = gc_out_masses > 0
    gc_out_masses = gc_out_masses[mask]
    gc_r_core = gc_r_core[mask]
    gc_err_rc = gc_err_rc[mask]
    gc_m_core = gc_m_core[mask]
    gc_r_trunc = gc_r_trunc[mask]
    gc_char_age = gc_char_age[mask]
    gc_alpha = gc_alpha[mask]
    gc_err_alpha = gc_err_alpha[mask]
    gc_sigma0 = gc_sigma0[mask]
    gc_err_sigma_0 = gc_err_sigma_0[mask]
    gc_sigmabg = gc_sigmabg[mask]
    gc_err_sigma_bg = gc_err_sigma_bg[mask]
    gc_labels = gc_labels[mask]
    print("> found", gc_char_age.size, "good profiles")

    # make the time get along with the rest of dimensions
    t_myr = np.array(time)
    t_myr.resize(np.size(gc_char_age))

    output = np.vstack(
        (
            t_myr,
            gc_labels,
            gc_char_age,
            gc_out_masses,
            gc_m_core,
            gc_r_trunc,
            gc_r_core,
            gc_err_rc,
            gc_alpha,
            gc_err_alpha,
            gc_sigma0,
            gc_err_sigma_0,
            gc_sigmabg,
            gc_err_sigma_bg,
        )
    ).T
    # comment = "These are just the succesful fits with reasonable alpha."
    header = (
        "\t\tTime[Myr] \t\t GC Label"
        "\t\t Age[Myr] \t\t TotalOrTruncreMass[Msun]"
        "\t CoreMass[Msun] \t  TruncRadii[pc]"
        "\t CoreRadii[pc] \t ErrCoreRadii"
        "\t FitAlpha \t ErrFitAlpha"
        "\t FitSigma0 \t  ErrFitSigma0"
        "\t FitSigmaBG \t  ErrFitSigmaBG"
    )
    np.savetxt(fname=save_name + "info.txt", X=output, header=header)

    return gc_out_masses, gc_r_core, gc_m_core, gc_r_trunc, gc_char_age, time


if __name__ == "__main__":
    data_directory = r"./pop_2_data/"
    pop2_data_directory = r"../pop_2_data/fs07_refine"
    halo_data_directory = r"../halo_data/fs07_refine/fof"
    pop2_data_set = filter_snapshots(pop2_data_directory, 200, 800, 10)
    halo_data_directory = filter_snapshots(halo_data_directory, 200, 800, 10)

    for pop_data, hc_data in zip(pop2_data_set, halo_data_directory):
        # get where the stars are centered
        pop2_data = np.loadtxt(pop_data)
        ctr_at = pop2_data[5:8, 6]  # pc

        # get the hdf5 catalogue inside each folder
        halo_cat = yt.load(os.path.join(hc_data, os.listdir(hc_data)[0]))
        halo_data = halo_cat.all_data()
        # get centers of halos
        # specific to FOF halo finder output
        x_halos = np.array(halo_data["all", "particle_position_x"].to("pc")) - ctr_at[0]
        y_halos = np.array(halo_data["all", "particle_position_y"].to("pc")) - ctr_at[1]
        z_halos = np.array(halo_data["all", "particle_position_z"].to("pc")) - ctr_at[2]
        ids = np.array(halo_data["all", "particle_identifier"])

        gc_centers = np.vstack((x_halos, y_halos, z_halos, ids)).T  # pc

        # data_file = "./pop_2_data/pos_00418_462_09_myr.txt"
        # data_file = "./pop_2_data/pos_00486_476_23_myr.txt"
        # data_file = data_directory + file_name
        # make output directory
        folder_name = "./gc_profiles/snapshot_" + pop_data[30:-8]

        if not os.path.exists(folder_name):
            print("# Creating new sequence directory", folder_name)
            os.makedirs(folder_name)

        # put all verbose output into a text file
        orig_stdout = sys.stdout
        sys.stdout = open(folder_name + "/log.txt", "w")

        masses, core_radii, core_masses, r_trunc, ages, time = run_profiler(
            file_name=pop_data,
            proj_width=400,
            gc_radii=20,  # uniform radii to be used to extract clusters
            lum_map_bins=1000,  # bad resolution so that you can see it better
            centers=gc_centers,
        )

        sys.stdout.close()
        sys.stdout = orig_stdout

    # for file_name in filtered_files:

    #     # data_file = "./pop_2_data/pos_00418_462_09_myr.txt"
    #     # data_file = "./pop_2_data/pos_00486_476_23_myr.txt"
    #     data_file = data_directory + file_name
    #     # make output directory
    #     folder_name = "./gc_profiles/snapshot_" + data_file[18:-8]

    #     if not os.path.exists(folder_name):
    #         print("# Creating new sequence directory", folder_name)
    #         os.makedirs(folder_name)

    #     # put all verbose output into a text file
    #     orig_stdout = sys.stdout
    #     sys.stdout = open(folder_name + "/log.txt", "w")

    #     masses, core_radii, core_masses, r_trunc, ages, time = run_profiler(
    #         file_name=data_file,
    #         proj_width=400,
    #         gc_radii=20,  # uniform radii to be used to extract clusters
    #         lum_map_bins=1000,  # bad resolution so that you can see it better
    #         num_ctr=250,
    #         ctr_dist_thresh=5,
    #         ctr_rel_thresh=0.001,
    #     )

    #     sys.stdout.close()
    #     sys.stdout = orig_stdout
