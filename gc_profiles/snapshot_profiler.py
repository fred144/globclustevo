import os
import sys

sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/clean-install/lib/python3.7/site-packages"
)
sys.path.append("..")  # makes sure that importing the modules in the main folder work
from modules.luminosity.lum_functions import unpack_pop_ii_data
from modules.luminosity.lum_plotting import star_luminosity_plot
from modules.profiles.profile_plotting import king_profile_plotter
import matplotlib.pyplot as plt
import numpy as np


# TODO: chagen the profiles directory.
def run_profiler(
    star_file_path,
    parent_save_path,
    proj_width,
    gc_radii,
    lum_map_bins,
    centers=None,
    **kwargs
):
    """
    Runs the profiler for a given snapshot.
    Finds centers and loops over centers.
    Makes profiles and returns properties of GCs found.
    """
    star_positions, scaled_stellar_lums, masses, ages, t_myr = unpack_pop_ii_data(
        star_file_path
    )

    print("# read in:", star_file_path)
    snapshot_num = star_file_path.split("/")[-1].split("_")[1]
    time = np.round(t_myr, 2)
    save_folder = "snapshot_{}_t_{}".format(
        snapshot_num,
        str(time).replace(".", "_"),
    )

    # save folder for the snapshot, make it if it doesnt exist
    save_folder_abs_path = os.path.join(parent_save_path, save_folder)
    if not os.path.exists(save_folder_abs_path):
        print("# Creating new sequence directory", save_folder_abs_path)
        os.makedirs(save_folder_abs_path)

    print("> snapshot time", time, "Myr")
    print("> snapshot number", int(snapshot_num))

    # allow non-uniform, user-defined radii
    if np.size(gc_radii) > 1:
        print("> nonuniform radius, using virial or other")
    else:
        print("> uniform radius of", gc_radii, "pc")

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
        # centers are contained in the files
        gc_ctrs = centers[:, 1:4]
        gc_labels = centers[:, 0]

        # make a master plot of all the gcs, just for visualization
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
            plt_ctrs=np.column_stack((gc_labels, gc_ctrs)),
            masses=masses,
            **kwargs
        )

    # save the figure with annotated found GCs
    annotated_gc_save = os.path.join(save_folder_abs_path, "annotated_gcs.png")
    plt.savefig(annotated_gc_save, dpi=300, bbox_inches="tight", pad_inches=0.05)
    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    plt.close("all")

    # loop over the centers, make profiles, and get data on a cluster basis.
    gc_out_masses = []  # trunc mass (for uniform) or  mass within gc_radii (user_def)
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
    gc_particle_counts = []

    # iterate over x,y maximas and plot
    for i, (ctr, label) in enumerate(zip(gc_ctrs, gc_labels)):
        label = int(label)
        # can allow non uniform radii.
        if np.size(gc_radii) > 1:
            radius = gc_radii[i]
        else:
            pass
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
            counts,
        ) = king_profile_plotter(
            star_pos=star_positions,
            lums=scaled_stellar_lums,
            masses=masses,
            ages=ages,
            gc_ctr=ctr,
            gc_rad=radius,
            gc_label=label,
            bins=25,
            # x=gc_radii[i],
        )
        # per globular cluster inside a snap shot
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
        gc_particle_counts.append(counts)

        if m_tot > 0:
            plt_save_path = os.path.join(
                save_folder_abs_path, "gc_{}.png".format(str(label).zfill(3))
            )
            plt.savefig(plt_save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        elif m_tot == -2:
            # save the failed fits
            plt_save_path = os.path.join(
                save_folder_abs_path, "no_fit_gc_{}.png".format(str(label).zfill(3))
            )
            plt.savefig(plt_save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)

        plt.cla()
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
    gc_particle_counts = np.array(gc_particle_counts)

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
    gc_particle_counts = gc_particle_counts[mask]

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
    info_save_path = os.path.join(save_folder_abs_path, "info.txt")
    np.savetxt(fname=info_save_path, X=output, header=header)

    return (
        gc_out_masses,
        gc_r_core,
        gc_m_core,
        gc_r_trunc,
        gc_char_age,
        time,
        gc_particle_counts,
    )
