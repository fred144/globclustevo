import numpy as np
import os
import matplotlib.pyplot as plt
from lum_funcs import unpack_pop_ii_data
from lum_plotting_lib import king_profiler, star_luminosity_plot
import sys


plt.rcParams.update({"figure.max_open_warning": 0})
# mpl.rc('font', family='serif')
# mpl.rc('text', usetex=True)
plt.style.use("dark_background")

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })

def run_profiler(file_name, proj_width, gc_radii, lum_map_bins):
    """
    Runs the profiler for a given snapshot. Finds centers and loops over centers.
    Makes profiles and returns properties of GCs found. 
    """
    
    print("# read in:", file_name)

    time_str = file_name[23:29].replace("_", ".")  # in myr
    time = float(time_str)
    snapshot_num = int(file_name[17:22])

    save_name = "./gc_profiles/snapshot_{}_{}/".format(
        str(snapshot_num).zfill(4),
        str(time).ljust(6, "0").replace(".", "_"),
    )

    print("> snapshot time", time, "Myr")
    print("> snapshot number", snapshot_num)
    print("> uniform radius of", gc_radii, "pc")

    star_positions, scaled_stellar_lums, masses, ages, t_myr = unpack_pop_ii_data(
        file_name
        )
    # get center x and y coordinates
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
        )

    if not os.path.exists(save_name):
        print("# Creating new sequence directory", save_name)
        os.makedirs(save_name)
    
    # save the figure with annotated found GCs
    plt.savefig(
        save_name + "annotated_gcs.png", dpi=300, bbox_inches="tight", pad_inches=0.05
        )
    
    # loop over the centers, make profiles, and get data on a cluster basis.
    gc_tot_masses = []
    gc_r_core = []
    gc_m_core = []
    gc_r_trunc = []
    gc_char_age = []
    gc_alpha = []
    gc_sigma0 = []
    gc_sigmabg = []

    test_ctrs = np.array([peak_x, peak_y]).T
    
    # iterate over x,y maximas and plot
    for ctr, label in zip(test_ctrs, gc_labels):
        # print(ctr)
        # print(label)
        (_, _, _, m_tot, r_c, m_r_c, r_trunc, char_age, alpha, sigma_0, sigma_bg,) = king_profiler(
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
        gc_tot_masses.append(m_tot)
        gc_r_core.append(r_c)
        gc_m_core.append(m_r_c)
        gc_r_trunc.append(r_trunc)
        gc_char_age.append(char_age)
        gc_alpha.append(alpha)
        gc_sigma0.append(sigma_0)
        gc_sigmabg.append(sigma_bg)

        if m_tot > 0:
            plt.savefig(
                save_name + "gc_{}".format(str(label).zfill(3)),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.05,
                )
        elif m_tot == -2:
            # save the failed fits
            plt.savefig(
                save_name + "no_fit_gc_{}".format(str(label).zfill(3)),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.05,
                )
        
        plt.close() 
        
    # turn into arrays so we can index them and then clean up
    gc_tot_masses = np.array(gc_tot_masses)
    gc_r_core = np.array(gc_r_core)
    gc_m_core = np.array(gc_m_core)
    gc_r_trunc = np.array(gc_r_trunc)
    gc_char_age = np.array(gc_char_age)
    gc_alpha = np.array(gc_alpha)
    gc_sigma0 = np.array(gc_sigma0)
    gc_sigmabg = np.array(gc_sigmabg)
    
    # mask out invalid values
    mask = gc_tot_masses > 0
    gc_tot_masses = gc_tot_masses[mask]
    gc_r_core = gc_r_core[mask]
    gc_m_core = gc_m_core[mask]
    gc_r_trunc = gc_r_trunc[mask]
    gc_char_age = gc_char_age[mask]
    gc_alpha = gc_alpha[mask]
    gc_sigma0 = gc_sigma0[mask]
    gc_sigmabg = gc_sigmabg[mask]

    print("> found", gc_char_age.size, "good profiles")
    
    output = np.vstack(
        (
            gc_char_age,
            gc_tot_masses,
            gc_m_core,
            gc_r_trunc,
            gc_r_core,
            gc_alpha,
            gc_sigma0,
            gc_sigmabg,
        )
    ).T
    # comment = "These are just the succesful fits with reasonable alpha."
    header = (
        "Age[Myr] \t\t\t Masses[Msun]"
        "\t\t\t CoreMass[Msun] \t\t\t  TruncRadii[pc]"
        "\t\t\t  CoreRadii[pc] \t\t\t  FitAlpha"
        "\t\t\t FitSigma0 \t\t\t  FitSigmaBg"
    )
    np.savetxt(fname=save_name + "info.txt", X=output, header=header)

    return gc_tot_masses, gc_r_core, gc_m_core, gc_r_trunc, gc_char_age, time




if __name__ == "__main__":

    # data_file = "./pop_2_data/pos_00418_462_09_myr.txt"
    data_file = "./pop_2_data/pos_00486_476_23_myr.txt"

    folder_name = "./gc_profiles/snapshot_" + data_file[18:-8]
    
    if not os.path.exists(folder_name):
        print("# Creating new sequence directory", folder_name)
        os.makedirs(folder_name)
        
    # put all verbose output into a text file
    # orig_stdout = sys.stdout
    # sys.stdout = open(folder_name + "/log.txt", "w")

    masses, core_radii, core_masses, r_trunc, ages, time = run_profiler(
        file_name=data_file,
        proj_width=400,
        gc_radii=10,
        lum_map_bins=1000,
        )

    core_diameter = core_radii * 2


    # sys.stdout.close()
    # sys.stdout = orig_stdout
