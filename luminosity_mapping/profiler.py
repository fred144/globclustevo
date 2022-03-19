import numpy as np
import os
import matplotlib.pyplot as plt
from lum_funcs import (
    chi_squared,
    get_cluster,
    unpack_pop_ii_data,
    projected_surf_densities,
)
from lum_plotting_lib import star_luminosity_plot
from scipy.optimize import curve_fit
import matplotlib as mpl
import scipy.stats as st
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


def king_model(r, k, r_c, r_t):
    """
    https://articles.adsabs.harvard.edu/pdf/1962AJ.....67..471K
    """
    f = (
        k
        * ((1 / np.sqrt(1 + (r / r_c) ** 2)) - (1 / np.sqrt(1 + (r_t / r_c) ** 2))) ** 2
    )
    return f


def modified_king_model(r, sigma_naught, r_c, alpha, bg):

    sigma = bg + (sigma_naught / (1 + (r / r_c) ** alpha))
    return sigma


def trunc_radius(sigma_0, r_c, alpha, sigma_bg):
    """
    set to 1.5bg =  bg + (peak)/( 1 + (r/r_c)^alpha)
    0.5bg = (peak)/( 1 + (r/r_c)^alpha)
    """
    # trunc_r = (r_c**alpha * ((sigma_0/(.5*sigma_bg)) - 1) )**(1/alpha)
    r_trunc = (r_c ** alpha * ((sigma_0 / ((1.5 - 1) * sigma_bg) - 1))) ** (1 / alpha)
    return r_trunc


def get_masses(x_coord, y_coord, masses, r_characteristic):
    """
    get core mass or any mass enclosed a characteristic radius
    """
    all_positions = np.vstack((x_coord, y_coord)).T
    distances = np.sqrt(np.sum(np.square(all_positions), axis=1))
    mask = distances <= r_characteristic
    core_mass = np.sum(masses[mask])
    return core_mass


def king_profiler(star_pos, lums, masses, ages, gc_ctr, gc_rad, gc_label, bins=30):
    """
    depends on projected_surf_densities
    """
    # get a cluster given a center --> (0,0,0) and spherically mask around it
    # returns positons (recentered), masses, luminosites, and ages of each star
    if gc_ctr.size == 2:
        clust_x, clust_y, clust_lums, clust_masses, clust_ages = get_cluster(
            xpos=star_pos[:, 0],
            ypos=star_pos[:, 1],
            zpos=None,
            ctr_at=gc_ctr,
            masses=masses,
            ages=ages,
            cluster_radius=gc_rad,
            lums=lums,
            trns_coord=True,
        )
    else:
        # 3d case
        clust_x, clust_y, clust_z, clust_lums, clust_masses, clust_ages = get_cluster(
            xpos=star_pos[:, 0],
            ypos=star_pos[:, 1],
            zpos=star_pos[:, 2],
            ctr_at=gc_ctr,
            masses=masses,
            ages=ages,
            cluster_radius=gc_rad,
            lums=lums,
            trns_coord=True,
        )

    # given an isolated cluster, find projected density quantities
    # as a function of radius with log bins or linear bins
    r, rho, err, tot_m = projected_surf_densities(
        x_coord=clust_x,
        y_coord=clust_y,
        lums=clust_lums,
        masses=clust_masses,
        radius=gc_rad,
        num_bins=bins,
        log_bins=True,
        dr=None,
        )
    gc_char_age = float(st.mode(clust_ages)[0] / 1e6)

    plt.close()
    plt.figure(figsize=(8, 8), dpi=200)
    plt.errorbar(
        r,
        rho,
        yerr=err,
        fmt="o",
        capsize=5,
        capthick=3,
        elinewidth=3,
        label=(
            r"$M_{{total}}= {:.2e} \: M_{{\odot}}$" "\n" r"$t_{{age}}= {:.2f}$ Myr"
        ).format(tot_m, gc_char_age),
        )

    try:
        # try to do the fit
        fit_params, cov_matrix = curve_fit(
            f=modified_king_model,
            xdata=r,
            ydata=rho,
            sigma=err,
            absolute_sigma=True,
            p0=[1e4, 0.2, 2, 10],
            bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
            )
        # r, sigma_naught, r_c, alpha, bg
        fit_sigma = np.sqrt(np.diag(cov_matrix))
        # print(fit_params)
        # print(fit_sigma)

        # calculate theoretical best fit
        theory_rho = modified_king_model(r, *fit_params)

        # ===========================calc derived quantities===========================
        fit_sigma_naught = fit_params[0]
        fit_r_c = fit_params[1]
        fit_alpha = fit_params[2]
        fit_sigma_bg = fit_params[3]
        truncation_radius = trunc_radius(
            r_c=fit_r_c,
            alpha=fit_alpha,
            sigma_0=fit_sigma_naught,
            sigma_bg=fit_sigma_bg,
            )
        core_mass = get_masses(clust_x, clust_y, clust_masses, fit_r_c)

        # =============================================================================

        # quantify goodness of fit
        p_value, reduced_chi_2 = chi_squared(
            theory=theory_rho, data=rho, sigma=err, num_params=4
        )
        if truncation_radius <= 20:
            plot_label = (
                r"$R_{{trunc}} = {:.2f} \: pc$"
                "\n"
                r"$R_{{core}} = {:.2f} \: pc$"
                "\n"
                r"$\alpha = {:.2f} $"
                "\n"
                r"$\Sigma_0 = {:.2e} $"
                "\n"
                r"$\Sigma_{{bg}} = {:.2f} $"
                "\n"
                r"$M_{{r_c}} = {:.2e} \: M_{{\odot}}$"
            ).format(
                truncation_radius,
                fit_r_c,
                fit_alpha,
                fit_sigma_naught,
                fit_sigma_bg,
                core_mass,
            )
        else:
            plot_label = (
                r"$R_{{trunc}} > 20 $ pc"
                "\n"
                r"$R_{{core}} = {:.2f}$ pc"
                "\n"
                r"$\alpha = {:.2f} $"
                "\n"
                r"$\Sigma_0 = {:.2e} $"
                "\n"
                r"$\Sigma_{{bg}} = {:.2f} $"
                "\n"
                r"$M_{{r_c}} = {:.2e} \: M_{{\odot}}$"
            ).format(fit_r_c, fit_alpha, fit_sigma_naught, fit_sigma_bg, core_mass)

        # plot the fit if it is good
        if fit_alpha < 4 and core_mass > 1:

            plt.figure(figsize=(8, 8), dpi=200)
            plt.errorbar(
                r,
                rho,
                yerr=err,
                fmt="o",
                capsize=5,
                capthick=3,
                elinewidth=3,
                label=(
                    r"$M_{{total}}= {:.2e} \: M_{{\odot}}$"
                    "\n"
                    r"$t_{{age}}= {:.2f}$ Myr"
                ).format(tot_m, gc_char_age),
                )
            plt.plot(r, theory_rho, linewidth=4, label=plot_label)
            # plt.axvline(fit_r_c)
            # plt.text(fit_r_c, 0,r'$R_core$',rotation=90)
            plt.title(r"GC # {:.0f}".format(gc_label), fontsize=16)
            plt.ylabel(r"Surface Mass Density ($M_{\odot} \; pc^{-2}$)", fontsize=16)
            plt.xlabel(r"R ($pc$)", fontsize=16)
            plt.xscale("log")
            plt.yscale("log")
            plt.grid(visible=True, which="both", axis="y", ls="--")
            plt.legend(fontsize=16)
            return (
                r,
                rho,
                err,
                tot_m,
                fit_r_c,
                core_mass,
                truncation_radius,
                gc_char_age,
                fit_alpha,
                fit_sigma_naught,
                fit_sigma_bg,
                )
        else:
            print(r"> bad alpha for GC #{:.0f}".format(gc_label))
            # return invalid values
            return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2

    except:
        # if it can't fit it

        plt.title(r"No Fit GC # {:.0f}".format(gc_label), fontsize=16)
        plt.ylabel(r"Surface Mass Density ($M_{\odot} \; pc^{-2}$)", fontsize=16)
        plt.xlabel(r"R ($pc$)", fontsize=16)
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(visible=True, which="both", axis="y", ls="--")
        plt.legend(fontsize=16)

        print(r"> can't fit GC #{:.0f}".format(gc_label))
        # return invalid values
        return -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2


def run_profiler(file_name, proj_width, gc_radii, lum_map_bins):
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
            plt.savefig(
                save_name + "no_fit_gc_{}".format(str(label).zfill(3)),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.05,
                )

    # turn into arrays so we can index them and then clean up
    gc_tot_masses = np.array(gc_tot_masses)
    gc_r_core = np.array(gc_r_core)
    gc_m_core = np.array(gc_m_core)
    gc_r_trunc = np.array(gc_r_trunc)
    gc_char_age = np.array(gc_char_age)
    gc_alpha = np.array(gc_alpha)
    gc_sigma0 = np.array(gc_sigma0)
    gc_sigmabg = np.array(gc_sigmabg)

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
    comment = "# These are just the succesful fits with reasonable alpha."
    header = (
        "Age[Myr] \t\t\t Masses[Msun]"
        "\t\t\t CoreMass[Msun] \t\t\t  TruncRadii[pc]"
        "\t\t\t  CoreRadii[pc] \t\t\t  FitAlpha"
        "\t\t\t FitSigma0 \t\t\t  FitSigmaBg"
    )
    np.savetxt(fname=save_name + "info.txt", X=output, header=header)

    return gc_tot_masses, gc_r_core, gc_m_core, gc_r_trunc, gc_char_age, time


#%%

if __name__ == "__main__":

    # data_file = "./pop_2_data/pos_00418_462_09_myr.txt"
    data_file = "./pop_2_data/pos_00486_476_23_myr.txt"
    # # put all verbose output into a text file
    folder_name = "./gc_profiles/snapshot_" + data_file[18:-8]
    if not os.path.exists(folder_name):
        print("# Creating new sequence directory", folder_name)
        os.makedirs(folder_name)

    orig_stdout = sys.stdout
    sys.stdout = open(folder_name + "/log.txt", "w")

    masses, core_radii, core_masses, r_trunc, ages, time = run_profiler(
        data_file,
        400,
        10,
        1000,
        )

    core_diameter = core_radii * 2

    # plt.figure(figsize = (8,8), dpi=200)
    # plt.hist(core_masses, bins=np.geomspace(core_masses.min(), core_masses.max(),10), histtype='step',  fill=False)
    # plt.xscale('log')

    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html

    colors = np.random.uniform(size=masses.size)
    biggest_gc = np.max(core_diameter)
    # map to differnt sizes for better plotting
    core_diameter_per_size = (500 * core_diameter) / biggest_gc

    # fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

    # scatter = ax.scatter(
    #     ages,
    #     masses,
    #     c=colors,
    #     s=core_diameter_per_size,
    #     cmap="Set3",
    #     alpha=0.6,
    #     linewidths=2,
    #     )

    # # remap to actual sizes for legend
    # legend_properties = dict(
    #     prop="sizes",
    #     num=4,
    #     color="white",
    #     fmt=" {x:.2f}",
    #     func=lambda r: (r * biggest_gc) / 500,
    #     )
    # legend = ax.legend(
    #     *scatter.legend_elements(**legend_properties),
    #     loc="upper right",
    #     title="$d_{core}$ (pc)",
    #     title_fontsize=16,
    #     fontsize=15,
    #     )

    # ax.set_yscale("log")
    # ax.set_title(r"$t_{{sim}} = {} Myr$".format(time))
    # ax.set_ylabel(r"Total GC Mass ($M_{\odot}$)", fontsize=16)
    # ax.set_xlabel(r"Age (Myr)", fontsize=16)
    # fig.savefig(folder_name + "/scatter.png", dpi=300)
    # # fig.close()

    sys.stdout.close()
    sys.stdout = orig_stdout
