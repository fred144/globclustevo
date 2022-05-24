import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from ..macros import characterisitc_mass
from .profile_functions import get_cluster, projected_surf_densities
from .profile_models import modified_king_model, trunc_radius
from scipy.optimize import curve_fit
import scipy.stats as st


def king_profile_plotter(
    star_pos,
    lums,
    masses,
    ages,
    gc_ctr,
    gc_rad,
    gc_label,
    bins=30,
    good_alpha=5,
    **kwargs
):
    """
    Contructs one projected density profile for a globular cluster within a
    a frame and fits it. Need to loop over all the found GCs.
    Depends on projected_surf_densities.
    Returns info about one GC if fit is good.
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
        (
            clust_x,
            clust_y,
            clust_z,
            clust_lums,
            clust_masses,
            clust_ages,
        ) = get_cluster(
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

    # numpy way of doing mode
    # unique_ages, indices = np.unique(gc_char_age, return_inverse=True)
    # index_of_most_common_age = np.argmax(np.bincount(indices))
    # most_common_age = unique_ages[index_of_most_common_age]

    gc_char_age = float(st.mode(clust_ages)[0] / 1e6)

    plt.close()

    # plot the data with error bars
    # plt.figure(figsize=(8, 8), dpi=200)

    try:
        # try to do the fit
        try:
            fit_params, cov_matrix = curve_fit(
                f=modified_king_model,
                xdata=r,
                ydata=rho,
                sigma=err,
                absolute_sigma=True,
                p0=[1e4, 0.2, 2, 10],
                bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
            )
        except Exception as e:
            print(e)
            # try to do the fit with cleaned up data
            print(r"> Trying another fit for GC #{:.0f}".format(gc_label))
            fit_params, cov_matrix = curve_fit(
                f=modified_king_model,
                xdata=r[:-5:2],
                ydata=rho[:-5:2],
                sigma=err[:-5:2],
                absolute_sigma=True,
                p0=[1e4, 0.2, 2, 10],
                bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
            )

        # params: sigma_naught, r_c, alpha, bg
        # fit_err = np.sqrt(np.diag(cov_matrix))

        err_fit_sigma_naught, err_fit_r_c, err_fit_alpha, err_fit_sigma_bg = np.sqrt(
            np.diag(cov_matrix)
        )
        # print(fit_params)
        # print(fit_sigma)

        # calculate theoretical best fit
        theory_r = np.geomspace(r[0], gc_rad, 200, endpoint=False)  # smooth version
        # theory_r = np.linspace(r[0], gc_rad, 2000)
        theory_rho = modified_king_model(theory_r, *fit_params)

        # ===========================calc derived quantities===================
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
        core_mass, core_count = characterisitc_mass(
            clust_x, clust_y, clust_masses, fit_r_c, counts=True
        )

        # =============================================================================

        # quantify goodness of fit
        # p_value, reduced_chi_2 = chi_squared(
        #     theory=theory_rho, data=rho, sigma=err, num_params=4
        #     )

        if truncation_radius <= gc_rad:
            # realistic truncation radius labels
            trunc_mass, trunc_count = characterisitc_mass(
                clust_x, clust_y, clust_masses, truncation_radius, counts=True
            )
            output_mass = trunc_mass
            plot_label = (
                r"$R_{{core}} = {:.2f} \: pc$"
                "\n"
                r"$R_{{trunc}} = {:.2f} \: pc$"
                "\n"
                r"$\alpha = {:.2f} $"
                "\n"
                r"$\Sigma_0 = {:.2e} $"
                "\n"
                r"$\Sigma_{{bg}} = {:.2f} $"
                "\n"
                r"$M_{{core}} = {:.2e} \: M_{{\odot}}$"
                "\n"
                r"$M_{{trunc}} = {:.2e} \: M_{{\odot}}$"
            ).format(
                fit_r_c,
                truncation_radius,
                fit_alpha,
                fit_sigma_naught,
                fit_sigma_bg,
                core_mass,
                trunc_mass,
            )

        else:
            # if the truncation radius is a mathematical artifact, use this label
            output_mass = tot_m
            plot_label = (
                r"$R_{{core}} = {:.2f}$ pc"
                "\n"
                r"$R_{{trunc}} > {:.0f} $ pc"  # def
                "\n"
                r"$\alpha = {:.2f} $"
                "\n"
                r"$\Sigma_0 = {:.2e} $"
                "\n"
                r"$\Sigma_{{bg}} = {:.2f} $"
                "\n"
                r"$M_{{core}} = {:.2e} \: M_{{\odot}}$"
                "\n"
                r"$M_{{(r < {:.0f} pc)}} = {:.2e} \: M_{{\odot}}$"
            ).format(
                fit_r_c,
                gc_rad,
                fit_alpha,
                fit_sigma_naught,
                fit_sigma_bg,
                core_mass,  ########
                gc_rad,
                tot_m,
            )

        # plot the fit if it is good
        if (
            fit_alpha < good_alpha
            and core_mass > 1
            and np.max(rho) > 0
            and np.max(r) > 0
        ):

            plt.figure(figsize=(8, 8), dpi=200)
            plt.errorbar(
                r,
                rho,
                yerr=err,
                fmt="o",
                capsize=5,
                capthick=3,
                elinewidth=3,
                label=(r"$t_{{age}}= {:.2f}$ Myr").format(gc_char_age),
                zorder=1,
            )
            # plot curve fit
            plt.plot(
                theory_r,
                theory_rho,
                ls="--",
                color="tab:red",
                linewidth=4,
                label=plot_label,
                zorder=20,
            )
            plt.axvline(**kwargs)
            # plt.text(fit_r_c, 0,r'$R_core$',rotation=90)
            plt.title(r"GC # {:.0f}".format(gc_label), fontsize=16)
            plt.ylabel(r"Surface Mass Density ($M_{\odot} \; pc^{-2}$)", fontsize=16)
            plt.xlabel(r"R ($pc$)", fontsize=16)
            plt.xscale("log")
            plt.yscale("log")
            # plt.xscale('symlog')
            # plt.yscale('symlog')
            plt.grid(visible=True, which="both", axis="y", ls="--")
            plt.legend(fontsize=16)

            return (
                r,
                rho,
                err,
                output_mass,
                fit_r_c,
                err_fit_r_c,
                core_mass,
                truncation_radius,
                gc_char_age,
                fit_alpha,
                err_fit_alpha,
                fit_sigma_naught,
                err_fit_sigma_naught,
                fit_sigma_bg,
                err_fit_sigma_bg,
                np.size(clust_x),
            )
        else:
            print(r"> bad alpha for GC #{:.0f}".format(gc_label))
            # return invalid values
            return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

    # except BaseException:
    except Exception as e:
        print(e)
        # if it can't fit it
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
            zorder=1,
        )

        plt.title(r"No Fit GC # {:.0f}".format(gc_label), fontsize=16)
        plt.ylabel(r"Surface Mass Density ($M_{\odot} \; pc^{-2}$)", fontsize=16)
        plt.xlabel(r"R ($pc$)", fontsize=16)
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(visible=True, which="both", axis="y", ls="--")
        plt.legend(fontsize=16)

        print(r"> can't fit GC #{:.0f} ".format(gc_label))

        # return invalid values
        return -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2


def king_profile_plotter_lite(
    gc_stars, lums, gc_rad, gc_label, bins=30, good_alpha=5, **kwargs
):
    """
    kig profile given the coordinates (recentered) of the stars within a cluster.
    made for profiling stars within a halo finder's derived virial radius that
    have already been separated out.
    """
    clust_x = gc_stars[:, 0]
    clust_y = gc_stars[:, 1]

    clust_masses = np.ones(np.size * (clust_x)) * 10  # solar masses
    # given an isolated cluster, find projected density quantities
    # as a function of radius with log bins or linear bins
    r, rho, err, tot_m = projected_surf_densities(
        x_coord=clust_x,
        y_coord=clust_y,
        lums=0,
        masses=clust_masses,
        radius=gc_rad,
        num_bins=bins,
        log_bins=True,
        dr=None,
    )

    # gc_char_age = float(st.mode(clust_ages)[0] / 1e6)

    plt.close()

    # plot the data with error bars
    # plt.figure(figsize=(8, 8), dpi=200)

    try:
        # try to do the fit
        try:
            fit_params, cov_matrix = curve_fit(
                f=modified_king_model,
                xdata=r,
                ydata=rho,
                sigma=err,
                absolute_sigma=True,
                p0=[1e4, 0.2, 2, 10],
                bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
            )
        except Exception as e:
            print(e)
            # try to do the fit with cleaned up data
            print(r"> Trying another fit for GC #{:.0f}".format(gc_label))
            fit_params, cov_matrix = curve_fit(
                f=modified_king_model,
                xdata=r[:-5:2],
                ydata=rho[:-5:2],
                sigma=err[:-5:2],
                absolute_sigma=True,
                p0=[1e4, 0.2, 2, 10],
                bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
            )

        # params: sigma_naught, r_c, alpha, bg
        # fit_err = np.sqrt(np.diag(cov_matrix))

        err_fit_sigma_naught, err_fit_r_c, err_fit_alpha, err_fit_sigma_bg = np.sqrt(
            np.diag(cov_matrix)
        )
        # print(fit_params)
        # print(fit_sigma)

        # calculate theoretical best fit
        theory_r = np.geomspace(r[0], gc_rad, 200, endpoint=False)  # smooth version
        # theory_r = np.linspace(r[0], gc_rad, 2000)
        theory_rho = modified_king_model(theory_r, *fit_params)

        # ===========================calc derived quantities===================
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
        core_mass, core_count = characterisitc_mass(
            clust_x, clust_y, clust_masses, fit_r_c, counts=True
        )

        # =============================================================================

        # quantify goodness of fit
        # p_value, reduced_chi_2 = chi_squared(
        #     theory=theory_rho, data=rho, sigma=err, num_params=4
        #     )

        if truncation_radius <= gc_rad:
            # realistic truncation radius labels
            trunc_mass, trunc_count = characterisitc_mass(
                clust_x, clust_y, clust_masses, truncation_radius, counts=True
            )
            output_mass = trunc_mass
            plot_label = (
                r"$R_{{core}} = {:.2f} \: pc$"
                "\n"
                r"$R_{{trunc}} = {:.2f} \: pc$"
                "\n"
                r"$\alpha = {:.2f} $"
                "\n"
                r"$\Sigma_0 = {:.2e} $"
                "\n"
                r"$\Sigma_{{bg}} = {:.2f} $"
                "\n"
                r"$M_{{core}} = {:.2e} \: M_{{\odot}}$"
                "\n"
                r"$M_{{trunc}} = {:.2e} \: M_{{\odot}}$"
            ).format(
                fit_r_c,
                truncation_radius,
                fit_alpha,
                fit_sigma_naught,
                fit_sigma_bg,
                core_mass,
                trunc_mass,
            )

        else:
            # if the truncation radius is a mathematical artifact, use this label
            output_mass = tot_m
            plot_label = (
                r"$R_{{core}} = {:.2f}$ pc"
                "\n"
                r"$R_{{trunc}} > {:.0f} $ pc"  # def
                "\n"
                r"$\alpha = {:.2f} $"
                "\n"
                r"$\Sigma_0 = {:.2e} $"
                "\n"
                r"$\Sigma_{{bg}} = {:.2f} $"
                "\n"
                r"$M_{{core}} = {:.2e} \: M_{{\odot}}$"
                "\n"
                r"$M_{{(r < {:.0f} pc)}} = {:.2e} \: M_{{\odot}}$"
            ).format(
                fit_r_c,
                gc_rad,
                fit_alpha,
                fit_sigma_naught,
                fit_sigma_bg,
                core_mass,  ########
                gc_rad,
                tot_m,
            )

        # plot the fit if it is good
        if (
            fit_alpha < good_alpha
            and core_mass > 1
            and np.max(rho) > 0
            and np.max(r) > 0
        ):

            plt.figure(figsize=(8, 8), dpi=200)
            plt.errorbar(
                r,
                rho,
                yerr=err,
                fmt="o",
                capsize=5,
                capthick=3,
                elinewidth=3,
                # label=(r"$t_{{age}}= {:.2f}$ Myr").format(gc_char_age),
                zorder=1,
            )
            # plot curve fit
            plt.plot(
                theory_r,
                theory_rho,
                ls="--",
                color="tab:red",
                linewidth=4,
                label=plot_label,
                zorder=20,
            )

            # plt.text(fit_r_c, 0,r'$R_core$',rotation=90)
            plt.title(r"GC # {:.0f}".format(gc_label), fontsize=16)
            plt.ylabel(r"Surface Mass Density ($M_{\odot} \; pc^{-2}$)", fontsize=16)
            plt.xlabel(r"R ($pc$)", fontsize=16)
            plt.xscale("log")
            plt.yscale("log")
            # plt.xscale('symlog')
            # plt.yscale('symlog')
            plt.grid(visible=True, which="both", axis="y", ls="--")
            plt.legend(fontsize=16)

            return (
                r,
                rho,
                err,
                output_mass,
                fit_r_c,
                err_fit_r_c,
                core_mass,
                truncation_radius,
                np.nan,  # gc_char_age
                fit_alpha,
                err_fit_alpha,
                fit_sigma_naught,
                err_fit_sigma_naught,
                fit_sigma_bg,
                err_fit_sigma_bg,
                np.size(clust_x),
            )
        else:
            print(r"> bad alpha for GC #{:.0f}".format(gc_label))
            # return invalid values
            return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

    # except BaseException:
    except Exception as e:
        print(e)
        # if it can't fit it
        plt.figure(figsize=(8, 8), dpi=200)

        plt.errorbar(
            r,
            rho,
            yerr=err,
            fmt="o",
            capsize=5,
            capthick=3,
            elinewidth=3,
            # label=(
            #     r"$M_{{total}}= {:.2e} \: M_{{\odot}}$"
            #     "\n"
            #     r"$t_{{age}}= {:.2f}$ Myr"
            # ).format(tot_m, gc_char_age),
            zorder=1,
        )

        plt.title(r"No Fit GC # {:.0f}".format(gc_label), fontsize=16)
        plt.ylabel(r"Surface Mass Density ($M_{\odot} \; pc^{-2}$)", fontsize=16)
        plt.xlabel(r"R ($pc$)", fontsize=16)
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(visible=True, which="both", axis="y", ls="--")
        plt.legend(fontsize=16)

        print(r"> can't fit GC #{:.0f} ".format(gc_label))

        # return invalid values
        return -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2
