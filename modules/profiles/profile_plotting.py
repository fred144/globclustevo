import sys

sys.path.append("..")
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ..macros import characterisitc_mass, chi_squared
from .profile_functions import get_cluster, projected_surf_densities
from .profile_models import modified_king_model, trunc_radius
from scipy.optimize import curve_fit
import scipy.stats as st


def sci_notation(decimal_places, exp_string):
    """
    turns '1.00e+06' into scientific notation with times and power formatter
    for LaTeX
    """
    start = decimal_places + 2
    end = decimal_places + 3
    return r"{}\;\times\;10^{:1} ".format(exp_string[:start], int(exp_string[end:]))


def master_king(rads, rhos, errs, ages, t_myr):
    """
    gets all the raw profiles for a given snapshot and plots them, note
    both theory and fitted curves are taken
    """
    print("> making master plot")
    ages = np.array(ages)
    birth_times = t_myr - ages
    with plt.rc_context(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.size": 12,
        }
    ):
        # initialize figure
        fig, (ax, cbar_ax) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(4, 4),
            dpi=400,
            gridspec_kw={"width_ratios": [20, 1]},
        )

        # make subplot
        plt.subplots_adjust(wspace=0)
        cmap = plt.cm.winter
        # extend the range of color bars
        norm = plt.Normalize(
            vmin=np.min(birth_times) - 1, vmax=np.max(birth_times) + 1
        )
        # just integers
        kwargs = {"format": "%.0f"}
        cbar = mpl.colorbar.ColorbarBase(
            cbar_ax,
            cmap=cmap,
            norm=norm,
            label=r"$\: \mathrm{t_{birth}}  \: \mathrm{(Myr)}$",
            alpha=0.6,
            # **kwargs,
        )
        # loop and plot
        for i, (r, rho, err, bt) in enumerate(zip(rads, rhos, errs, birth_times)):

            ax.plot(
                r,
                rho,
                color=cmap(norm(bt)),
                linewidth=2,
                alpha=0.6,
                # zorder=20,
            )
            # ax.fill_between(
            #     r,
            #     y1=rho - err,
            #     y2=rho + err,
            #     facecolor=cmap(norm(age)),
            #     alpha=0.5,
            #     linewidth=0.0,
            #     interpolate=True,
            #     zorder=1,
            # )

        # current simulation time annotation
        props = dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.5,
            linewidth=0.8,
            edgecolor="gray",
        )
        textstr = r"$\mathrm{{t}} = {:.1f} \: \mathrm{{Myr}}$".format(t_myr)
        # place a text box in upper left in axes coords
        ax.text(
            0.62,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )
        # indicate max precision
        ax.axvspan(ax.get_xlim()[0], 0.15, facecolor="grey", alpha=0.5, zorder=0)
        ax.set_xlabel(r"$ \mathrm{R} (\mathrm{pc})$")
        ax.set_ylabel(
            r"$ \mathrm{\Sigma} \: (\mathrm{M}_{\odot} \; \mathrm{pc}^{-2})$"
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(right=20)
        ax.set_ylim(bottom=3e-1, top=1e5)


# TODO: add filter to avoid small double count by filtering clust_x, y,z using fof
def king_profile_plotter(
    star_pos,
    lums,
    masses,
    ages,
    gc_ctr,
    gc_rad,
    gc_label,
    bins=25,
    good_alpha=8,
    particle_filter=None,
    using_vir_rad=True,
    **kwargs,
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

    if particle_filter is not None:
        mask = np.isin(clust_x, particle_filter)
        clust_x = clust_x[mask]
        clust_y = clust_y[mask]
        clust_z = clust_z[mask]
        clust_lums = clust_lums[mask]
        clust_masses = clust_masses[mask]
        clust_ages = clust_ages[mask]
    else:
        pass

    # try:
    # given an isolated cluster, find projected density quantities
    # as a function of radius with log bins or linear bins
    r, rho, err, tot_m, tot_lum, r_half_m, r_half_l = projected_surf_densities(
        x_coord=clust_x,
        y_coord=clust_y,
        lums=clust_lums,
        masses=clust_masses,
        radius=gc_rad,
        num_bins=bins,
        log_bins=True,
        dr=None,
        calc_half_r=True,
    )
    # except:
    #     print("> not enough stars to profile")
    #     return (
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #         -2,
    #     )

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
            print("* ", e)
            # try to do the fit with cleaned up data
            print(r"> trying another fit for GC #{:.0f}".format(gc_label))
            # cuts tail off and scales down sampling
            fit_params, cov_matrix = curve_fit(
                f=modified_king_model,
                xdata=r[:-5:1],
                ydata=rho[:-5:1],
                sigma=err[:-5:1],
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
        if using_vir_rad is True:
            output_mass = tot_m  # tot_m is the mass within the vir radius
            if fit_sigma_bg > 1:
                plot_label = (
                    r"$R_{{core}} = {:.2f} \: \mathrm{{pc}}$"
                    "\n"
                    r"$\alpha = {:.2f} $"
                    "\n"
                    r"$\Sigma_0 = {} (\mathrm{{M}}_{{\odot}} \; \mathrm{{pc}}^{{-2}})$"
                    "\n"
                    r"$\Sigma_{{bg}} = {:.2f} (\mathrm{{M}}_{{\odot}} \; \mathrm{{pc}}^{{-2}})$"
                    "\n"
                    r"$M_{{core}} = {} \: \mathrm{{M}}_{{\odot}}$"
                    "\n"
                    r"$M_{{(R < {:.0f} \:  pc)}} = {} \: \mathrm{{M}}_{{\odot}}$"
                ).format(
                    fit_r_c,
                    fit_alpha,
                    sci_notation(2, "{:.2e}".format(fit_sigma_naught)),
                    fit_sigma_bg,
                    sci_notation(2, "{:.2e}".format(core_mass)),
                    gc_rad,
                    sci_notation(2, "{:.2e}".format(tot_m)),
                )
            else:
                plot_label = (
                    r"$R_{{core}} = {:.2f} \: \mathrm{{pc}}$"
                    "\n"
                    r"$\alpha = {:.2f} $"
                    "\n"
                    r"$\Sigma_0 = {} (\mathrm{{M}}_{{\odot}} \; \mathrm{{pc}}^{{-2}})$"
                    "\n"
                    r"$M_{{core}} = {} \: \mathrm{{M}}_{{\odot}}$"
                    "\n"
                    r"$M_{{(R < {:.0f} \:  pc)}} = {} \: \mathrm{{M}}_{{\odot}}$"
                ).format(
                    fit_r_c,
                    fit_alpha,
                    sci_notation(2, "{:.2e}".format(fit_sigma_naught)),
                    sci_notation(2, "{:.2e}".format(core_mass)),
                    gc_rad,
                    sci_notation(2, "{:.2e}".format(tot_m)),
                )
        elif truncation_radius <= gc_rad:
            # realistic truncation radius labels
            trunc_mass, trunc_count = characterisitc_mass(
                clust_x, clust_y, clust_masses, truncation_radius, counts=True
            )
            output_mass = trunc_mass
            plot_label = (
                r"$R_{{core}} = {:.2f} \: \mathrm{{pc}}$"
                "\n"
                r"$R_{{trunc}} = {:.2f} \: \mathrm{{pc}}$"
                "\n"
                r"$\alpha = {:.2f} $"
                "\n"
                r"$\Sigma_0 = {:.2e} \; (\mathrm{{M}}_{{\odot}} \; \mathrm{{pc}}^{{-2}})$"
                "\n"
                r"$\Sigma_{{bg}} = {:.2f} \; (\mathrm{{M}}_{{\odot}} \; \mathrm{{pc}}^{{-2}})$"
                "\n"
                r"$M_{{core}} = {:.2e} \: \mathrm{{M}}_{{\odot}}$"
                "\n"
                r"$M_{{trunc}} = {:.2e} \: \mathrm{{M}}_{{\odot}}$"
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
                r"$\Sigma_0 = {:.2e} \; (\mathrm{{M}}_{{\odot}} \; \mathrm{{pc}}^{{-2}})$"
                "\n"
                r"$\Sigma_{{bg}} = {:.2f} \; (\mathrm{{M}}_{{\odot}} \; \mathrm{{pc}}^{{-2}})$"
                "\n"
                r"$M_{{core}} = {:.2e} \: M_{{\odot}}$"
                "\n"
                r"$M_{{(r < {:.0f} \:pc)}} = {:.2e} \: M_{{\odot}}$"
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
            with plt.rc_context(
                {
                    "font.family": "serif",
                    "mathtext.fontset": "cm",
                    "xtick.labelsize": 12,
                    "ytick.labelsize": 12,
                    "font.size": 12,
                }
            ):
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=400)

                ax.errorbar(
                    r,
                    rho,
                    yerr=err,
                    c="mediumpurple",
                    fmt="o",
                    capsize=5,
                    capthick=3,
                    elinewidth=3,
                    label=(
                        r"$t_{{age}}\:{} \:\mathrm{{stars}} : {:.2f}\:\mathrm{{Myr}}$"
                    ).format(np.size(clust_masses), gc_char_age),
                    zorder=1,
                )
                # plot curve fit
                ax.plot(
                    theory_r,
                    theory_rho,
                    color="darkorange",
                    linewidth=4,
                    label=plot_label,
                    alpha=0.9,
                )

                # plt.axvline(**kwargs) #!!!
                # plt.text(fit_r_c, 0,r'$R_core$',rotation=90)

                ax.set_title(
                    r"GC # {:.0f}".format(
                        gc_label,
                    ),
                )
                p_value, reduced_chi_2 = chi_squared(
                    modified_king_model(r, *fit_params), rho, err, num_params=4
                )
                fit_str = (
                    r"$\chi^2_{{\nu}} = {:.1f}"
                    r",\:\mathrm{{P}}\left(\chi^2,\nu \right) = {:.2f}$"
                ).format(reduced_chi_2, p_value)
                props = dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.8,
                    linewidth=0.8,
                    edgecolor="gray",
                )

                #  text box in upper left in axes coords for the goodness of fit
                ax.text(
                    0.56,
                    0.96,
                    fit_str,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=props,
                    fontsize=8,
                )
                ax.set_ylabel(
                    r" $ \mathrm{\Sigma} \: (\mathrm{M}_{\odot} \; \mathrm{pc}^{-2})$"
                )
                ax.set_xlabel(r"$ \mathrm{R} \: (\mathrm{pc})$")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(visible=True, which="both", axis="y", ls="--")
                ax.legend(loc="lower left", fontsize=8)
            # print(r"> fitted GC #{:.0f}".format(gc_label))
            # !!!
            return (
                (r, theory_r),
                (rho, theory_rho),
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
                p_value,
                np.size(clust_masses),  # size of the masses array, count star in gc
                r_half_m,
                r_half_l,
                tot_lum,
            )

        else:
            print(r"> bad alpha: {:.5f} for GC #{:.0f}".format(fit_alpha, gc_label))
            # return invalid values
            return (
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            )

    # except BaseException:
    except Exception as e:
        print("* might be masked out to emptiness", e)
        print(r"> can't fit GC #{:.0f} ".format(gc_label))
        # if it can't fit it
        plt.figure(figsize=(4, 4), dpi=200)

        plt.errorbar(
            r,
            rho,
            yerr=err,
            fmt="o",
            capsize=5,
            capthick=3,
            elinewidth=3,
            label=(
                r"$M_{{total}}= {:.2e} \: \mathrm{{M}}_{{\odot}}$"
                "\n"
                r"$t_{{age}}= {:.2f}$ Myr"
            ).format(tot_m, gc_char_age),
            zorder=1,
        )

        plt.title(r"No Fit GC # {:.0f}".format(gc_label), fontsize=12)
        plt.ylabel(
            r"Surface Mass Density ($M_{\odot} \;\mathrm{pc}^{-2}$)", fontsize=12
        )
        plt.xlabel(r"R ($\mathrm{pc}$)", fontsize=12)
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(visible=True, which="both", axis="y", ls="--")
        plt.legend(fontsize=12)

        # return invalid values
        return (
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
        )


# def king_profile_plotter_lite(
#     gc_stars, lums, gc_rad, gc_label, bins=30, good_alpha=5, **kwargs
# ):
#     """
#     kig profile given the coordinates (recentered) of the stars within a cluster.
#     made for profiling stars within a halo finder's derived virial radius that
#     have already been separated out.
#     """
#     clust_x = gc_stars[:, 0]
#     clust_y = gc_stars[:, 1]

#     clust_masses = np.ones(np.size * (clust_x)) * 10  # solar masses
#     # given an isolated cluster, find projected density quantities
#     # as a function of radius with log bins or linear bins
#     r, rho, err, tot_m, r_half_mass, r_half_light = projected_surf_densities(
#         x_coord=clust_x,
#         y_coord=clust_y,
#         lums=0,
#         masses=clust_masses,
#         radius=gc_rad,
#         num_bins=bins,
#         log_bins=True,
#         dr=None,
#         calc_half_r=True,
#     )

#     # gc_char_age = float(st.mode(clust_ages)[0] / 1e6)

#     plt.close()

#     # plot the data with error bars
#     # plt.figure(figsize=(8, 8), dpi=200)

#     try:
#         # try to do the fit
#         try:
#             fit_params, cov_matrix = curve_fit(
#                 f=modified_king_model,
#                 xdata=r,
#                 ydata=rho,
#                 sigma=err,
#                 absolute_sigma=True,
#                 p0=[1e4, 0.2, 2, 10],
#                 bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
#             )
#         except Exception as e:
#             print(e)
#             # try to do the fit with cleaned up data
#             print(r"> Trying another fit for GC #{:.0f}".format(gc_label))
#             fit_params, cov_matrix = curve_fit(
#                 f=modified_king_model,
#                 xdata=r[:-5:2],
#                 ydata=rho[:-5:2],
#                 sigma=err[:-5:2],
#                 absolute_sigma=True,
#                 p0=[1e4, 0.2, 2, 10],
#                 bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
#             )

#         # params: sigma_naught, r_c, alpha, bg
#         # fit_err = np.sqrt(np.diag(cov_matrix))

#         err_fit_sigma_naught, err_fit_r_c, err_fit_alpha, err_fit_sigma_bg = np.sqrt(
#             np.diag(cov_matrix)
#         )
#         # print(fit_params)
#         # print(fit_sigma)

#         # calculate theoretical best fit
#         theory_r = np.geomspace(r[0], gc_rad, 200, endpoint=False)  # smooth version
#         # theory_r = np.linspace(r[0], gc_rad, 2000)
#         theory_rho = modified_king_model(theory_r, *fit_params)

#         # ===========================calc derived quantities===================
#         fit_sigma_naught = fit_params[0]
#         fit_r_c = fit_params[1]
#         fit_alpha = fit_params[2]
#         fit_sigma_bg = fit_params[3]
#         truncation_radius = trunc_radius(
#             r_c=fit_r_c,
#             alpha=fit_alpha,
#             sigma_0=fit_sigma_naught,
#             sigma_bg=fit_sigma_bg,
#         )
#         core_mass, core_count = characterisitc_mass(
#             clust_x, clust_y, clust_masses, fit_r_c, counts=True
#         )

#         # =============================================================================

#         # quantify goodness of fit
#         # p_value, reduced_chi_2 = chi_squared(
#         #     theory=theory_rho, data=rho, sigma=err, num_params=4
#         #     )

#         if truncation_radius <= gc_rad:
#             # realistic truncation radius labels
#             trunc_mass, trunc_count = characterisitc_mass(
#                 clust_x, clust_y, clust_masses, truncation_radius, counts=True
#             )
#             output_mass = trunc_mass
#             plot_label = (
#                 r"$R_{{core}} = {:.2f} \: pc$"
#                 "\n"
#                 r"$R_{{trunc}} = {:.2f} \: pc$"
#                 "\n"
#                 r"$\alpha = {:.2f} $"
#                 "\n"
#                 r"$\Sigma_0 = {:.2e} $"
#                 "\n"
#                 r"$\Sigma_{{bg}} = {:.2f} $"
#                 "\n"
#                 r"$M_{{core}} = {:.2e} \: M_{{\odot}}$"
#                 "\n"
#                 r"$M_{{trunc}} = {:.2e} \: M_{{\odot}}$"
#             ).format(
#                 fit_r_c,
#                 truncation_radius,
#                 fit_alpha,
#                 fit_sigma_naught,
#                 fit_sigma_bg,
#                 core_mass,
#                 trunc_mass,
#             )

#         else:
#             # if the truncation radius is a mathematical artifact, use this label
#             output_mass = tot_m
#             plot_label = (
#                 r"$R_{{core}} = {:.2f}$ pc"
#                 "\n"
#                 r"$R_{{trunc}} > {:.0f} $ pc"  # def
#                 "\n"
#                 r"$\alpha = {:.2f} $"
#                 "\n"
#                 r"$\Sigma_0 = {:.2e} $"
#                 "\n"
#                 r"$\Sigma_{{bg}} = {:.2f} $"
#                 "\n"
#                 r"$M_{{core}} = {:.2e} \: M_{{\odot}}$"
#                 "\n"
#                 r"$M_{{(r < {:.0f} pc)}} = {:.2e} \: M_{{\odot}}$"
#             ).format(
#                 fit_r_c,
#                 gc_rad,
#                 fit_alpha,
#                 fit_sigma_naught,
#                 fit_sigma_bg,
#                 core_mass,  ########
#                 gc_rad,
#                 tot_m,
#             )

#         # plot the fit if it is good
#         if (
#             fit_alpha < good_alpha
#             and core_mass > 0
#             and np.max(rho) > 0
#             and np.max(r) > 0
#         ):

#             plt.figure(figsize=(8, 8), dpi=200)
#             plt.errorbar(
#                 r,
#                 rho,
#                 yerr=err,
#                 fmt="o",
#                 capsize=5,
#                 capthick=3,
#                 elinewidth=3,
#                 # label=(r"$t_{{age}}= {:.2f}$ Myr").format(gc_char_age),
#                 zorder=1,
#             )
#             # plot curve fit
#             plt.plot(
#                 theory_r,
#                 theory_rho,
#                 ls="--",
#                 color="tab:red",
#                 linewidth=4,
#                 label=plot_label,
#                 zorder=20,
#             )

#             # plt.text(fit_r_c, 0,r'$R_core$',rotation=90)
#             plt.title(r"GC # {:.0f}".format(gc_label), fontsize=16)
#             plt.ylabel(r"Surface Mass Density ($M_{\odot} \; pc^{-2}$)", fontsize=16)
#             plt.xlabel(r"R ($pc$)", fontsize=16)
#             plt.xscale("log")
#             plt.yscale("log")
#             # plt.xscale('symlog')
#             # plt.yscale('symlog')
#             plt.grid(visible=True, which="both", axis="y", ls="--")
#             plt.legend(fontsize=16)

#             return (
#                 r,
#                 rho,
#                 err,
#                 output_mass,
#                 fit_r_c,
#                 err_fit_r_c,
#                 core_mass,
#                 truncation_radius,
#                 np.nan,  # gc_char_age
#                 fit_alpha,
#                 err_fit_alpha,
#                 fit_sigma_naught,
#                 err_fit_sigma_naught,
#                 fit_sigma_bg,
#                 err_fit_sigma_bg,
#                 np.size(clust_x),
#             )
#         else:
#             print(r"> bad alpha for GC #{:.0f}".format(gc_label))
#             # return invalid values
#             return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

#     # except BaseException:
#     except Exception as e:
#         print(e)
#         # if it can't fit it
#         plt.figure(figsize=(8, 8), dpi=200)

#         plt.errorbar(
#             r,
#             rho,
#             yerr=err,
#             fmt="o",
#             capsize=5,
#             capthick=3,
#             elinewidth=3,
#             # label=(
#             #     r"$M_{{total}}= {:.2e} \: M_{{\odot}}$"
#             #     "\n"
#             #     r"$t_{{age}}= {:.2f}$ Myr"
#             # ).format(tot_m, gc_char_age),
#             zorder=1,
#         )

#         plt.title(r"No Fit GC # {:.0f}".format(gc_label), fontsize=16)
#         plt.ylabel(r"Surface Mass Density ($M_{\odot} \; pc^{-2}$)", fontsize=16)
#         plt.xlabel(r"R ($pc$)", fontsize=16)
#         plt.xscale("log")
#         plt.yscale("log")
#         plt.grid(visible=True, which="both", axis="y", ls="--")
#         plt.legend(fontsize=16)

#         print(r"> can't fit GC #{:.0f} ".format(gc_label))

#         # return invalid values
#         return -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2
