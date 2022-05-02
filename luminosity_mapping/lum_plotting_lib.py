import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from pytreegrav import Potential
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
import matplotlib as mpl
import scipy.stats as st
from lum_funcs import (
    chi_squared,
    get_cluster,
    get_masses,
    projected_surf_densities,
)

# unique_ages, indices = np.unique(gc_char_age, return_inverse=True)
# index_of_most_common_age = np.argmax(np.bincount(indices))
# most_common_age = unique_ages[index_of_most_common_age]


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
    """
    king model with an additional fitting parameter
    """
    sigma = bg + (sigma_naught / (1 + (r / r_c) ** alpha))
    return sigma


def trunc_radius(sigma_0, r_c, alpha, sigma_bg):
    """
    set to 1.5bg =  bg + (peak)/( 1 + (r/r_c)^alpha)
    0.5bg = (peak)/( 1 + (r/r_c)^alpha)
    """

    r_trunc = (r_c**alpha * ((sigma_0 / ((1.5 - 1) * sigma_bg) - 1))) ** (1 / alpha)
    return r_trunc


def star_luminosity_plot(
    proj_width,
    star_positions,
    scaled_stellar_lums,
    time,
    snapshot_num,
    pi_multiple,
    plt_bins=2000,
    lum_scale=("static", 3e32, 3e36),
    get_ctr=(True, "potential", 0.01, False),
    num_ctr=100,
    ctr_dist_thresh=10,
    ctr_rel_thresh=0.0001,
    masses=None,
    plt_ctrs=None,
):
    """
    Main single panel movie for luminosities of popII stars.
    Currently makes projection plot along projected on an axis.
    Can return and/or annotate local maxima corresponding to centers of clusters.

    Parameters
    ----------
    proj_width
        width of path in pc
    star_positions
       postions of stars, (x,y)
    scaled_stellar_lums
        scaled star luminosities
    time
        current time of snapshot in myr
    snapshot_num
        snapshot number for simulation
    pi_multiple
        pi multiple of the rotation matrix, can be 0
    plt_bins
        number of bins along the axis of the luminosity map, not for center finding
    lum_scale: tuple (log_scale_type, min, max)
        log_scale_type - static or dynamic

        min - minimum value color scale

        max- maximum value color scale
    get_ctr: tuple  (get_ctr, method, resolution, overplot)
        get_ctr - True or False, gets centers of GCs

        method - either using raw "counts" of star in a bin or grav "potential"

        resolution - produces bins for center finding using width/resolution in pc

        overplot - True or False, overplots found centers
    num_ctr: int
        maximum number of GC centers to find
    ctr_dist_thresh: float
        the minimum (projected) distance in pc that each center should be
    ctr_rel_thresh: float
        everything above maximum_value * ctr_rel_thresh is included
    masses
        all the masses of the particels, needed to find potentials
    sfc/psc_positions
        array of the test particle positons

    Returns
    -------
    x_peak
        2d coordinates of centers
    y_peak
        2d coordinates of centers
    labels
        number labels for each GC

    """
    with plt.style.context("dark_background"):

        # 2d histogram using luminosities
        lums, xedges, yedges = np.histogram2d(
            star_positions[:, 0],
            star_positions[:, 1],
            bins=plt_bins,
            weights=scaled_stellar_lums,
            normed=False,
            range=[
                [-proj_width / 2, proj_width / 2],
                [-proj_width / 2, proj_width / 2],
            ],
        )
        # need to transpose for some reason
        lums = lums.T
        # =============================================================================
        #              get GC centers based on potential or density counts
        # =============================================================================
        if get_ctr is not None:  ####### edit this for halo finder support
            print("> calculating centers")
            if get_ctr[1] == "counts":
                print("> finding peaks using star density counts ")
                pc_accuracy = get_ctr[2]
                centring_bins = int(proj_width / pc_accuracy)

                center_threshold_pixels = ctr_dist_thresh / pc_accuracy

                # find peaks, returns indeces in the matrix

                # 2d histogram using raw counts
                counts, _, _ = np.histogram2d(
                    star_positions[:, 0],
                    star_positions[:, 1],
                    bins=centring_bins,
                    normed=False,
                    range=[
                        [-proj_width / 2, proj_width / 2],
                        [-proj_width / 2, proj_width / 2],
                    ],
                )
                counts = counts.T
                # get bin centers
                x_ctr = 0.5 * (xedges[1:] + xedges[:-1])
                y_ctr = 0.5 * (yedges[1:] + yedges[:-1])

                peaks = peak_local_max(
                    counts,
                    threshold_rel=ctr_rel_thresh,
                    min_distance=center_threshold_pixels,
                )

                col_idx = peaks[:, 1]
                row_idx = peaks[:, 0]
                # translate indeces to coordinates
                x_peak = x_ctr[col_idx]
                y_peak = y_ctr[row_idx]

            elif get_ctr[1] == "potential":
                # print('> finding peaks using grav potentials')
                pc_accuracy = get_ctr[2]  # pc/pixel
                centring_bins = int(proj_width / pc_accuracy)  # no. pixels ctr
                pc_per_pixel = proj_width / centring_bins  # might be rounded
                # calculate the minimum distance between centers in pc
                center_threshold_pixels = int(ctr_dist_thresh / pc_accuracy)
                print(
                    "> finding peaks using grav potentials with precision",
                    pc_per_pixel,
                    "pc/pixel",
                )
                print("> pixels along each dimension", centring_bins)
                print(
                    "> centers have to  be",
                    center_threshold_pixels,
                    "pixels apart, corresponding to ",
                    ctr_dist_thresh,
                    "pc",
                )
                print("> top", 100 * (1 - ctr_rel_thresh), "% of deepest potentials")

                phi = Potential(pos=star_positions, m=masses, method="bruteforce")

                grav, xedges, yedges = np.histogram2d(
                    star_positions[:, 0],
                    star_positions[:, 1],
                    bins=centring_bins,
                    weights=np.abs(phi),
                    normed=False,
                    range=[
                        [-proj_width / 2, proj_width / 2],
                        [-proj_width / 2, proj_width / 2],
                    ],
                )
                grav = grav.T
                x_ctr = 0.5 * (xedges[1:] + xedges[:-1])
                y_ctr = 0.5 * (yedges[1:] + yedges[:-1])

                peaks = peak_local_max(
                    grav,
                    num_peaks=num_ctr,
                    min_distance=center_threshold_pixels,
                    threshold_rel=ctr_rel_thresh,
                )

                col_idx = peaks[:, 1]
                row_idx = peaks[:, 0]

                x_peak = x_ctr[col_idx]
                y_peak = y_ctr[row_idx]

                gc_labels = np.arange(1, x_peak.size + 1, 1)

                print("> found Centers for", x_peak.size)

            else:
                print("!centering method not supported!")
                exit

        else:
            pass

        fig = plt.figure(figsize=(14, 12), dpi=300)
        ax = fig.add_subplot(111, facecolor=cm.inferno(0))

        # color maps
        if lum_scale[0] == "static":
            rectbin = plt.imshow(
                lums,
                cmap="inferno",
                interpolation="gaussian",
                origin="lower",
                extent=[
                    -proj_width / 2,
                    proj_width / 2,
                    -proj_width / 2,
                    proj_width / 2,
                ],
                norm=LogNorm(vmin=lum_scale[1], vmax=lum_scale[2]),
            )
        elif lum_scale[0] == "dynamic":
            rectbin = plt.imshow(
                lums,
                cmap="inferno",
                interpolation="gaussian",
                origin="lower",
                extent=[
                    -proj_width / 2,
                    proj_width / 2,
                    -proj_width / 2,
                    proj_width / 2,
                ],
                norm=LogNorm(),
            )
        else:
            print("!not a valid color scale!")
            exit()

        # print('lum plot')

        # =============================================================================
        #                            Optionally plot annotations
        # =============================================================================
        if plt_ctrs is not None:
            x_peak = plt_ctrs[:, 0]
            y_peak = plt_ctrs[:, 1]
            gc_labels = plt_ctrs[:, 3]

            plt.scatter(x_peak, y_peak, color="green", marker="x", linewidths=0.5, s=10)
            plt.xlim(-proj_width / 2, proj_width / 2)
            plt.ylim(-proj_width / 2, proj_width / 2)

            # iterate over labels and label each scatter point
            for i, label in enumerate(gc_labels):
                plt.annotate(
                    label,
                    (x_peak[i], y_peak[i]),
                    fontsize=3,
                    ha="center",
                    color="white",
                )
        # =============================================================================
        #                              plot centers of max values if true
        # =============================================================================
        if get_ctr is not None:
            # ^can't put in the same line since it will check the tuple regardless
            if get_ctr[3] is True:
                # optionally annotate with found centers
                plt.scatter(
                    x_peak, y_peak, color="green", marker="x", linewidths=0.5, s=10
                )
                plt.xlim(-proj_width / 2, proj_width / 2)
                plt.ylim(-proj_width / 2, proj_width / 2)

                # iterate over labels and label each scatter point
                for i, label in enumerate(gc_labels):
                    plt.annotate(
                        label,
                        (x_peak[i], y_peak[i]),
                        fontsize=3,
                        ha="center",
                        color="white",
                    )
        # =============================================================================
        #                            plot aesthetics
        # =============================================================================

        plt_label = (
            r"$\lambda = 1500\;\AA$ Projected Monochromatic Luminosity"
            r" $\left(erg\;s^{-1}\AA^{-1} \right)$"
        )

        # add color bar to the bottom
        # fig.subplots_adjust(wspace=0, hspace=0, bottom=.1)
        # cbar_ax = fig.add_axes([.178, .090, 0.67, 0.010])
        # cbar = fig.colorbar(
        #              rectbin,
        #              cax=cbar_ax,
        #              orientation='horizontal',
        #              pad=0
        #             )
        # cbar.set_label(
        #     label=plt_label,
        #     size=12
        #     )

        # add color bar inside
        fig.subplots_adjust(wspace=0, hspace=0, bottom=0.1)
        # [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.319, 0.835, 0.39, 0.008])
        cbar = fig.colorbar(rectbin, cax=cbar_ax, orientation="horizontal", pad=0)
        cbar_ax.xaxis.set_label_position("top")
        cbar.set_label(label=plt_label, labelpad=8, size=12)

        # annotate with time
        ax.text(
            -proj_width * 0.375,
            -proj_width * 0.45,
            "t = %.2f Myr" % (time),
            size=12,
            ha="center",
            va="center",
            color="white",
        )

        # add scale bar
        rect = patches.Rectangle(
            xy=(-proj_width * 0.125, -proj_width * 0.45),
            width=proj_width * 0.25,
            height=proj_width * 0.005,
            linewidth=0,
            edgecolor="white",
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(
            0,
            -proj_width * 0.475,
            "{} pc".format(int(proj_width / 4)),
            size=12,
            ha="center",
            va="center",
            color="white",
        )
        # ax.set_axis_off()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        ax.add_artist(ax.patch)
        ax.patch.set_zorder(-1)

        if get_ctr is not None:
            return x_peak, y_peak, gc_labels
        else:
            pass


def king_profiler(
    star_pos,
    lums,
    masses,
    ages,
    gc_ctr,
    gc_rad,
    gc_label,
    bins=30,
    good_alpha=4,
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
        core_mass = get_masses(clust_x, clust_y, clust_masses, fit_r_c)

        # =============================================================================

        # quantify goodness of fit
        # p_value, reduced_chi_2 = chi_squared(
        #     theory=theory_rho, data=rho, sigma=err, num_params=4
        #     )

        if truncation_radius <= gc_rad:
            # realistic truncation radius labels
            trunc_mass = get_masses(clust_x, clust_y, clust_masses, truncation_radius)
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
            # plt.axvline(fit_r_c)
            # plt.text(fit_r_c, 0,r'$R_core$',rotation=90)
            plt.title(r"GC # {:.0f}".format(gc_label), fontsize=16)
            plt.ylabel(r"Surface Mass Density ($M_{\odot} \; pc^{-2}$)", fontsize=16)
            plt.xlabel(r"R ($pc$)", fontsize=16)
            plt.xscale("symlog")
            plt.yscale("symlog")
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
            )
        else:
            print(r"> bad alpha for GC #{:.0f}".format(gc_label))
            # return invalid values
            return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

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
                r"$M_{{total}}= {:.2e} \: M_{{\odot}}$" "\n" r"$t_{{age}}= {:.2f}$ Myr"
            ).format(tot_m, gc_char_age),
            zorder=1,
        )

        plt.title(r"No Fit GC # {:.0f}".format(gc_label), fontsize=16)
        plt.ylabel(r"Surface Mass Density ($M_{\odot} \; pc^{-2}$)", fontsize=16)
        plt.xlabel(r"R ($pc$)", fontsize=16)
        plt.xscale("symlog")
        plt.yscale("symlog")
        plt.grid(visible=True, which="both", axis="y", ls="--")
        plt.legend(fontsize=16)

        print(r"> can't fit GC #{:.0f} ".format(gc_label))

        # return invalid values
        return -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2
