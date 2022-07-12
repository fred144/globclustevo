import sys

sys.path.append("..")
import numpy as np
import os
import glob
from modules.macros import filter_snapshots, characterisitc_mass
from modules.luminosity.lum_functions import unpack_pop_ii_data
from modules.profiles.profile_functions import projected_surf_densities
from modules.profiles.profile_models import modified_king_model
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
from scipy.optimize import curve_fit

runsavepath = "../rendering/luminosity/fs07_refine/gc_tracking"
if not os.path.exists(runsavepath):
    print("# Creating new sequence directory", runsavepath)
    os.makedirs(runsavepath)


strt = 665
end = 918
step = 500

halo_data_directory = r"../halo_data/fs07_refine/fof_best"
pop2_data_directory = r"../particle_data/pop_2_data/fs07_refine"
pop2 = filter_snapshots(pop2_data_directory, strt, end, step)
halo_ds = filter_snapshots(halo_data_directory, strt, end, step)

# change font for entire module
mpl.rc("font", family="serif")


def sci_notation(decimal_places, exp_string):
    """
    turns '1.00e+06' into scientific notation with times and power formatter
    for LaTeX
    """
    start = decimal_places + 2
    end = decimal_places + 3
    return r"{}\:\times\:10^{:1} ".format(exp_string[:start], int(exp_string[end:]))


project_plot_bins = 20
radius = 20
cmap = cm.get_cmap("inferno")
cmap = cmap(np.linspace(0, 1, 20))
line_color = cmap[18]
scatter_color = cmap[15]
# TODO: make a an independent profiler and lum map three panel. etc.
for p2, ds in zip(pop2, halo_ds):
    output_num = int(ds.split("/")[-1].split("_")[-1])

    _, scaled_stellar_lums, masses, ages, tz, ids = unpack_pop_ii_data(
        p2, return_ids=True, return_z=True
    )
    t_myr = tz[0]
    redshift = tz[1]

    halos = sorted(glob.glob(os.path.join(ds, "gc*.txt")))
    for halo in halos:

        halo_data = np.loadtxt(halo)

        if 33387 in halo_data[:, 0]:  # the star id number was found via trial and error
            try:
                halo_num = int(halo.split("/")[-1].split("_")[-1].split(".")[0])
                gc_lum_mask = np.isin(ids, halo_data[:, 0])
                gc_lums = scaled_stellar_lums[gc_lum_mask]
                gc_mass = masses[gc_lum_mask]
                gc_ages = ages[gc_lum_mask] / 1e6
                gc_be = t_myr - gc_ages
                x = halo_data[:, 1]
                y = halo_data[:, 2]
                z = halo_data[:, 3]

                xy_lums, _, _ = np.histogram2d(
                    x,
                    y,
                    bins=250,
                    weights=gc_lums,
                    normed=False,
                    range=[[-radius, radius], [-radius, radius]],
                )

                xz_lums, _, _ = np.histogram2d(
                    x,
                    z,
                    bins=250,
                    weights=gc_lums,
                    normed=False,
                    range=[[-radius, radius], [-radius, radius]],
                )

                yz_lums, _, _ = np.histogram2d(
                    y,
                    z,
                    bins=250,
                    weights=gc_lums,
                    normed=False,
                    range=[[-radius, radius], [-radius, radius]],
                )

                xy_lums = xy_lums.T
                xz_lums = xy_lums.T
                yz_lums = yz_lums.T

                with plt.style.context("dark_background"):
                    fig, ax = plt.subplots(
                        nrows=2,
                        ncols=3,
                        sharex="row",
                        sharey="row",
                        figsize=(10, 7),
                        dpi=300,
                        facecolor=cm.Greys_r(0),
                    )
                    # plot the luminosity projection for 3 viewing angles
                    xy = ax[0, 0].imshow(
                        xy_lums,
                        cmap="inferno",
                        interpolation="gaussian",
                        origin="lower",
                        extent=[-radius, radius, -radius, radius],
                        norm=LogNorm(),
                    )

                    xz = ax[0, 1].imshow(
                        xz_lums,
                        cmap="inferno",
                        interpolation="gaussian",
                        origin="lower",
                        extent=[-radius, radius, -radius, radius],
                        norm=LogNorm(),
                    )

                    yz = ax[0, 2].imshow(
                        yz_lums,
                        cmap="inferno",
                        interpolation="gaussian",
                        origin="lower",
                        extent=[-radius, radius, -radius, radius],
                        norm=LogNorm(),
                    )

                    # calculated projected surface densities for each three projections

                    xy_r, xy_rho, xy_err, _, _, xy_half_r, _ = projected_surf_densities(
                        x_coord=x,
                        y_coord=y,
                        lums=gc_lums,
                        masses=gc_mass,
                        radius=radius,
                        num_bins=project_plot_bins,
                        log_bins=True,
                        dr=None,
                        calc_half_r=True,
                    )

                    xz_r, xz_rho, xz_err, _, _, _, xz_half_r = projected_surf_densities(
                        x_coord=x,
                        y_coord=z,
                        lums=gc_lums,
                        masses=gc_mass,
                        radius=radius,
                        num_bins=project_plot_bins,
                        log_bins=True,
                        dr=None,
                        calc_half_r=True,
                    )

                    yz_r, yz_rho, yz_err, _, _, _, yz_half_r = projected_surf_densities(
                        x_coord=y,
                        y_coord=z,
                        lums=gc_lums,
                        masses=gc_mass,
                        radius=radius,
                        num_bins=project_plot_bins,
                        log_bins=True,
                        dr=None,
                        calc_half_r=True,
                    )

                    # fit them
                    fit_params, cov_matrix = curve_fit(
                        f=modified_king_model,
                        xdata=xy_r,
                        ydata=xy_rho,
                        sigma=xy_err,
                        absolute_sigma=True,
                        p0=[1e5, 0.2, 2, 10],
                        bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
                    )
                    xy_sigma_naught = fit_params[0]
                    xy_fit_r_c = fit_params[1]
                    xy_fit_alpha = fit_params[2]
                    xy_fit_sigma_bg = fit_params[3]
                    xy_core_mass = characterisitc_mass(x, y, gc_mass, xy_fit_r_c)
                    xy_theory_r = np.geomspace(
                        xy_r[0], radius, 200, endpoint=False
                    )  # smooth version
                    xy_theory_rho = modified_king_model(xy_theory_r, *fit_params)
                    xy_plot_label = (
                        r"$R_{{core}} = {:.2f} \: \mathrm{{pc}}$"
                        "\n"
                        r"$\alpha = {:.2f} $"
                        "\n"
                        r"$\Sigma_0 = {} (\mathrm{{M}}_{{\odot}} \; \mathrm{{pc}}^{{-2}})$"
                        "\n"
                        r"$M_{{core}} = {} \: \mathrm{{M}}_{{\odot}}$"
                    ).format(
                        xy_fit_r_c,
                        xy_fit_alpha,
                        sci_notation(2, "{:.2e}".format(xy_sigma_naught)),
                        sci_notation(2, "{:.2e}".format(xy_core_mass)),
                    )

                    fit_params, cov_matrix = curve_fit(
                        f=modified_king_model,
                        xdata=xz_r,
                        ydata=xz_rho,
                        sigma=xz_err,
                        absolute_sigma=True,
                        p0=[1e5, 0.2, 2, 10],
                        bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
                    )
                    xz_sigma_naught = fit_params[0]
                    xz_fit_r_c = fit_params[1]
                    xz_fit_alpha = fit_params[2]
                    xz_fit_sigma_bg = fit_params[3]
                    xz_core_mass = characterisitc_mass(x, z, gc_mass, xz_fit_r_c)
                    xz_theory_r = np.geomspace(
                        xz_r[0], radius, 200, endpoint=False
                    )  # smooth version
                    xz_theory_rho = modified_king_model(xz_theory_r, *fit_params)
                    xz_plot_label = (
                        r"$R_{{core}} = {:.2f} \: \mathrm{{pc}}$"
                        "\n"
                        r"$\alpha = {:.2f} $"
                        "\n"
                        r"$\Sigma_0 = {} (\mathrm{{M}}_{{\odot}} \; \mathrm{{pc}}^{{-2}})$"
                        "\n"
                        r"$M_{{core}} = {} \: \mathrm{{M}}_{{\odot}}$"
                    ).format(
                        xz_fit_r_c,
                        xz_fit_alpha,
                        sci_notation(2, "{:.2e}".format(xz_sigma_naught)),
                        sci_notation(2, "{:.2e}".format(xz_core_mass)),
                    )

                    fit_params, cov_matrix = curve_fit(
                        f=modified_king_model,
                        xdata=yz_r,
                        ydata=yz_rho,
                        sigma=yz_err,
                        absolute_sigma=True,
                        p0=[1e5, 0.2, 2, 10],
                        bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
                    )
                    yz_sigma_naught = fit_params[0]
                    yz_fit_r_c = fit_params[1]
                    yz_fit_alpha = fit_params[2]
                    yz_fit_sigma_bg = fit_params[3]
                    yz_core_mass = characterisitc_mass(y, z, gc_mass, yz_fit_r_c)
                    yz_theory_r = np.geomspace(
                        yz_r[0], radius, 200, endpoint=False
                    )  # smooth version
                    yz_theory_rho = modified_king_model(yz_theory_r, *fit_params)
                    yz_plot_label = (
                        r"$R_{{core}} = {:.2f} \: \mathrm{{pc}}$"
                        "\n"
                        r"$\alpha = {:.2f} $"
                        "\n"
                        r"$\Sigma_0 = {} (\mathrm{{M}}_{{\odot}} \; \mathrm{{pc}}^{{-2}})$"
                        "\n"
                        r"$M_{{core}} = {} \: \mathrm{{M}}_{{\odot}}$"
                    ).format(
                        yz_fit_r_c,
                        yz_fit_alpha,
                        sci_notation(2, "{:.2e}".format(yz_sigma_naught)),
                        sci_notation(2, "{:.2e}".format(yz_core_mass)),
                    )

                    # plot the error bars and theory curves
                    leg_font = font_manager.FontProperties(
                        family="serif", math_fontfamily="cm", size=9
                    )
                    leg_font
                    ax[1, 0].errorbar(
                        xy_r,
                        xy_rho,
                        yerr=xy_err,
                        # c="mediumpurple",
                        fmt="o",
                        capsize=5,
                        capthick=3,
                        elinewidth=3,
                        label=(r"$R_{{half}} = {:.2f} \: \mathrm{{pc}}$").format(
                            xy_half_r
                        ),
                        c=scatter_color,
                        # zorder=1,
                    )
                    ax[1, 0].plot(
                        xy_theory_r,
                        xy_theory_rho,
                        # color="darkorange",
                        linewidth=4,
                        label=xy_plot_label,
                        alpha=0.9,
                        zorder=3,
                        color=line_color,
                    )
                    ax[1, 0].legend(loc="lower left", fontsize=8, prop=leg_font)

                    ax[1, 1].errorbar(
                        xz_r,
                        xz_rho,
                        yerr=xz_err,
                        # c="mediumpurple",
                        fmt="o",
                        capsize=5,
                        capthick=3,
                        elinewidth=3,
                        label=(r"$R_{{half}} = {:.2f} \: \mathrm{{pc}}$").format(
                            xz_half_r
                        ),
                        c=scatter_color,
                        # zorder=1,
                    )
                    ax[1, 1].plot(
                        xz_theory_r,
                        xz_theory_rho,
                        # color="darkorange",
                        linewidth=4,
                        label=xz_plot_label,
                        alpha=0.9,
                        zorder=3,
                        color=line_color,
                    )
                    ax[1, 1].legend(loc="lower left", fontsize=8, prop=leg_font)

                    ax[1, 2].errorbar(
                        yz_r,
                        yz_rho,
                        yerr=yz_err,
                        # c="mediumpurple",
                        fmt="o",
                        capsize=5,
                        capthick=3,
                        elinewidth=3,
                        label=(r"$R_{{half}} = {:.2f} \: \mathrm{{pc}}$").format(
                            yz_half_r
                        ),
                        c=scatter_color,
                        # zorder=1,
                    )
                    ax[1, 2].plot(
                        yz_theory_r,
                        yz_theory_rho,
                        # color="darkorange",
                        linewidth=4,
                        label=yz_plot_label,
                        alpha=0.9,
                        zorder=3,
                        color=line_color,
                    )
                    ax[1, 2].legend(loc="lower left", fontsize=8, prop=leg_font)

                    # add the luminosity color bar
                    fig.subplots_adjust(wspace=0, hspace=0, bottom=0.1)
                    # [left, bottom, width, height]
                    cbar_ax = fig.add_axes([0.90, 0.1, 0.01, 0.768])
                    cbar = fig.colorbar(xy, cax=cbar_ax, pad=0)
                    cbar_label = (
                        r"$\mathrm{Projected}\:\mathrm{Monochromatic}\:\mathrm{Luminosity}"
                        r", \mathrm{\lambda = 1500 \: \AA \:}"
                        r"\mathrm{\left(erg \:\: s^{-1} \: \AA^{-1} \right)} $"
                    )
                    cbar.set_label(
                        label=cbar_label,
                        fontsize=12,
                        labelpad=8,
                        fontproperties=leg_font,
                    )

                    # edit ticks, remove the numbers
                    ax[0, 0].axes.yaxis.set_ticklabels([])
                    ax[0, 0].axes.xaxis.set_ticklabels([])
                    ax[0, 0].tick_params(
                        axis="x",
                        which="both",
                        bottom=False,
                        top=False,
                        labelbottom=False,
                    )
                    ax[0, 1].tick_params(
                        axis="x",
                        which="both",
                        bottom=False,
                        top=False,
                        labelbottom=False,
                    )
                    ax[0, 2].tick_params(
                        axis="x",
                        which="both",
                        bottom=False,
                        top=False,
                        labelbottom=False,
                    )
                    # axes labels
                    ax[1, 0].set_ylabel(
                        r"$ \mathrm{\Sigma} \: (\mathrm{M}_{\odot} \; \mathrm{pc}^{-2})$",
                        fontproperties=leg_font,
                        fontsize=12,
                    )
                    ax[1, 1].set_xlabel(
                        r"$ \mathrm{R} \: (\mathrm{pc})$",
                        fontproperties=leg_font,
                        fontsize=12,
                    )
                    # grids
                    ax[1, 0].grid(
                        visible=True,
                        which="both",
                        axis="y",
                        ls="--",
                        color="dimgrey",
                        zorder=0.5,
                    )
                    ax[1, 1].grid(
                        visible=True,
                        which="both",
                        axis="y",
                        ls="--",
                        color="dimgrey",
                        zorder=0.5,
                    )
                    ax[1, 2].grid(
                        visible=True,
                        which="both",
                        axis="y",
                        ls="--",
                        color="dimgrey",
                        zorder=0.5,
                    )

                    # edit bottom row limits
                    ax[1, 0].set_xlim(left=0.012, right=radius * 2)
                    ax[1, 0].set_ylim(bottom=3e-1, top=3e5)
                    ax[1, 0].set_xscale("log")
                    ax[1, 0].set_yscale("log")

                    # add axes indicators
                    axes_ind = [("x", "y"), ("x", "z"), ("y", "z")]
                    for i, l in enumerate(axes_ind):
                        ax[0, i].text(
                            -radius * 0.55,
                            -radius * 0.9,
                            l[0],
                            size=6,
                            ha="center",
                            va="center",
                            color="white",
                        )
                        ax[0, i].text(
                            -radius * 0.9,
                            -radius * 0.55,
                            l[1],
                            size=6,
                            ha="center",
                            va="center",
                            color="white",
                        )

                        ax[0, i].arrow(
                            -radius * 0.9,
                            -radius * 0.9,
                            radius * 0.3,
                            0,
                            head_width=0.3,
                            head_length=0.3,
                            linewidth=0.5,
                            color="w",
                            length_includes_head=True,
                        )
                        ax[0, i].arrow(
                            -radius * 0.9,
                            -radius * 0.9,
                            0,
                            radius * 0.3,
                            head_width=0.3,
                            head_length=0.3,
                            linewidth=0.5,
                            color="w",
                            length_includes_head=True,
                        )

                    # add a scale
                    ax[0, 0].set_ylabel(
                        (r"$ \mathrm{10 \: pc}$" "\n"),
                        fontproperties=leg_font,
                        fontsize=12,
                    )

                    rect = patches.Rectangle(
                        xy=(-radius * 1.2, -50),
                        width=0.25,
                        height=100,
                        linewidth=0,
                        edgecolor="white",
                        facecolor="white",
                        clip_on=False,
                    )
                    ax[0, 0].add_patch(rect)
                    ax[0, 0].set_xlim(-radius, radius)
                    ax[0, 1].set_xlim(-radius, radius)
                    ax[0, 2].set_xlim(-radius, radius)

                    # add time and redshift
                    props = dict(
                        boxstyle="round",
                        facecolor="black",
                        alpha=0.5,
                        linewidth=0.8,
                        edgecolor="white",
                    )
                    ax[0, 0].text(
                        -radius * 0.9,
                        radius * 0.9,
                        (
                            r"$\mathrm{{t = {:.2f} \: Myr}}$"
                            "\n"
                            r"$\mathrm{{z = {:.2f} }}$"
                        ).format(t_myr, redshift),
                        size=12,
                        ha="left",
                        va="top",
                        color="white",
                        fontproperties=leg_font,
                        bbox=props,
                    )

                    # add efficiency label
                    ax[0, 2].text(
                        radius * 0.9,
                        -radius * 0.9,
                        "$f_{*} = 0.70$",
                        size=12,
                        ha="right",
                        va="bottom",
                        color="white",
                        fontproperties=leg_font,
                        bbox=props,
                    )

                    plt.subplots_adjust(hspace=-0.033, wspace=0)

                    # add a histogram of ages inside the cluster
                    # [left, bottom, width, height]
                    ax_inset = fig.add_axes([0.53, 0.75, 0.10, 0.10])
                    ax_inset.patch.set_alpha(0.5)
                    bins = np.linspace(300, 600, 10)
                    ax_inset.hist(gc_be, bins, alpha=0.8, color="w", edgecolor="black")

                    ax_inset.tick_params(labelsize=5)
                    ax_inset.set_xlabel(
                        "$\mathrm{Star \: Birth \: Time \: (Myr)}$",
                        fontproperties=leg_font,
                        fontsize=6,
                        labelpad=0,
                    )
                    ax_inset.set_ylabel(
                        "$\mathrm{Counts}$",
                        fontproperties=leg_font,
                        fontsize=6,
                        labelpad=0,
                    )
                    ax_inset.set_yscale("log")
                    # ax_inset.set_xlim("log")
                    ax_inset.set_xlim(300, 600)
                    ax_inset.set_ylim(1, 1e4)

                    save_name = os.path.join(
                        runsavepath, "tracked_{}".format(output_num)
                    )

                    # plt.savefig(save_name, dpi=300, bbox_inches="tight")

                print(output_num, halo_num)
            except:
                print(">skipping", output_num)
                pass

            break
        else:
            pass
