"""
Shows how profiling works.
"""
import sys

sys.path.append("../")
import numpy as np
import os
import glob
from modules.macros import filter_snapshots, characterisitc_mass, sci_notation
from modules.luminosity.lum_functions import unpack_pop_ii_data
from modules.profiles.profile_functions import projected_surf_densities
from modules.profiles.profile_models import modified_king_model
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import colors
from scipy.optimize import curve_fit

# 0.70 biggest cluster
# strt = 421
# end = 421
# step = 1
# tr_id = 33387

# strt = 464
# end = 464
# step = 713
# tr_id = 21355

# 0.70 old cluster
# strt = 342
# end = 1000
# step = end - strt
# tr_id = 6319.0

# halo_data_directory = r"../halo_data/fs07_refine/fof_best"
# pop2_data_directory = r"../particle_data/pop_2_data/fs07_refine"
# pop2 = filter_snapshots(pop2_data_directory, strt, end, step)
# halo_ds = filter_snapshots(halo_data_directory, strt, end, step)
# efficiencies = 0.70


# 0.35  cluster
# strt = 464
# end = 464
# step = 713
# tr_id = 21355

# another 0.35
# strt = 669
# end = 667
# step = end - strt
# tr_id = 25761.0

strt = 448
end = 448
step = 1
tr_id = 15516.0

halo_data_directory = r"../halo_data/fs035_ms10/fof_best"
pop2_data_directory = r"../particle_data/pop_2_data/fs035_ms10"
pop2 = filter_snapshots(pop2_data_directory, strt, end, step)
halo_ds = filter_snapshots(halo_data_directory, strt, end, step)
efficiencies = 0.35

profile_plot_bins = 20
m_rad = 100
star_bins = 1000
star_lum_range = (2e34, 2e36)  # (9e32, 2e37)  # (2e34, 2e36)
pxl_size = (m_rad * 2 / star_bins) ** 2
cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))
eff_lcolor = "grey"  # cmap[0]
eff_errcol = "white"  # cmap[1]

leg_font = font_manager.FontProperties(family="serif", math_fontfamily="cm", size=10)
props = dict(
    boxstyle="round",
    facecolor="black",
    alpha=0.5,
    linewidth=0.8,
    edgecolor="white",
)
# process where the plot where be centered, its vir radius, etc. clean up code
pop_2 = []
snap_nums = []
halo_nums = []
halo_rads = []
halo_ctr = []

with plt.style.context("dark_background"):
    with plt.rc_context(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.size": 12,
        }
    ):
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(6, 6.0),
            dpi=400,
            sharex="row",
            sharey="row",
            facecolor=cm.Greys_r(0),
        )


for idx, (p2, ds) in enumerate(zip(pop2, halo_ds)):
    idx = idx % 2
    output_num = int(ds.split("/")[-1].split("_")[-1])
    cata = glob.glob(os.path.join(ds, "catalogue*.txt"))[0]
    cata_data = np.loadtxt(cata)
    halos = sorted(glob.glob(os.path.join(ds, "gc*.txt")))
    for halo in halos:
        halo_data = np.loadtxt(halo)
        if tr_id in halo_data[:, 0]:  # the star id number was found via trial and error
            halo_num = int(halo.split("/")[-1].split("_")[-1].split(".")[0])
            halo_center = cata_data[:, 1:4][cata_data[:, 0] == halo_num][0]
            halo_radius = float(cata_data[:, -1][cata_data[:, 0] == halo_num])
            print(output_num, halo_num, halo_center, halo_radius)

            pop_2.append(p2)
            snap_nums.append(output_num)
            halo_nums.append(halo_num)
            halo_rads.append(halo_radius)
            halo_ctr.append(halo_center)

            # get data
            t_myr = np.loadtxt(p2, max_rows=2)[0, 6]
            redshift = np.loadtxt(p2, max_rows=2)[1, 6]
            field_stars = np.loadtxt(os.path.join(ds, "field_stars.txt"))
            bound_stars = np.loadtxt(os.path.join(ds, "bound_stars.txt"))
            stars = np.vstack((field_stars, bound_stars))
            star_ids = stars[:, 0]
            star_lums = stars[:, 2]
            star_masses = stars[:, 6]
            star_ages = stars[:, 1]  # Myr
            star_bes = t_myr - star_ages

            x = stars[:, 3]
            y = stars[:, 4]
            z = stars[:, 5]

            halo_x = halo_data[:, 1]
            halo_y = halo_data[:, 2]
            halo_z = halo_data[:, 3]

            halo_star_ids = halo_data[:, 0]
            gc_lum_mask = np.isin(star_ids, halo_star_ids)
            halo_lums = star_lums[gc_lum_mask]
            halo_masses = star_masses[gc_lum_mask]
            with plt.style.context("dark_background"):
                with plt.rc_context(
                    {
                        "font.family": "serif",
                        "mathtext.fontset": "cm",
                        "xtick.labelsize": 10,
                        "ytick.labelsize": 10,
                        "font.size": 12,
                    }
                ):

                    # make master projection
                    xy_lums, _, _ = np.histogram2d(
                        x,
                        y,
                        bins=star_bins,
                        weights=star_lums,
                        normed=False,
                        range=[[-m_rad, m_rad], [-m_rad, m_rad]],
                    )
                    xy_lums = xy_lums.T
                    xy = ax.imshow(
                        xy_lums / pxl_size,
                        cmap="inferno",
                        # interpolation="gaussian",
                        origin="lower",
                        extent=[-m_rad, m_rad, -m_rad, m_rad],
                        norm=LogNorm(star_lum_range[0], star_lum_range[1]),
                    )
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.xaxis.set_ticks_position("none")
                    ax.yaxis.set_ticks_position("none")

                    # draw inset
                    axins = ax.inset_axes([0.10, 0.65, 0.25, 0.25])
                    axins.imshow(
                        xy_lums / pxl_size,
                        origin="lower",
                        cmap="inferno",
                        extent=[-m_rad, m_rad, -m_rad, m_rad],
                        norm=LogNorm(star_lum_range[0], star_lum_range[1]),
                    )
                    #!!! make a 20 pc inset
                    axins.set_xlim(halo_center[0] - 10, halo_center[0] + 10)
                    axins.set_ylim(halo_center[1] - 10, halo_center[1] + 10)

                    axins.set_xticklabels([])
                    axins.set_yticklabels([])
                    axins.xaxis.set_ticks_position("none")
                    axins.yaxis.set_ticks_position("none")
                    axins.patch.set_alpha(0.8)

                    mark_inset(ax, axins, loc1=1, loc2=3, edgecolor="white", alpha=0.5)

                    # cleaned up inset for profiling
                    ax_clean_ins = ax.inset_axes([0.10, 0.40, 0.25, 0.25])
                    gc_lums, _, _ = np.histogram2d(
                        halo_x,
                        halo_y,
                        bins=int(
                            20 / (m_rad * 2 / 1000)
                        ),  # make sure the resolution is the same 20 pc inset
                        weights=halo_lums,
                        normed=False,
                        range=[[-10, 10], [-10, 10]],
                    )
                    gc_lums = gc_lums.T
                    ax_clean_ins.imshow(
                        gc_lums / pxl_size,
                        origin="lower",
                        cmap="inferno",
                        extent=[-10, 10, -10, 10],
                        norm=LogNorm(star_lum_range[0], star_lum_range[1]),
                        alpha=1,
                    )
                    ax_clean_ins.patch.set_alpha(0.8)
                    ax_clean_ins.set_xticklabels([])
                    ax_clean_ins.set_yticklabels([])
                    ax_clean_ins.xaxis.set_ticks_position("none")
                    ax_clean_ins.yaxis.set_ticks_position("none")

                    # profile inset
                    with plt.rc_context(
                        {
                            "xtick.labelsize": 7,
                            "ytick.labelsize": 7,
                            "font.size": 10,
                        }
                    ):
                        ax_prof_ins = ax.inset_axes([0.10, 0.15, 0.25, 0.25])
                    r, rho, err, _, _, _, half_r = projected_surf_densities(
                        x_coord=halo_x,
                        y_coord=halo_y,
                        lums=halo_lums,
                        masses=halo_masses,
                        radius=halo_radius,
                        num_bins=profile_plot_bins,
                        log_bins=True,
                        dr=None,
                        calc_half_r=True,
                    )
                    fit_params, cov_matrix = curve_fit(
                        f=modified_king_model,
                        xdata=r,
                        ydata=rho,
                        sigma=err,
                        absolute_sigma=True,
                        p0=[1e5, 0.2, 2, 10],
                        bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
                    )
                    sigma_naught = fit_params[0]
                    fit_r_c = fit_params[1]
                    fit_alpha = fit_params[2]
                    fit_sigma_bg = fit_params[3]
                    core_mass = characterisitc_mass(
                        halo_x, halo_y, halo_masses, fit_r_c
                    )
                    half_mass = characterisitc_mass(halo_x, halo_y, halo_masses, half_r)
                    theory_r = np.geomspace(
                        0.010, 20, 200, endpoint=False
                    )  # smooth version
                    theory_rho = modified_king_model(theory_r, *fit_params)
                    theory_rho_sig = modified_king_model(
                        theory_r, *fit_params - 2 * np.sqrt(np.diag(cov_matrix))
                    )

                    ax_prof_ins.errorbar(
                        r,
                        rho,
                        yerr=err,
                        fmt="o",
                        capsize=3,
                        capthick=1,
                        elinewidth=1,
                        ms=3,
                        alpha=1,
                        c=eff_errcol,
                    )
                    ax_prof_ins.plot(
                        theory_r,
                        theory_rho,
                        ls="--",
                        linewidth=2,
                        alpha=0.6,
                        zorder=3,
                        color=eff_lcolor,
                    )
                    # profiler legend
                    # insleg = ax_prof_ins.legend(
                    #     loc="upper right",
                    #     bbox_to_anchor=(2.05, 0.53),
                    #     fontsize=8,
                    # )
                    # insleg.get_frame().set_boxstyle("Square")

                    ax_prof_ins.patch.set_alpha(0.8)
                    ax_prof_ins.set_ylabel(
                        r"$\mathrm{\log_{10}\; \Sigma \:\:(M_{\odot}\:pc^{-2})}$",
                        fontproperties=leg_font,
                    )
                    ax_prof_ins.set_xlabel(
                        r"$ \mathrm{\log_{10}\;  R \:(pc)}$",
                        fontproperties=leg_font,
                    )
                    ax_prof_ins.set_ylim(bottom=3, top=2e5)
                    ax_prof_ins.set_xlim(left=0.010, right=20)
                    ax_prof_ins.set_xscale("log")
                    ax_prof_ins.set_yscale("log")
                    # ax_prof_ins.yaxis.set_major_locator(MaxNLocator(5))
                    ax_prof_ins.yaxis.set_major_locator(
                        mpl.ticker.LogLocator(base=10, numticks=10)
                    )

                    # tick label mod
                    fig.canvas.draw()
                    x_labels = [
                        i.get_text().replace("10^", "")
                        for i in ax_prof_ins.get_xticklabels()
                    ]
                    y_labels = [
                        i.get_text().replace("10^", "")
                        for i in ax_prof_ins.get_yticklabels()
                    ]
                    ax_prof_ins.set_xticklabels(x_labels)
                    ax_prof_ins.set_yticklabels(y_labels)

                    # fit results
                    fit_results = (
                        r"$R_{{\mathrm{{core}}}} = {:.2f} \: \mathrm{{pc}}$"
                        "\n"
                        r"$R_{{\mathrm{{half}}}} = {:.2f} \: \mathrm{{pc}}$"
                        "\n"
                        r"$\alpha = {:.2f} $"
                        "\n"
                        r"$\Sigma_0={}\:\left(\mathrm{{M_{{\odot}}{{pc^{{2}}}}}}\right)$"
                        "\n"
                        r"$M_{{\mathrm{{core}}}} = {} \: \mathrm{{M}}_{{\odot}}$"
                        "\n"
                        r"$M_{{\mathrm{{half}}}} = {} \: \mathrm{{M}}_{{\odot}}$"
                    ).format(
                        fit_r_c,
                        half_r,
                        fit_alpha,
                        sci_notation(2, sigma_naught),
                        sci_notation(2, core_mass),
                        sci_notation(2, half_mass),
                    )
                    ax.text(
                        0.36,
                        0.39,
                        fit_results,
                        fontsize=10,
                        ha="left",
                        va="top",
                        color="white",
                        fontproperties=leg_font,
                        transform=ax.transAxes,
                        bbox={
                            "boxstyle": "Square",
                            # have control over edge alpha and face alpha
                            "facecolor": colors.to_rgba("black")[:-1] + (0.7,),
                            "linewidth": 0.8,
                            "edgecolor": "white",
                            "pad": 0.40,
                        },
                    )

                    # add scales
                    master_scale = patches.Rectangle(
                        xy=(0.85 * m_rad, -0.25 * m_rad),
                        width=2,
                        height=m_rad / 2,
                        linewidth=0,
                        edgecolor="white",
                        facecolor="white",
                        # transform=ax.transAxes,
                    )
                    ax.text(
                        0.92 * m_rad,
                        0,
                        r"$\mathrm{{{:.0f}\: pc}}$".format(m_rad / 2),
                        ha="center",
                        va="center",
                        color="white",
                        rotation=270,
                        fontproperties=leg_font,
                        # transform=ax.transAxes,
                    )
                    ax.add_patch(master_scale)

                    inset_scale = patches.Rectangle(
                        xy=(halo_center[0] - 11.5, halo_center[1] - 5),
                        width=0.5,
                        height=10,
                        linewidth=0,
                        alpha=1,
                        edgecolor="white",
                        facecolor="white",
                        clip_on=False,
                    )
                    axins.set_ylabel(
                        r"$\mathrm{10\:pc}$", fontproperties=leg_font, labelpad=1
                    )
                    axins.add_patch(inset_scale)

                    #!!! add time and redshift
                    ax.text(
                        m_rad * 0.51,
                        m_rad * -0.93,
                        (
                            r"$\mathrm{{t = {:.1f} \: Myr}}$"
                            "\n"
                            r"$\mathrm{{z = {:.1f} }}$"
                        ).format(t_myr, redshift),
                        fontsize=12,
                        ha="left",
                        va="bottom",
                        color="white",
                        fontproperties=leg_font,
                        bbox=props,
                    )
                    if idx == 0:

                        #!!! add efficiency label
                        ax.text(
                            m_rad * -0.93,
                            m_rad * 0.93,
                            r"$f_* = {:.2f}$".format(efficiencies),
                            fontsize=12,
                            ha="left",
                            va="top",
                            color="white",
                            fontproperties=leg_font,
                            bbox=props,
                        )
                        # !!! add the luminosity color bar
                        # [left, bottom, width, height]
                        cbar_ax = fig.add_axes([0.52, 0.805, 0.35, 0.02])
                        cbar = fig.colorbar(
                            xy, cax=cbar_ax, pad=0, orientation="horizontal"
                        )
                        cbar_units = (
                            r"$\mathrm{\left(erg\:\:s^{-1}\:\AA^{-1}\:pc^{-2}\right)}$"
                        )
                        cbar_label = (
                            r"$\mathrm{Surface\: Brightness}$"
                            r"$, \mathrm{\lambda = 1500 \: \AA \:}$"
                        )
                        cbar.set_label(
                            label=cbar_units,
                            labelpad=2,
                            fontproperties=leg_font,
                            fontsize=10,
                        )
                        cbar.ax.xaxis.set_tick_params(pad=2, labelsize=8)
                        cbar_ax.set_title(
                            cbar_label, fontsize=11, fontproperties=leg_font
                        )
                    #!!! enable for time evolution
                    # if idx == 1:
                    #     with plt.style.context("dark_background"):
                    #         with plt.rc_context(
                    #             {
                    #                 "font.family": "serif",
                    #                 "mathtext.fontset": "cm",
                    #                 "xtick.labelsize": 10,
                    #                 "ytick.labelsize": 10,
                    #                 "font.size": 12,
                    #             }
                    #         ):
                    #             fig, ax = plt.subplots(
                    #                 nrows=1,
                    #                 ncols=2,
                    #                 figsize=(12.0, 6.0),
                    #                 dpi=400,
                    #                 sharex="row",
                    #                 sharey="row",
                    #                 facecolor=cm.Greys_r(0),
                    #             )
                    # plt.subplots_adjust(hspace=0, wspace=-0.05)

# plt.subplots_adjust(hspace=0)
plt.savefig(
    os.path.expanduser(
        (
            "~/g_drive/Research/AstrophysicsSimulation/sci_plots/final/"
            "profiling_demo_f3.png"
        )
    ),
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.05,
)

# grids
# ax_prof_ins.grid(
#     visible=True,
#     which="both",
#     axis="y",
#     ls="--",
#     color="dimgrey",
#     zorder=0.5,
# )
