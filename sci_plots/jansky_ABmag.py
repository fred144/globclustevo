"""
Projected  luminosity time series plots.
"""
import sys

sys.path.append("../")
import numpy as np
import os
from modules.macros import (
    filter_snapshots,
    characterisitc_mass,
    sci_notation,
    ram_fields,
)
from modules.profiles.profile_functions import projected_surf_densities
from modules.profiles.profile_models import modified_king_model
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
from matplotlib import colors
from modules.match_t_sims import find_matching_time, get_snapshots
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


f7_strt = 113
f7_end = 1110
f3_strt = 154
f3_end = 1316
step = 1


f7_pop2_dir = r"../particle_data/pop_2_data/fs07_refine"
f3_pop2_dir = r"../particle_data/pop_2_data/fs035_ms10"


f7_halo_dir = r"../halo_data/fs07_refine/fof_best"
f3_halo_dir = r"../halo_data/fs035_ms10/fof_best"

f7_pop2 = filter_snapshots(
    r"../particle_data/pop_2_data/fs07_refine", f7_strt, f7_end, step
)
# matched snapshots
f3_pop2 = filter_snapshots(
    r"../particle_data/pop_2_data/fs035_ms10", f3_strt, f3_end, step
)

# optional if you want to match them
# _, f3_matched_nums = find_matching_time(sequence=f7_pop2, look_up_sequence=f3_pop2)
# f3_pop2_matched = get_snapshots(f3_pop2, get_list=f3_matched_nums)


f7_halos = filter_snapshots(os.path.relpath(f7_halo_dir), f7_strt, f7_end)
f3_halos = filter_snapshots(os.path.relpath(f3_halo_dir), f3_strt, f3_end)


# dictate which snapshots will be plotted
# f7_sn_list = np.array([377, 502, 820, 1069])  # looks promising
# f3_sn_list = np.array([414, 492, 730, 1277])

f7_sn_list = np.array([377, 502])  # looks promising
f3_sn_list = np.array([414, 492])

f7_plt_p2 = get_snapshots(f7_pop2, get_list=f7_sn_list)
f3_plt_p2 = get_snapshots(f3_pop2, get_list=f3_sn_list)

f7_plt_halo = get_snapshots(f7_halos, get_list=f7_sn_list)
f3_plt_halo = get_snapshots(f3_halos, get_list=f3_sn_list)


width = 400  # pc
rows = f7_sn_list.size
cols = 2
star_lum_bin = 5000
pxl_width = width / star_lum_bin
pxl_area = (pxl_width) ** 2  # pc^2
proj_r = width / 2
row_lims = [(-60, 60), (-100, 100)]
pc2_to_cm2 = 9.52140614e36
star_lum_range = [
    (3e34 / pc2_to_cm2, 5e37 / pc2_to_cm2),
    (3e34 / pc2_to_cm2, 5e36 / pc2_to_cm2),
]
leg_font = font_manager.FontProperties(family="serif", math_fontfamily="cm", size=8)

low_jansky = []
high_jansky = []
low_redshifts = []
high_redshifts = []
low_times = []
high_times = []
low_m_ab = []
high_m_ab = []
low_surf_b = []
high_surf_b = []

lum_distnaces = np.round([117645.8, 107989.8], -4)  # mpc
# lum_distnaces = [118321.3, 63724.7]
redshifts = [11, 11]
wavelength_angstrom = 1500
rad_per_arsec = np.pi / (180 * 60 * 60)


# theta = (width / star_lum_bin) / angular_size_distance

for i, (f7, f3) in enumerate(zip(f7_sn_list, f3_sn_list)):
    luminosity_distance = lum_distnaces[i]
    angular_size_distance = luminosity_distance / (1 + redshifts[i]) ** 2
    galaxy_angular_width = ((width / 1e6) / angular_size_distance) / rad_per_arsec
    print(
        "entire galaxy size",
        galaxy_angular_width,
        "at z = ",
        redshifts[i],
    )

    # get pre processed data from pop2 data sets
    f7_t_myr, f7_redshift = np.loadtxt(f7_plt_p2[i], max_rows=2)[0:2, 6]
    f3_t_myr, f3_redshift = np.loadtxt(f3_plt_p2[i], max_rows=2)[0:2, 6]

    f7_field_stars = np.loadtxt(os.path.join(f7_plt_halo[i], "field_stars.txt"))
    f7_bound_stars = np.loadtxt(os.path.join(f7_plt_halo[i], "bound_stars.txt"))
    f7_stars = np.vstack((f7_field_stars, f7_bound_stars))

    f7_star_ages = f7_stars[:, 1]  # Myr
    # f7_star_bes = f7_t_myr - f7_star_ages
    f7_pos_pc = f7_stars[:, 3:6]
    f7_star_lums = f7_stars[:, 2]

    f3_field_stars = np.loadtxt(os.path.join(f3_plt_halo[i], "field_stars.txt"))
    f3_bound_stars = np.loadtxt(os.path.join(f3_plt_halo[i], "bound_stars.txt"))
    f3_stars = np.vstack((f3_field_stars, f3_bound_stars))

    f3_star_ages = f3_stars[:, 1]  # Myr
    # f3_star_bes = f3_t_myr - f3_star_ages
    f3_pos_pc = f3_stars[:, 3:6]
    f3_star_lums = f3_stars[:, 2]

    f7_xy_lums, _, _ = np.histogram2d(
        f7_pos_pc[:, 0],
        f7_pos_pc[:, 1],
        bins=star_lum_bin,
        weights=f7_star_lums,
        normed=False,
        range=[[-proj_r, proj_r], [-proj_r, proj_r]],
    )
    f7_xy_lums = f7_xy_lums.T
    f7_surface_brigthness = f7_xy_lums / pxl_area

    f3_xy_lums, _, _ = np.histogram2d(
        f3_pos_pc[:, 0],
        f3_pos_pc[:, 1],
        bins=star_lum_bin,
        weights=f3_star_lums,
        normed=False,
        range=[[-proj_r, proj_r], [-proj_r, proj_r]],
    )
    f3_xy_lums = f3_xy_lums.T
    f3_surface_brigthness = f3_xy_lums / pxl_area

    f7_jy_per_arcsec2 = (
        1.56e-6 * (f7_surface_brigthness / 1e36) * ((1 + redshifts[i]) / 10) ** -4
    )
    f3_jy_per_arcsec2 = (
        1.56e-6 * (f3_surface_brigthness / 1e36) * ((1 + redshifts[i]) / 10) ** -4
    )
    f7_mag_per_arcsec2 = (
        -2.5
        * np.log10(
            f7_surface_brigthness / 1e36,
            out=np.full_like(f7_surface_brigthness, np.nan),
            where=(f7_surface_brigthness != 0),
        )
        + 10 * np.log10((1 + redshifts[i]) / 10)
        + 23.4
    )
    f3_mag_per_arcsec2 = (
        -2.5
        * np.log10(
            f3_surface_brigthness / 1e36,
            out=np.full_like(f3_surface_brigthness, np.nan),
            where=(f3_surface_brigthness != 0),
        )
        + 10 * np.log10((1 + redshifts[i]) / 10)
        + 23.4
    )

    # f7_jansky = (
    #     3.34e4 * (wavelength_angstrom) ** 2 * (f7_xy_lums / (pxl_area * pc2_to_cm2))
    # )

    # f3_jansky = (
    #     3.34e4 * (wavelength_angstrom) ** 2 * (f3_xy_lums / (pxl_area * pc2_to_cm2))
    # )

    # f3_ab_abs_mag = (
    #     -2.5
    #     * np.log10(
    #         f3_jansky,
    #         out=np.zeros_like(f3_jansky),
    #         where=(f3_jansky != 0),
    #     )
    #     + 8.9
    # )
    # f7_ab_abs_mag = (
    #     -2.5
    #     * np.log10(
    #         f7_jansky,
    #         out=np.zeros_like(f7_jansky),
    #         where=(f7_jansky != 0),
    #     )
    #     + 8.9
    # )
    # f3_surface_brigthness = f3_ab_abs_mag + 2.5 * np.log10(
    #     (pxl_width / (10 * rad_per_arsec)) ** 2
    # )
    # f7_surface_brigthness = f7_ab_abs_mag + 2.5 * np.log10(
    #     (pxl_width / (10 * rad_per_arsec)) ** 2
    # )

    # # f3_ab_apparent_mag = f3_ab_abs_mag + 5 * np.log10(luminosity_distance / 10) + 30
    # # f7_ab_apparent_mag = f7_ab_abs_mag + 5 * np.log10(luminosity_distance / 10) + 30

    # # find the flux density for the entire galaxy
    # f7_integrated_jansky = (
    #     3.34e4
    #     * (wavelength_angstrom) ** 2
    #     * (np.sum(f7_xy_lums) / (width**2 * pc2_to_cm2))
    # )
    # f3_integrated_jansky = (
    #     3.34e4
    #     * (wavelength_angstrom) ** 2
    #     * (np.sum(f3_xy_lums) / (width**2 * pc2_to_cm2))
    # )
    # # find corresponding absolute magnitude
    # f3_integrated_absmag = -2.5 * np.log10(f7_integrated_jansky) + 8.9
    # f7_integrated_absmag = -2.5 * np.log10(f3_integrated_jansky) + 8.9

    # # find the apparent magnitude
    # # recall, luminosity distance is in Mpc, that's why 30 is added
    # f3_integtrate_appmag = (
    #     f3_integrated_absmag + 5 * np.log10(luminosity_distance / 10) + 30
    # )
    # f7_integtrate_appmag = (
    #     f7_integrated_absmag + 5 * np.log10(luminosity_distance / 10) + 30
    # )

    # # + 5 * np.log10(luminosity_distance / 10)
    # # + 30
    # print(
    #     "M_AB of f3 galaxy",
    #     f3_integrated_absmag,
    #     "at z =",
    #     redshifts[i],
    # )
    # print(
    #     "M_AB of f7 galaxy",
    #     f7_integrated_absmag,
    #     "at z =",
    #     redshifts[i],
    # )
    # print(
    #     "m_AB of f3 galaxy",
    #     f3_integtrate_appmag,
    #     "at z =",
    #     redshifts[i],
    # )
    # print(
    #     "m_AB of f7 galaxy",
    #     f7_integtrate_appmag,
    #     "at z =",
    #     redshifts[i],
    # )

    # f7_integrated_ab_apparent_mag

    low_jansky.append(f3_jy_per_arcsec2)
    high_jansky.append(f7_jy_per_arcsec2)
    low_redshifts.append(f3_redshift)
    high_redshifts.append(f7_redshift)
    low_times.append(f3_t_myr)
    high_times.append(f7_t_myr)
    # low_m_ab.append(np.where(f3_ab_abs_mag >= 100, 0, f3_ab_abs_mag))
    # high_m_ab.append(np.where(f7_ab_abs_mag >= 100, 0, f7_ab_abs_mag))
    # account for empty bins.
    f3_mag_per_arcsec2[np.isnan(f3_mag_per_arcsec2)] = 30
    f7_mag_per_arcsec2[np.isnan(f7_mag_per_arcsec2)] = 30
    low_surf_b.append(f3_mag_per_arcsec2)
    high_surf_b.append(f7_mag_per_arcsec2)


#%%
with plt.style.context("dark_background"):
    with plt.rc_context(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "font.size": 7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "ytick.right": True,
            "xtick.top": True,
        }
    ):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4, 8), dpi=200)
        # plt.subplots_adjust(hspace=0, wspace=0)

        ax[0].text(
            0.05,
            0.95,
            r"$\mathrm{{high-SFE}}$"
            "\n"
            r"$\mathrm{{ t = {:.1f} \: Myr}}$"
            "\n"
            r"$\mathrm{{z = {:.1f} }}$".format(high_times[0], high_redshifts[0]),
            ha="left",
            va="top",
            color="white",
            transform=ax[0].transAxes,
            fontsize=9,
        )

        f7_render = ax[0].imshow(
            high_jansky[0],
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            norm=LogNorm(6e-8, 1e-5),
            cmap="inferno",
            origin="lower",
            interpolation="gaussian",
        )
        ax[0].set(xlim=row_lims[0], ylim=row_lims[0], xticklabels=[], yticklabels=[])

        scale = patches.Rectangle(
            xy=(row_lims[0][0] * 0.80, row_lims[0][0] * 0.80),
            width=row_lims[0][1] * 0.5,
            height=0.025 * row_lims[0][1],
            linewidth=0,
            edgecolor="white",
            facecolor="white",
        )
        ax[0].add_patch(scale)  # galaxy_angular_width
        ax[0].text(
            row_lims[0][0] * 0.55,
            row_lims[0][0] * 0.87,
            r"$\mathrm{{{:.0f} \: pc}}$".format(row_lims[0][1] * 0.5),
            ha="center",
            va="center",
            color="white",
            fontproperties=leg_font,
        )
        ax[0].text(
            row_lims[0][0] * 0.55,
            row_lims[0][0] * 0.72,
            "${:.3f}$ "
            "$\mathrm{{arcsec}}$".format(
                ((row_lims[0][1] * 0.5) / width) * galaxy_angular_width
            ),
            ha="center",
            va="center",
            color="white",
            fontproperties=leg_font,
        )
        with plt.rc_context(
            {
                "font.family": "serif",
                "mathtext.fontset": "cm",
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "font.size": 8,
            }
        ):
            cbar_ax = ax[0].inset_axes([0.50, 0.88, 0.45, 0.04])

            cbar = fig.colorbar(f7_render, cax=cbar_ax, pad=0, orientation="horizontal")

            cbar_label = (
                r"$I_\lambda (\mathrm{\lambda = 1500 \: \AA \:} )$"
                r"$\mathrm{\left(Jy \: arcsec^{-2}\right)} $"
            )
            cbar.set_label(
                label=cbar_label,
                # fontsize=10,
                size=8
                # fontproperties=leg_font,
            )
            cbar.ax.xaxis.set_ticks_position("bottom")
            cbar.ax.xaxis.set_label_position("top")
            cbar_ax.tick_params(axis="both", direction="in", which="both")
            cbar.ax.xaxis.set_tick_params(pad=3)

        f7_zoom = ax[0].inset_axes([1, 0, 1, 1])
        f7_mag = f7_zoom.imshow(
            high_surf_b[0],
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            vmin=19,
            vmax=30,
            cmap="cubehelix_r",
            origin="lower",
            interpolation="gaussian",
        )
        f7_zoom.set(xlim=(-25, 25), ylim=(-15, 35), xticklabels=[], yticklabels=[])
        scale = patches.Rectangle(
            xy=(-20, 20),
            width=10,
            height=0.025 * 25,
            linewidth=0,
            edgecolor="white",
            facecolor="white",
            alpha=0.6,
        )
        f7_zoom.add_patch(scale)
        f7_zoom.text(
            -15,
            18,
            r"$\mathrm{{{:.0f} \: pc}}$".format(10),
            ha="center",
            va="center",
            color="white",
            fontproperties=leg_font,
            alpha=0.6,
        )
        mark_inset(
            ax[0], f7_zoom, loc1=3, loc2=3, edgecolor="white", ls="--", alpha=0.4
        )
        with plt.rc_context(
            {
                "font.family": "serif",
                "mathtext.fontset": "cm",
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "font.size": 8,
            }
        ):

            cbar_ax = f7_zoom.inset_axes([0.05, 0.08, 0.40, 0.04])
            cbar = fig.colorbar(f7_mag, cax=cbar_ax, pad=0, orientation="horizontal")
            cbar.ax.locator_params(nbins=5)
            cbar_label = r"$\mathrm{\mu_{AB} \: (mag \: arcsec^{-2})}$"
            cbar.set_label(label=cbar_label, size=8)
            cbar.ax.xaxis.set_ticks_position("bottom")
            cbar.ax.xaxis.set_label_position("top")
            cbar_ax.tick_params(direction="in")
            cbar.ax.xaxis.set_tick_params(pad=3)

        f7_zoom2_1 = f7_zoom.inset_axes([1, 0.5, 0.5, 0.5])
        f7_zoom2_1.imshow(
            high_surf_b[0],
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            vmin=19,
            vmax=30,
            cmap="cubehelix_r",
            origin="lower",
            interpolation="gaussian",
        )
        f7_zoom2_1.set(xlim=(-20, -10), ylim=(22, 32), xticklabels=[], yticklabels=[])
        mark_inset(
            f7_zoom, f7_zoom2_1, loc1=2, loc2=2, edgecolor="white", ls="--", alpha=0.4
        )

        f7_zoom2_2 = f7_zoom.inset_axes([1, 0, 0.5, 0.5])
        f7_zoom2_2.imshow(
            high_surf_b[0],
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            vmin=19,
            vmax=30,
            cmap="cubehelix_r",
            origin="lower",
            interpolation="gaussian",
        )
        f7_zoom2_2.set(xlim=(0, 10), ylim=(10, 20), xticklabels=[], yticklabels=[])
        mark_inset(
            f7_zoom, f7_zoom2_2, loc1=3, loc2=3, edgecolor="white", ls="--", alpha=0.4
        )
        # =============================================================================
        #
        # =============================================================================
        ax[1].text(
            0.05,
            0.95,
            r"$\mathrm{{low-SFE}}$"
            "\n"
            r"$\mathrm{{ t = {:.1f} \: Myr}}$"
            "\n"
            r"$\mathrm{{z = {:.1f} }}$".format(low_times[0], low_redshifts[0]),
            ha="left",
            va="top",
            color="white",
            transform=ax[1].transAxes,
            fontsize=9,
        )

        f3_render = ax[1].imshow(
            low_jansky[0],
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            norm=LogNorm(6e-8, 1e-5),
            cmap="inferno",
            origin="lower",
            interpolation="gaussian",
        )
        ax[1].set(xlim=row_lims[0], ylim=row_lims[0], xticklabels=[], yticklabels=[])

        scale = patches.Rectangle(
            xy=(row_lims[0][0] * 0.80, row_lims[0][0] * 0.80),
            width=row_lims[0][1] * 0.5,
            height=0.025 * row_lims[0][1],
            linewidth=0,
            edgecolor="white",
            facecolor="white",
        )
        ax[1].add_patch(scale)
        ax[1].text(
            row_lims[0][0] * 0.55,
            row_lims[0][0] * 0.87,
            r"$\mathrm{{{:.0f} \: pc}}$".format(row_lims[0][1] * 0.5),
            ha="center",
            va="center",
            color="white",
            fontproperties=leg_font,
        )

        f3_zoom = ax[1].inset_axes([1, 0, 1, 1])
        f3_mag = f3_zoom.imshow(
            low_surf_b[0],
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            vmin=19,
            vmax=30,
            cmap="cubehelix_r",
            origin="lower",
            interpolation="gaussian",
        )
        f3_zoom.set(xlim=(-10, 40), ylim=(-50, 0), xticklabels=[], yticklabels=[])
        scale = patches.Rectangle(
            xy=(0, -22),
            width=10,
            height=0.025 * 25,
            linewidth=0,
            edgecolor="white",
            facecolor="white",
            alpha=0.6,
        )
        f3_zoom.add_patch(scale)
        f3_zoom.text(
            5,
            -24,
            r"$\mathrm{{{:.0f} \: pc}}$".format(10),
            ha="center",
            va="center",
            color="white",
            fontproperties=leg_font,
            alpha=0.6,
        )
        mark_inset(
            ax[1], f3_zoom, loc1=2, loc2=2, edgecolor="white", ls="--", alpha=0.4
        )

        f3_zoom2_1 = f3_zoom.inset_axes([1, 0, 0.5, 0.5])
        f3_zoom2_1.imshow(
            low_surf_b[0],
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            vmin=19,
            vmax=30,
            cmap="cubehelix_r",
            origin="lower",
            interpolation="gaussian",
        )
        f3_zoom2_1.set(xlim=(12, 22), ylim=(-46, -36), xticklabels=[], yticklabels=[])
        mark_inset(
            f3_zoom, f3_zoom2_1, loc1=3, loc2=3, edgecolor="white", ls="--", alpha=0.4
        )

        f3_zoom2_2 = f3_zoom.inset_axes([1, 0.5, 0.5, 0.5])
        f3_zoom2_2.imshow(
            low_surf_b[0],
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            vmin=19,
            vmax=30,
            cmap="cubehelix_r",
            origin="lower",
            interpolation="gaussian",
        )
        f3_zoom2_2.set(xlim=(0, 10), ylim=(-20, -10), xticklabels=[], yticklabels=[])
        mark_inset(
            f3_zoom, f3_zoom2_2, loc1=2, loc2=2, edgecolor="white", ls="--", alpha=0.4
        )

        # =============================================================================
        # render late stage evo of plots
        # =============================================================================
        f7_late_ax = ax[1].inset_axes([0, -1.0, 1.0, 1.0])
        f3_late_ax = ax[1].inset_axes([1.5, -1.0, 1.0, 1.0])

        f7_late_render = f7_late_ax.imshow(
            high_jansky[1],
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            norm=LogNorm(2e-8, 5e-6),  # 8e7, 8e10
            cmap="inferno",
            origin="lower",
            interpolation="gaussian",
        )
        f7_late_ax.set(
            xlim=row_lims[1], ylim=row_lims[1], xticklabels=[], yticklabels=[]
        )
        scale = patches.Rectangle(
            xy=(row_lims[1][0] * 0.80, row_lims[1][0] * 0.80),
            width=row_lims[1][1] * 0.5,
            height=0.025 * row_lims[1][1],
            linewidth=0,
            edgecolor="white",
            facecolor="white",
        )
        f7_late_ax.add_patch(scale)
        f7_late_ax.text(
            row_lims[1][0] * 0.55,
            row_lims[1][0] * 0.87,
            r"$\mathrm{{{:.0f} \: pc}}$".format(row_lims[1][1] * 0.5),
            ha="center",
            va="center",
            color="white",
            fontproperties=leg_font,
        )
        f3_late_render = f3_late_ax.imshow(
            low_jansky[1],
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            norm=LogNorm(2e-8, 5e-6),
            cmap="inferno",
            origin="lower",
            interpolation="gaussian",
        )
        f3_late_ax.set(
            xlim=row_lims[1], ylim=row_lims[1], xticklabels=[], yticklabels=[]
        )

        f7_late_ax_zoom = f7_late_ax.inset_axes([1, 0.5, 0.5, 0.5])
        f3_late_ax_zoom = f3_late_ax.inset_axes([-0.5, 0, 0.5, 0.5])
        f7_late_ax_zoom.imshow(
            high_jansky[1],
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            norm=LogNorm(2e-8, 5e-6),
            cmap="inferno",
            origin="lower",
            interpolation="gaussian",
        )
        f3_late_ax_zoom.imshow(
            low_jansky[1],
            extent=[-proj_r, proj_r, -proj_r, proj_r],
            norm=LogNorm(2e-8, 5e-6),
            cmap="inferno",
            origin="lower",
            interpolation="gaussian",
        )
        f7_late_ax_zoom.set(
            xlim=(-25, 25), ylim=(-25, 25), xticklabels=[], yticklabels=[]
        )
        f3_late_ax_zoom.set(
            xlim=(-25, 25), ylim=(-10, 40), xticklabels=[], yticklabels=[]
        )
        mark_inset(
            f7_late_ax,
            f7_late_ax_zoom,
            loc1=2,
            loc2=2,
            edgecolor="white",
            ls="--",
            alpha=0.4,
        )
        mark_inset(
            f3_late_ax,
            f3_late_ax_zoom,
            loc1=4,
            loc2=4,
            edgecolor="white",
            ls="--",
            alpha=0.4,
        )

        f7_late_ax.text(
            0.05,
            0.95,
            r"$\mathrm{{high-SFE}}$"
            "\n"
            r"$\mathrm{{ t = {:.1f} \: Myr}}$"
            "\n"
            r"$\mathrm{{z = {:.1f} }}$".format(high_times[1], high_redshifts[1]),
            ha="left",
            va="top",
            color="white",
            transform=f7_late_ax.transAxes,
            fontsize=9,
        )
        f3_late_ax.text(
            0.05,
            0.95,
            r"$\mathrm{{low-SFE}}$"
            "\n"
            r"$\mathrm{{ t = {:.1f} \: Myr}}$"
            "\n"
            r"$\mathrm{{z = {:.1f} }}$".format(low_times[1], low_redshifts[1]),
            ha="left",
            va="top",
            color="white",
            transform=f3_late_ax.transAxes,
            fontsize=9,
        )

        with plt.rc_context(
            {
                "font.family": "serif",
                "mathtext.fontset": "cm",
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "font.size": 8,
            }
        ):
            cbar_ax = f7_late_ax.inset_axes([0.50, 0.12, 0.45, 0.04])
            cbar = fig.colorbar(
                f7_late_render, cax=cbar_ax, pad=0, orientation="horizontal"
            )
            cbar_label = (
                r"$I_\lambda (\mathrm{\lambda = 1500 \: \AA \:} )$"
                r"$\mathrm{\left(Jy \: arcsec^{-2}\right)} $"
            )
            cbar.set_label(
                label=cbar_label,
                # fontsize=10,
                size=8
                # fontproperties=leg_font,
            )
            cbar.ax.xaxis.set_ticks_position("bottom")
            cbar.ax.xaxis.set_label_position("top")
            cbar_ax.tick_params(axis="both", direction="in", which="both")
            cbar.ax.xaxis.set_tick_params(pad=3)

        ax[0].tick_params(axis="both", which="both", length=0)
        ax[1].tick_params(axis="both", which="both", length=0)
        f7_zoom.tick_params(axis="both", which="both", length=0)
        f3_zoom.tick_params(axis="both", which="both", length=0)

        f7_zoom2_1.tick_params(axis="both", which="both", direction="in")
        f7_zoom2_2.tick_params(axis="both", which="both", direction="in")

        f3_zoom2_1.tick_params(axis="both", which="both", direction="in")
        f3_zoom2_2.tick_params(axis="both", which="both", direction="in")

        f3_late_ax.tick_params(axis="both", which="both", length=0)
        f7_late_ax.tick_params(axis="both", which="both", length=0)

        f3_late_ax_zoom.tick_params(axis="both", which="both", direction="in")
        f7_late_ax_zoom.tick_params(axis="both", which="both", direction="in")

        plt.subplots_adjust(hspace=-0.0002, wspace=0)


plt.savefig(
    os.path.expanduser(
        (
            "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/lowres/"
            "jansky_ABmag.png"
        )
    ),
    dpi=250,
    bbox_inches="tight",
    pad_inches=0.00,
)
