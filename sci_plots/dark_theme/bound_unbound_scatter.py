import sys

sys.path.append("../../")
import numpy as np
import os
from modules.macros import filter_snapshots, characterisitc_mass, sci_notation
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from modules.match_t_sims import find_matching_time, get_snapshots
import matplotlib as mpl

# runsavepath = "../rendering/luminosity/fs07_refine/gc_tracking"
# if not os.path.exists(runsavepath):
#     print("# Creating new sequence directory", runsavepath)
#     os.makedirs(runsavepath)


f7_strt = 113
f7_end = 1196
f3_strt = 154
f3_end = 1368
step = 1

f7_pop2 = filter_snapshots(
    r"../../particle_data/pop_2_data/fs07_refine", f7_strt, f7_end, step
)
f7_halo_ds = filter_snapshots(
    r"../../halo_data/fs07_refine/fof_best", f7_strt, f7_end, step
)
# matched snapshots

f3_pop2 = filter_snapshots(
    r"../../particle_data/pop_2_data/fs035_ms10", f3_strt, f3_end, step
)
f3_halo = filter_snapshots(
    r"../../halo_data/fs035_ms10/fof_best", f3_strt, f3_end, step
)

f3_pop2_matched, f3_matched_nums = find_matching_time(
    sequence=f7_pop2, look_up_sequence=f3_pop2
)
f3_halo_matched = get_snapshots(snapshot_file_list=f3_halo, get_list=f3_matched_nums)

# sampple idxs
prof_start = 1000
prof_end = 1001
prof_step = 1

fs070_p2 = f7_pop2[prof_start:prof_end:prof_step]
fs070_ds = f7_halo_ds[prof_start:prof_end:prof_step]
fs035_p2 = f3_pop2_matched[prof_start:prof_end:prof_step]
fs035_ds = f3_halo_matched[prof_start:prof_end:prof_step]

# cmap = cm.get_cmap("Set1")
# cmap = cmap(np.linspace(0, 1, 9))
# bound_clr = cmap[0]
# field_clr = cmap[1]

cmap = cm.get_cmap("Set3")
cmap = cmap(np.linspace(0, 1, 11))
bound_clr = "red"
field_clr = "grey"
mpl.rcParams["hatch.linewidth"] = 0.5
#%%
for f7_p2, f7_ds, f3_p2, f3_ds in zip(fs070_p2, fs070_ds, fs035_p2, fs035_ds):
    f7_t_myr, f7_redshift = np.loadtxt(f7_p2, max_rows=2)[0:2, 6]
    f3_t_myr, f3_redshift = np.loadtxt(f3_p2, max_rows=2)[0:2, 6]

    f7_field_stars = np.loadtxt(os.path.join(f7_ds, "field_stars.txt"))
    f3_field_stars = np.loadtxt(os.path.join(f3_ds, "field_stars.txt"))

    f7_bound_stars = np.loadtxt(os.path.join(f7_ds, "bound_stars.txt"))
    f3_bound_stars = np.loadtxt(os.path.join(f3_ds, "bound_stars.txt"))

    f7_x_field = f7_field_stars[:, 3]
    f7_y_field = f7_field_stars[:, 4]
    f7_z_field = f7_field_stars[:, 5]

    f3_x_field = f3_field_stars[:, 3]
    f3_y_field = f3_field_stars[:, 4]
    f3_z_field = f3_field_stars[:, 5]

    f7_x_bound = f7_bound_stars[:, 3]
    f7_y_bound = f7_bound_stars[:, 4]
    f7_z_bound = f7_bound_stars[:, 5]

    f3_x_bound = f3_bound_stars[:, 3]
    f3_y_bound = f3_bound_stars[:, 4]
    f3_z_bound = f3_bound_stars[:, 5]

    f7_field_masses = f7_field_stars[:, 6]
    f7_field_be = f7_t_myr - f7_field_stars[:, 1]
    f7_bound_masses = f7_bound_stars[:, 6]
    f7_bound_be = f7_t_myr - f7_bound_stars[:, 1]

    f3_field_masses = f3_field_stars[:, 6]
    f3_field_be = f3_t_myr - f3_field_stars[:, 1]
    f3_bound_masses = f3_bound_stars[:, 6]
    f3_bound_be = f3_t_myr - f3_bound_stars[:, 1]

    with plt.rc_context(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.size": 12,
        }
    ):
        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(
                nrows=1,
                ncols=2,
                sharex=True,
                sharey=True,
                figsize=(8, 4),
                dpi=500,
            )

            ax[0].scatter(f7_x_field, f7_y_field, color=field_clr, s=0.5, alpha=0.05)
            ax[0].scatter(f7_x_bound, f7_y_bound, color=bound_clr, s=0.5, alpha=0.05)

            ax[1].scatter(f3_x_field, f3_y_field, color=field_clr, s=0.5, alpha=0.05)
            ax[1].scatter(f3_x_bound, f3_y_bound, color=bound_clr, s=0.5, alpha=0.05)

            ax[1].set_xlim(-200, 200)
            ax[1].set_ylim(-200, 200)

            # add histogram of ages
            ax_f7 = ax[0].inset_axes([0.12, 0.12, 0.30, 0.30])
            ax_f7.patch.set_alpha(0.5)
            bins = np.linspace(300, 619, 25)

            # f7_field_be, bin_edges = np.histogram(
            #     f7_field_be,
            #     bins=bins,
            #     weights=f7_field_masses,
            # )
            # right_edges = bin_edges[1:]
            # left_edges = bin_edges[:-1]
            # bin_centers = 0.5 * (left_edges + right_edges)

            # ax_f7.plot(
            #     bin_centers,
            #     f7_field_be,
            #     linewidth=0.5,
            #     alpha=0.8,
            #     color=field_clr,
            #     drawstyle="steps-mid",
            # )

            # f7_bound_be, bin_edges = np.histogram(
            #     f7_bound_be,
            #     bins=bins,
            #     weights=f7_bound_masses,
            # )
            # ax_f7.plot(
            #     bin_centers,
            #     f7_bound_be,
            #     linewidth=0.5,
            #     alpha=0.8,
            #     color=bound_clr,
            #     drawstyle="steps-mid",
            # )

            # ax_f7.hist(
            #     [f7_bound_be, f7_field_be],
            #     bins,
            #     weights=[f7_bound_masses, f7_field_masses],
            #     alpha=1,
            #     color=[bound_clr, field_clr],
            #     # edgecolor="white",
            #     linewidth=0.5,
            #     histtype="step",
            #     hatch="/////"
            #     # stacked=True,
            # )
            ax_f7.hist(
                f7_field_be,
                bins,
                weights=f7_field_masses,
                alpha=1,
                color=field_clr,
                # edgecolor="white",
                linewidth=0.5,
                histtype="step",
                hatch="\\\\\\\\\\\\\\",
                # stacked=True,
            )

            ax_f7.hist(
                f7_bound_be,
                bins,
                weights=f7_bound_masses,
                alpha=1,
                color=bound_clr,
                # edgecolor="white",
                linewidth=0.5,
                histtype="step",
                hatch="//////",
                # stacked=True,
            )

            # ax_f7.hist(
            #     f7_bound_be,
            #     bins,
            #     weights=f7_bound_masses,
            #     alpha=0.6,
            #     color=bound_clr,
            #     # edgecolor="white",
            #     linewidth=0.5,
            # )
            # ax_f7.axvline(x=f7_t_myr, ls="--", color="black", lw=1)
            ax_f7.tick_params(labelsize=6)
            ax_f7.set_xlabel(
                "$\mathrm{Star \: Birth \: (Myr)}$",
                fontsize=7,
                labelpad=0,
            )
            ax_f7.set_ylabel(
                r"$\mathrm{Mass \: (M_{\odot})}$",
                fontsize=7,
                labelpad=0,
            )
            # ax_f7.text(
            #     f7_t_myr + 10,
            #     2e3,
            #     r"$\mathrm{t_{sim}}$",
            #     va="center",
            #     rotation=270,
            #     fontsize=8,
            # )
            ax_f7.set_yscale("log")
            # ax_f7.set_xlim("log")
            ax_f7.set_xlim(300, 650)
            ax_f7.set_ylim(1, 5e5)
            ax_f7.tick_params(axis="both", direction="in", which="both")

            ax_f3 = ax[1].inset_axes([0.12, 0.12, 0.30, 0.30])
            ax_f3.patch.set_alpha(0.5)
            # ax_f3.hist(
            #     f3_field_be,
            #     bins,
            #     weights=f3_field_masses,
            #     alpha=1,
            #     color=field_clr,
            #     edgecolor="white",
            #     linewidth=0.5,
            # )
            # ax_f3.hist(
            #     f3_bound_be,
            #     bins,
            #     weights=f3_bound_masses,
            #     alpha=0.6,
            #     color=bound_clr,
            #     edgecolor="white",
            #     linewidth=0.5,
            # )
            ax_f3.hist(
                f3_field_be,
                bins,
                weights=f3_field_masses,
                alpha=1,
                color=field_clr,
                # edgecolor="white",
                linewidth=0.5,
                histtype="step",
                hatch="\\\\\\\\\\\\\\",
                # stacked=True,
            )
            ax_f3.hist(
                f3_bound_be,
                bins,
                weights=f3_bound_masses,
                alpha=1,
                color=bound_clr,
                # edgecolor="white",
                linewidth=0.5,
                histtype="step",
                hatch="//////",
                # stacked=True,
            )

            # ax_f3.axvline(x=f3_t_myr, ls="--", color="black", lw=1)
            ax_f3.tick_params(labelsize=6)
            ax_f3.set_xlabel(
                "$\mathrm{Star \: Birth \: (Myr)}$",
                fontsize=7,
                labelpad=0,
            )
            ax_f3.set_ylabel(
                r"$\mathrm{Mass \: (M_{\odot})}$",
                fontsize=7,
                labelpad=0,
            )
            # ax_f3.text(
            #     f3_t_myr + 10,
            #     2e3,
            #     r"$\mathrm{t_{sim}}$",
            #     va="center",
            #     rotation=270,
            #     fontsize=8,
            # )
            ax_f3.set_yscale("log")
            # ax_f3.set_xlim("log")
            ax_f3.set_xlim(300, 650)
            ax_f3.set_ylim(1, 5e5)
            ax_f3.tick_params(axis="both", direction="in", which="both")

            # add efficiency labels
            props = dict(
                boxstyle="round",
                facecolor="black",
                alpha=0.5,
                linewidth=0.8,
            )
            ax[0].text(
                0.75,
                0.95,
                "$f_{*} = 0.70$",
                transform=ax[0].transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=props,
            )
            ax[1].text(
                0.75,
                0.95,
                "$f_{*} = 0.35$",
                transform=ax[1].transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=props,
            )

            # manually make a legend
            bound = mlines.Line2D(
                [],
                [],
                color=bound_clr,
                ls="",
                marker="o",
                markersize=3,
                label=r"$\mathrm{Bound}$",
            )
            unbound = mlines.Line2D(
                [],
                [],
                color=field_clr,
                ls="",
                marker="o",
                markersize=3,
                label=r"$\mathrm{Field}$",
            )
            fig.legend(
                loc="lower right",
                title_fontsize=10,
                fontsize=10,
                handles=[bound, unbound],
                bbox_to_anchor=(0.90, 0.125),
                edgecolor="grey",
            )

            # add time and redshift
            props = dict(
                boxstyle="round",
                facecolor="black",
                alpha=0.5,
                linewidth=0.8,
                edgecolor="white",
            )
            ax[0].text(
                -180,
                180,
                (
                    r"$\mathrm{{t = {:.1f} \: Myr}}$" "\n" r"$\mathrm{{z = {:.1f} }}$"
                ).format(f7_t_myr, f7_redshift),
                fontsize=10,
                ha="left",
                va="top",
                color="white",
                bbox=props,
            )
            # ax[1].text(
            #     -180,
            #     180,
            #     (r"$\mathrm{{t = {:.2f} \: Myr}}$" "\n" r"$\mathrm{{z = {:.2f} }}$").format(
            #         f3_t_myr, f3_redshift
            #     ),
            #     size=12,
            #     ha="left",
            #     va="top",
            #     color="black",
            #     bbox=props,
            # )

            # add a top twin axis
            # top_ax = ax[0].twiny()
            # top_ax.set_xlabel("$\mathrm{100 \: pc}$", labelpad=10)
            # top_ax.set_xlim(-200, 200)
            # top_ax.axes.xaxis.set_ticklabels([])

            # add a scale bar
            rect = patches.Rectangle(
                xy=(50, -150),
                width=100,
                height=3,
                linewidth=1,
                edgecolor="black",
                facecolor="black",
                clip_on=False,
            )
            ax[0].text(
                0.68,
                0.11,
                "$\mathrm{100 \: pc}$",
                transform=ax[0].transAxes,
                fontsize=10,
                verticalalignment="top",
            )
            ax[0].add_patch(rect)

            # clean up edges
            ax[1].axes.xaxis.set_ticklabels([])
            ax[1].axes.yaxis.set_ticklabels([])

            ax[0].tick_params(
                axis="y", which="both", left=False, right=False, labelbottom=False
            )
            ax[0].tick_params(
                axis="x", which="both", top=False, bottom=False, labelbottom=False
            )
            ax[1].tick_params(
                axis="y", which="both", left=False, right=False, labelbottom=False
            )
            ax[1].tick_params(
                axis="x", which="both", top=False, bottom=False, labelbottom=False
            )

            plt.subplots_adjust(hspace=0, wspace=0)

        plt.savefig(
            os.path.expanduser(
                (
                    "../../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/darkmode/"
                    "bound_unbound_scatter.png"
                )
            ),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )

    # plt.savefig(
    #     os.path.expanduser(
    #         (
    #             "~/g_drive/Research/AstrophysicsSimulation/sci_plots/final/"
    #             "bound_unbound_scatter.png"
    #         )
    #     ),
    #     dpi=500,
    #     bbox_inches="tight",
    #     pad_inches=0.05,
    # )
