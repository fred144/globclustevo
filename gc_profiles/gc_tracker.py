import sys

sys.path.append("..")
import numpy as np
import os
import glob
from modules.macros import filter_snapshots
from modules.luminosity.lum_functions import unpack_pop_ii_data
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

strt = 404
end = 404
step = 1

halo_data_directory = r"../halo_data/fs07_refine/fof_best"
pop2_data_directory = r"../particle_data/pop_2_data/fs07_refine"
pop2 = filter_snapshots(pop2_data_directory, strt, end, step)
halo_ds = filter_snapshots(halo_data_directory, strt, end, step)
#%%
# TODO: make a an independent profiler and lum map three panel. etc.
for p2, ds in zip(pop2, halo_ds):
    output_num = int(ds.split("/")[-1].split("_")[-1])

    _, scaled_stellar_lums, masses, ages, t_myr, ids = unpack_pop_ii_data(
        p2, return_ids=True
    )

    halos = sorted(glob.glob(os.path.join(ds, "gc*.txt")))
    for halo in halos:

        halo_data = np.loadtxt(halo)

        if 33387 in halo_data[:, 0]:  # the star id number was found via trial and error
            halo_num = int(halo.split("/")[-1].split("_")[-1].split(".")[0])
            gc_lum_mask = np.isin(ids, halo_data[:, 0])
            gc_lums = scaled_stellar_lums[gc_lum_mask]
            x = halo_data[:, 1]
            y = halo_data[:, 2]
            z = halo_data[:, 3]

            xy_lums, _, _ = np.histogram2d(
                x,
                y,
                bins=200,
                weights=gc_lums,
                normed=False,
                range=[[-10, 10], [-10, 10]],
            )

            xz_lums, _, _ = np.histogram2d(
                x,
                z,
                bins=200,
                weights=gc_lums,
                normed=False,
                range=[[-10, 10], [-10, 10]],
            )

            yz_lums, _, _ = np.histogram2d(
                y,
                z,
                bins=200,
                weights=gc_lums,
                normed=False,
                range=[[-10, 10], [-10, 10]],
            )

            xy_lums = xy_lums.T
            xz_lums = xy_lums.T
            yz_lums = yz_lums.T

            with plt.style.context("dark_background"):
                fig, ax = plt.subplots(
                    nrows=2,
                    ncols=3,
                    sharex="col",
                    sharey="row",
                    figsize=(12, 10),
                    dpi=300,
                    facecolor=cm.Greys_r(0),
                )
                xy = ax[0, 0].imshow(
                    xy_lums,
                    cmap="inferno",
                    interpolation="gaussian",
                    origin="lower",
                    extent=[-10, 10, -10, 10],
                    norm=LogNorm(),
                )

                xz = ax[0, 1].imshow(
                    xz_lums,
                    cmap="inferno",
                    interpolation="gaussian",
                    origin="lower",
                    extent=[-10, 10, -10, 10],
                    norm=LogNorm(),
                )

                yz = ax[0, 2].imshow(
                    yz_lums,
                    cmap="inferno",
                    interpolation="gaussian",
                    origin="lower",
                    extent=[-10, 10, -10, 10],
                    norm=LogNorm(),
                )

                fig.subplots_adjust(wspace=0, hspace=0, bottom=0.1)
                cbar_ax = fig.add_axes([0.125, 0.3775, 0.775, 0.004])
                cbar = fig.colorbar(xy, cax=cbar_ax, orientation="horizontal", pad=0)

            print(output_num, halo_num)
            break
        else:
            pass
