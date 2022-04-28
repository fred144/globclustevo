import sys

sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/cosmology-clean/lib/python3.7/site-packages"
)

import warnings
import os
import yt
from yt.funcs import mylog
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# yt.enable_parallelism()
mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


pop2_data_directory = r"../pop_2_data/fs07_refine"
halo_data_directory = r"../halo_data/fs07_refine/fof"
# ---------------------------------DT2 Paths------------------------------------
# lustre data path
# pop2_data_directory = os.path.expanduser(
#     "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine"
# )
# # save path
# sequence_folder = "newcmap_gas_projected_density_z"
# parent_folder = "/homes/fgarcia4/analysis/cluster_evolution/sequences/new_refine"
# newpath = parent_folder + "/" + sequence_folder
# if not os.path.exists(newpath):
#     print("# Creating new sequence directory", newpath)
#     os.makedirs(newpath)
# =============================================================================


def filter_snapshots(folder_path, start_snap: int, end_snap: int, sampling=1):
    r"""Given a directory of outputs, return a list of absolute file
    paths given a range of snapshot values. Enables discrete selection
    of time range based on snapshot number.

    """
    strt_string = str(start_snap).zfill(5)
    end_string = str(end_snap).zfill(5)

    files = sorted(os.listdir(folder_path))

    strt_idx = [i for i, s in enumerate(files) if strt_string in s][0]
    end_idx = [i for i, s in enumerate(files) if end_string in s][0]

    filtered_files = files[strt_idx : end_idx + 1 : sampling]

    abs_paths = [os.path.join(folder_path, file) for file in filtered_files]

    return abs_paths


# =============================================================================
pop2_data_set = filter_snapshots(pop2_data_directory, 113, 808, 10)
halo_data_directory = filter_snapshots(halo_data_directory, 113, 808, 10)
# =============================================================================


def get_cluster(
    pos_3d,
    ctr_at,
    end_radius,
    masses,
    ages,
):
    r"""
    Mask out a cluster based on its center, radius.
    Returns luminosities masses and transformed coordinates,
    the ctr_at becomes the origin.

    Can take 2d or 3d coordinates.
    """
    # calculate distances and create mask to distances inside end_radius
    distances = np.sqrt(np.sum(np.square(pos_3d - ctr_at), axis=1))
    mask = distances <= end_radius
    # apply mask
    masked_masses = masses[mask]
    masked_ages = ages[mask]

    # characteristic age is the age most common, and the mass inside the end_radius
    char_age = float(st.mode(masked_ages)[0])
    glob_mass = np.sum(masked_masses)  # Msun

    return char_age, glob_mass


def mass_function(masses, t_sim, num_bins, m_vir=None):
    """Plot mass function for truncation and core masses"""
    bins = np.geomspace(np.min(masses), np.max(masses), num=num_bins, endpoint=True)
    count, bin_edges = np.histogram(masses, bins=bins)
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]
    bin_ctrs = 0.5 * (left_edges + right_edges)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    # ax.errorbar(bin_ctrs, count, fmt="-o")
    ax.hist(
        masses,
        bins,
        histtype="step",
        hatch="\\",
        linewidth=4,
        alpha=1,
        label=r"$M_{(R < uniform)}$",
    )

    if m_vir is not None:

        count, bin_edges = np.histogram(m_vir, bins=bins)
        right_edges = bin_edges[1:]
        left_edges = bin_edges[:-1]
        bin_ctrs = 0.5 * (left_edges + right_edges)

        # ax.errorbar(bin_ctrs, count, fmt="-o")
        ax.hist(
            m_vir,
            bins,
            histtype="step",
            hatch="/",
            linestyle="--",
            linewidth=4,
            alpha=1,
            label=r"$M_{vir}$",
        )

    ax.set_title(r"$t_{{sim}}$ = {:.2f} Myr".format(t_sim), fontsize=18)
    ax.set_xlabel(r"$ M \left( M_{\odot} \right ) $", fontsize=18)
    ax.set_ylabel(r"dN / d$\log \left( M \right)$", fontsize=18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xlim(1, 2e5)
    ax.set_ylim(bottom=1, top=150)

    ax.legend(fontsize=18, loc="upper left")


def bubble_plot(masses, vir_radii, ages, current_time):
    vir_diameter = vir_radii * 2

    colors = np.random.uniform(size=masses.size)
    norm = 5
    # map to differnt sizes for better plotting
    vir_diameter_per_size = (500 * vir_diameter) / norm

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

    scatter = ax.scatter(
        ages,
        masses,
        c="black",
        s=vir_diameter_per_size,
        cmap="Set3",
        alpha=0.2,
        linewidths=2,
    )

    # remap to actual sizes for legend
    legend_properties = dict(
        prop="sizes",
        num=[0.50, 1.0, 1.50],
        color="black",
        fmt=" {x:.2f}",
        func=lambda d: (d * norm) / 500,
    )
    legend = ax.legend(
        *scatter.legend_elements(**legend_properties),
        loc="lower left",
        title="$d_{vir}$ (pc)",
        title_fontsize=18,
        fontsize=15,
    )
    plt.grid(visible=True)

    ax.set_title(r"$t_{{sim}}$ = {:.2f} Myr".format(t_sim), fontsize=18)
    ax.set_ylabel(r"GC $M_{vir}$ ($M_{\odot}$)", fontsize=18)
    ax.set_xlabel(r"Formation Time (Myr)", fontsize=18)
    ax.set_xlim(300, 500)
    ax.set_ylim(10, 1e6)
    ax.set_yscale("log")


cell_fields = [
    "Density",
    "x-velocity",
    "y-velocity",
    "z-velocity",
    "Pressure",
    "Metallicity",
    "dark_matter_density",
    "xHI",
    "xHII",
    "xHeII",
    "xHeIII",
]
epf = [
    ("particle_family", "b"),
    ("particle_tag", "b"),
    ("particle_birth_epoch", "d"),
    ("particle_metallicity", "d"),
]


for pop, hc in zip(pop2_data_set, halo_data_directory):
    print("# Reading Pop II", pop)
    print("# Reading Catalogue", hc)
    pop2_data = np.loadtxt(pop)
    x_pop2 = pop2_data[:, 2]  # pc
    y_pop2 = pop2_data[:, 3]  # pc
    z_pop2 = pop2_data[:, 4]  # pc
    ctr_at = pop2_data[5:8, 6]  # pc
    ages = pop2_data[:, 1]  # Myr
    ages[ages < 1] = 1  # set minimum age
    pop2_masses = pop2_data[:, 5]  # Msun

    t_sim = pop2_data[0, 6]  # Myr

    # get the hdf5 catalogue inside each folder
    halo_cat = yt.load(os.path.join(hc, os.listdir(hc)[0]))
    halo_data = halo_cat.all_data()
    # get centers of halos
    # specific to FOF halo finder output
    x_halos = np.array(halo_data["all", "particle_position_x"].to("pc")) - ctr_at[0]
    y_halos = np.array(halo_data["all", "particle_position_y"].to("pc")) - ctr_at[1]
    z_halos = np.array(halo_data["all", "particle_position_z"].to("pc")) - ctr_at[2]

    vir_radius = np.array(halo_data["all", "virial_radius"].to("pc"))
    halos = np.vstack((x_halos, y_halos, z_halos)).T  # pc

    gc_ages = []
    gc_vir_mass = []
    gc_uniform_rad_mass = []
    for halo_ctr, r_vir in zip(halos, vir_radius):
        age, mass = get_cluster(pop2_data[:, 2:5], halo_ctr, r_vir, pop2_masses, ages)
        _, uni_mass = get_cluster(pop2_data[:, 2:5], halo_ctr, 5, pop2_masses, ages)
        gc_uniform_rad_mass.append(uni_mass)
        gc_ages.append(age)
        gc_vir_mass.append(mass)
    gc_ages = np.array(gc_ages)
    gc_vir_mass = np.array(gc_vir_mass)
    gc_uniform_rad_mass = np.array(gc_uniform_rad_mass)
    # plt.scatter(z_pop2, y_pop2, s=0.1, alpha=0.1)
    # plt.scatter(z_halos, y_halos, s=0.2, color="red")
    # plt.xlim(-200, 200)
    # plt.ylim(-200, 200)
    plt.figure(figsize=(8, 8), dpi=400)
    # mass_function(
    #     masses=gc_uniform_rad_mass, t_sim=t_sim, num_bins=10, m_vir=gc_vir_mass
    # )
    birth_time = t_sim - gc_ages
    bubble_plot(
        masses=gc_vir_mass, vir_radii=vir_radius, ages=birth_time, current_time=t_sim
    )
#%%
