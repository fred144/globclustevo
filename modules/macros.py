import sys

sys.path.append("../..")
import numpy as np
import os
import scipy.stats as st
import yt
from yt.funcs import mylog
import warnings


# reduces some of the outputs when reading in yt data
mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def characterisitc_mass(x_coord, y_coord, masses, r_characteristic, counts=False):
    """
    get core mass or any characteristic mass enclosed by a characteristic radius;
    e.g. r_core
    """
    all_positions = np.vstack((x_coord, y_coord)).T
    distances = np.sqrt(np.sum(np.square(all_positions), axis=1))
    mask = distances <= r_characteristic
    char_mass = np.sum(masses[mask])
    if counts is True:
        return char_mass, np.size(masses[mask])
    else:
        return char_mass


def chi_squared(theory, data, sigma, num_params):

    chi2 = np.sum((theory - data) ** 2 / sigma**2)
    dof = np.size(data) - num_params
    reduced_chi_2 = chi2 / dof
    p_value = st.chi2.sf(chi2, dof)

    return p_value, reduced_chi_2


def code_age_to_myr(all_star_ages, hubble_const, unique_age=True, true_age=False):
    r"""
    Returns an array with unique birth epochs in Myr given
    raw_birth_epochs = ad['star', 'particle_birth_epoch']
    AND
    hubble = ds.hubble_constant
    Youngest is 0 Myr, all others are relative to the youngest.

    Relative ages option is currently yielding inconsistent results
    """
    cgs_yr = 3.1556926e7  # 1yr (in s)
    cgs_pc = 3.08567758e18  # pc (in cm)
    h_0 = hubble_const * 100  # hubble parameter (km/s/Mpc)
    h_0_invsec = h_0 * 1e5 / (1e6 * cgs_pc)  # hubble constant h [km/s Mpc-1]->[1/sec]
    h_0inv_yr = 1 / h_0_invsec / cgs_yr  # 1/h_0 [yr]

    if unique_age is True:
        # process to unique birth epochs only as well as sort them
        be_star_processed = np.array(sorted(list(set(all_star_ages.to_ndarray()))))
        star_age_myr = (be_star_processed * h_0inv_yr) / 1e6  # t=0 is the present
        relative_ages = star_age_myr - star_age_myr.min()

    else:
        all_stars = all_star_ages
        star_age_myr = all_stars * h_0inv_yr / 1e6  # t=0 is the present
        relative_ages = star_age_myr - star_age_myr.min()

    if true_age is True:
        return star_age_myr  # + 13.787 * 1e3
    else:
        return relative_ages  # t = 0 is the age of


def common_filter_snapshots(snapshots_to_filter, filter_list):
    r"""
    Given a list of snapshot file names (snapshots_to_filter),
    return only the snapshot file names in the given filter_list
    """
    filter_list = [f.split("_")[-1] for f in filter_list]
    common_items = []
    for file_name in snapshots_to_filter:
        if any(ext in file_name for ext in filter_list):
            common_items.append(file_name)

    return common_items


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


def succ_distance(current, previous):
    """
    # ctr_shift_thresh = 0.00060 #code length
    # ctr_shift_thresh =  0.000001 #code length
    # max_density_coords = []

    """
    dist = np.linalg.norm(np.array(current) - np.array(previous))
    return dist


def max_density_center_stable(
    ds, ad, loop_num, ctr_shift_thresh, slice_axis, width, prev_max_den
):
    r"""
    Center at the most dense point, but keep it relatively stable.

    """
    # centering using max density
    max_den = ad.argmax(("gas", "density"))
    max_density_coord = yt.YTArray(max_den).in_units("code_length")
    max_density_coord = np.array(max_density_coord)
    # keep center at density max
    if loop_num == 0:
        prev_max_den = max_density_coord
        distance = succ_distance(max_density_coord, prev_max_den)
        print("\n> distance b/w current and previously used max density:", distance)

    distance = succ_distance(max_density_coord, prev_max_den)

    if distance < ctr_shift_thresh:
        p = yt.ProjectionPlot(
            ds, slice_axis, "density", width=width, center=max_density_coord
        )
        x_pos = np.array(ad["star", "particle_position_x"]) - max_density_coord[0]
        y_pos = np.array(ad["star", "particle_position_y"]) - max_density_coord[1]
        z_pos = np.array(ad["star", "particle_position_z"]) - max_density_coord[2]
        # if the plot center migrates, annotate previous center
        # p.annotate_marker(
        #     max_density_coords[-1],
        #     marker="x",
        #     coord_system="data",
        #     plot_args={"color": "lime", "s": 30},)
        # appened new center to list
        print("> new plot centered at {}".format(max_density_coord))
        return p, max_density_coord, x_pos, y_pos, z_pos

    else:
        # center = prev_max_den
        p = yt.ProjectionPlot(
            ds, slice_axis, "density", width=width, center=prev_max_den
        )
        x_pos = np.array(ad["star", "particle_position_x"]) - prev_max_den[0]
        y_pos = np.array(ad["star", "particle_position_y"]) - prev_max_den[1]
        z_pos = np.array(ad["star", "particle_position_z"]) - prev_max_den[2]
        print("> using old center at {}".format(prev_max_den))
        return p, max_density_coord, x_pos, y_pos, z_pos
