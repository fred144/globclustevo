import numpy as np
import pandas as pd


def get_cluster(
    xpos,
    ypos,
    ctr_at,
    cluster_radius,
    lums,
    masses,
    ages,
    trns_coord,
    zpos=None,
):
    r"""
    Mask out a cluster based on its center, radius.
    Returns luminosities masses and transformed coordinates,
    the ctr_at becomes the origin.

    Can take 2d or 3d coordinates.
    """

    # all_positions_2d = np.vstack((xpos, ypos)).T
    # distances_2d = np.sqrt(np.sum(np.square(all_positions_2d - ctr_at[:-1]), axis=1))
    # mask = distances_2d <= cluster_radius
    # masked_positions = all_positions_2d[mask]
    # masked_lums = lums[mask]
    # masked_masses = masses[mask]
    # masked_ages = ages[mask]
    # print("***Previous Cluster Size", np.size(masked_lums))

    all_positions = np.vstack((xpos, ypos, zpos)).T
    distances = np.sqrt(np.sum(np.square(all_positions - ctr_at), axis=1))

    mask = distances < cluster_radius
    masked_positions = all_positions[mask]
    masked_lums = lums[mask]
    masked_masses = masses[mask]
    masked_ages = ages[mask]

    x = masked_positions[:, 0]
    y = masked_positions[:, 1]

    x_recentered = masked_positions[:, 0] - ctr_at[0]  # np.mean(x)
    y_recentered = masked_positions[:, 1] - ctr_at[1]  # np.mean(y)

    if zpos is not None:
        z = masked_positions[:, 2]
        if trns_coord is True:
            # make the center (0,0,0) ?
            z_recentered = masked_positions[:, 2] - ctr_at[2]  # p.median(z)
            # print("***Now Returning", np.size(x_recentered))
            return (
                x_recentered,
                y_recentered,
                z_recentered,
                masked_lums,
                masked_masses,
                masked_ages,
            )
        else:
            return x, y, z, masked_lums, masked_masses
    else:
        if trns_coord is True:
            return x_recentered, y_recentered, masked_lums, masked_masses, masked_ages
        else:
            return x, y, masked_lums, masked_masses, masked_ages


def surface_density(x_coord, y_coord, z_coord, masses, radius, num_bins):
    """
    Gets 3d density profile of a DM halo.

    """

    starting_point = 0.01  # in parsecs
    r = np.geomspace(starting_point, radius, num=num_bins, endpoint=True)

    all_positions = np.vstack((x_coord, y_coord, z_coord)).T

    distances = np.sqrt(np.sum(np.square(all_positions), axis=1))
    mass_per_bin, bin_edges = np.histogram(distances, bins=r, weights=masses)
    count_per_bin, _ = np.histogram(distances, bins=r)

    # mask out empty bins
    mask = count_per_bin > 0
    mass_per_bin = mass_per_bin[mask]
    count_per_bin = count_per_bin[mask]

    # getting bin properties
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]
    bin_ctrs = 0.5 * (left_edges + right_edges)[mask]
    # calculate the mass per thin surface area of a sphere
    surface_areas = 4 * np.pi * (right_edges**2 - left_edges**2)[mask]
    surf_mass_density = mass_per_bin / surface_areas
    avg_star_masses = mass_per_bin / count_per_bin
    err_surf_mass_density = np.sqrt(count_per_bin) * (avg_star_masses / surface_areas)

    return bin_ctrs, surf_mass_density, err_surf_mass_density


def projected_surf_densities(
    x_coord,
    y_coord,
    lums,
    masses,
    radius,
    num_bins,
    log_bins=True,
    dr=None,
    calc_half_r=None,
):
    r"""
    Gets projected density profiles centered at a given coordinate.
    Log bins by default.

    """
    # TODO: calculate half mass
    starting_point = 0.01  # pc might have to tweak this.

    # stack two-1d arrays
    all_positions = np.vstack((x_coord, y_coord)).T

    if log_bins is True:
        r = np.geomspace(starting_point, radius, num=num_bins, endpoint=True)
        # r_inner = np.geomspace(0, radius, num=num_bins, endpoint=False)
    else:
        r = np.arange(0, radius + dr, dr)

    # assume that the cluster fed to this has already been translated to (0,0)
    # ctr_at=np.array([0,0])

    distances = np.sqrt(np.sum(np.square(all_positions), axis=1))

    mass_per_bin, bin_edges = np.histogram(distances, bins=r, weights=masses)

    lum_per_bin, _ = np.histogram(distances, bins=r, weights=lums)
    count_per_bin, _ = np.histogram(distances, bins=r)

    # mask zero count bins
    mask = count_per_bin > 0
    count_per_bin = count_per_bin[mask]
    mass_per_bin = mass_per_bin[mask]
    lum_per_bin = lum_per_bin[mask]

    # getting bin properties
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]
    bin_ctrs = 0.5 * (left_edges + right_edges)[mask]
    ring_areas = np.pi * (right_edges**2 - left_edges**2)[mask]

    # calculate densities
    surf_mass_density = mass_per_bin / ring_areas
    surf_lum_density = lum_per_bin / ring_areas
    surf_number_density = count_per_bin / ring_areas

    # characterize what the typical mass is for a bin
    avg_star_masses = mass_per_bin / count_per_bin
    # piosson error in the surface density
    err_surf_mass_density = np.sqrt(count_per_bin) * (avg_star_masses / ring_areas)
    # sum the bins to get total mass out to the specified cluster radii
    total_clust_m = np.sum(mass_per_bin)
    total_clust_lum = np.sum(lum_per_bin)

    if calc_half_r is not None:
        # reevaluate with increased bin resolition to get as close to the actual values
        dr = 0.1
        high_res_r = np.arange(0, radius + 0.01, 0.01)
        high_res_m_per_bin, _ = np.histogram(distances, bins=high_res_r, weights=masses)
        high_res_l_per_bin, _ = np.histogram(distances, bins=high_res_r, weights=lums)
        integrated_mass = np.cumsum(high_res_m_per_bin)
        integrated_light = np.cumsum(high_res_l_per_bin)
        half_mass_point = np.abs(integrated_mass - 0.5 * total_clust_m).argmin()
        half_light_point = np.abs(integrated_light - 0.5 * total_clust_lum).argmin()

        half_mass_r = high_res_r[half_mass_point]
        half_light_r = high_res_r[half_light_point]

        return (
            bin_ctrs,
            surf_mass_density,
            err_surf_mass_density,
            total_clust_m,
            total_clust_lum,
            half_mass_r,
            half_light_r,
        )
    else:

        return (
            bin_ctrs,
            surf_mass_density,
            err_surf_mass_density,
            total_clust_m,
            total_clust_lum,
        )


def ring_2d_mask(xpos, ypos, ctr_at, lums, masses, outer_radius, inner_radius):
    """
    Gets stars within a 2d ring.
    Deprecate and unparallelized.
    """
    all_positions = np.vstack((xpos, ypos)).T
    distances = np.sqrt(np.sum(np.square(all_positions - ctr_at), axis=1))

    # get the points within [outer radius - dr, outer radius]
    mask = ((inner_radius) <= distances) & (distances <= outer_radius)

    ring_positions = all_positions[mask]
    ring_lums = lums[mask]
    ring_masses = masses[mask]

    x = ring_positions[:, 0]
    y = ring_positions[:, 1]
    return x, y, ring_lums, ring_masses


def surface_2d_brightness(
    xpos,
    ypos,
    lums,
    masses,
    clust_radius,
    dr,
    log_bins=True,
    num_bins=None,
):
    """
    Get surface brightness as a function of radius for a specified cluster.
    Deprecate and unparallelized.
    """
    #!!! Deprecated
    print("Deprecated, use king_profiler and projected_surf_densities.")
    if log_bins == True and num_bins != None:
        # returns log spaces outer rings, the width of each concentric ring
        # will be preserved and is handled by ring_2d_mask
        outer_rings = np.geomspace(dr, clust_radius + dr, num=num_bins, endpoint=False)
    elif log_bins == False:
        outer_rings = np.arange(dr, clust_radius + dr, dr)
    else:
        print("If log_bins is true, please set a desired # of bins.")

    # print(outer_rings)

    surface_lums_per_ring = []
    counts_per_ring = []
    masses_per_ring = []

    for i, outer_r in enumerate(outer_rings):
        x, y, masked_lums, masked_masses = ring_2d_mask(
            xpos,
            ypos,
            ctr_at=np.array([0, 0]),
            lums=lums,
            masses=masses,
            outer_radius=outer_rings[i],
            inner_radius=outer_rings[i - 1],
            # dr=dr
        )
        # total luminosity in a given ring
        surface_lums_per_ring.append(np.sum(masked_lums))
        # count of stars in a given ring
        counts_per_ring.append(len(masked_lums))
        # mass in a given 2d ring
        masses_per_ring.append(np.sum(masked_masses))

        # test the binning image save
        # plt.figure(figsize = (8,8), )
        # plt.scatter(x,y,s=1)
        # plt.xlim(-20,20)
        # plt.ylim(-20,20)
        # plt.savefig('test_sequence/dr_05_flat/{:05d}.png'.format(i))
        # print(i)
        # plt.clf()

    # reformat lists as arrays
    surface_lums_per_ring = (np.array(surface_lums_per_ring),)
    counts_per_ring = (np.array(counts_per_ring),)
    masses_per_ring = np.array(masses_per_ring)

    # take two circles bigger is charcterized by outer radius,
    # other is this radius minus the ring dr, and calculates area of ring
    ring_areas = np.pi * (outer_rings**2 - (outer_rings - dr) ** 2)

    # find quntatity densities
    surface_lum_density = surface_lums_per_ring / ring_areas
    surface_number_density = counts_per_ring / ring_areas
    surface_mass_density = masses_per_ring / ring_areas

    avg_star_mass_per_ring = masses_per_ring / counts_per_ring
    # print(counts_per_ring)

    err_surface_mass_density = np.sqrt(counts_per_ring) * (
        avg_star_mass_per_ring / ring_areas
    )
    # err_surface_mass_density = np.sqrt(counts_per_ring*avg_star_mass_per_ring / ring_areas )

    clstr_mass = np.sum(masses_per_ring)

    return (
        outer_rings,
        surface_mass_density,
        np.squeeze(err_surface_mass_density.T),
        clstr_mass,
    )
