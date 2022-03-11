import numpy as np
import pandas as pd


def look_up_table(stellar_ages, table_link, column_idx:int, log=True):
    """
    given stsci link and ages, returns likely (log) luminosities
    does this via residuals 
    Here are some tables.
    https://www.stsci.edu/science/starburst99/docs/table-index.html 
    """

    df = pd.read_csv(
        table_link,
        delim_whitespace=True,
        header=None
        )
    data = df.to_numpy().astype(float)
    look_up_times = data[:,0] # yr

    if log is True:
        look_up_lumi = 10**data[:,column_idx]
    else:
        look_up_lumi = data[:,column_idx]

    residuals = np.abs(look_up_times - stellar_ages[:, np.newaxis])

    closest_match_idxs = np.argmin(residuals, axis=1)

    luminosities = look_up_lumi[closest_match_idxs]
    return luminosities

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
    if zpos is not None:
        all_positions = np.vstack((xpos, ypos, zpos)).T
    else:
        all_positions = np.vstack((xpos, ypos)).T
        
    distances = np.sqrt(np.sum(np.square(all_positions - ctr_at), axis=1))
    mask = distances <= cluster_radius
    masked_positions = all_positions[mask]
    masked_lums = lums[mask]
    masked_masses = masses[mask]
    masked_ages = ages[mask]
    
    x = masked_positions[:,0]
    y = masked_positions[:,1]
    x_recentered = masked_positions[:,0] - ctr_at[0] #np.mean(x)
    y_recentered = masked_positions[:,1] - ctr_at[1] #np.mean(y)
    
    if zpos is not None:
        z = masked_positions[:,2]
        if trns_coord is True:
            z_recentered = masked_positions[:,2] - ctr_at[2]#p.median(z)
            return x_recentered, y_recentered, z_recentered, masked_lums, masked_masses
        else:
            return x, y, z, masked_lums, masked_masses
    else:
        if trns_coord is True:
            return x_recentered, y_recentered, masked_lums, masked_masses, masked_ages
        else:
            return x, y, masked_lums, masked_masses, masked_ages
        

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

    x =  ring_positions[:,0]
    y =  ring_positions[:,1]
    return x, y,  ring_lums, ring_masses

def surface_2d_brightness(
        xpos, ypos, lums, masses, clust_radius, dr, log_bins=True, num_bins=None,
        ):
    """
    Get surface brightness as a function of radius for a specified clusted
    Deprecate and unparallelized.
    """
    if log_bins == True and num_bins!= None:
        # returns log spaces outer rings, the width of each concentric ring
        # will be preserved and is handled by ring_2d_mask
        outer_rings = np.geomspace(dr, clust_radius+dr, num=num_bins, endpoint=False)
    elif log_bins == False:
        outer_rings = np.arange(dr,clust_radius+dr, dr)
    else:
        print('If log_bins is true, please set a desired # of bins.')

    #print(outer_rings)

    surface_lums_per_ring = []
    counts_per_ring = []
    masses_per_ring = []

    for i,outer_r in enumerate(outer_rings):
        x, y, masked_lums, masked_masses = ring_2d_mask(
            xpos,
            ypos,
            ctr_at=np.array([0,0]),
            lums=lums,
            masses=masses,
            outer_radius=outer_rings[i],
            inner_radius=outer_rings[i-1],
            #dr=dr
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
    surface_lums_per_ring = np.array(surface_lums_per_ring),
    counts_per_ring = np.array(counts_per_ring),
    masses_per_ring = np.array(masses_per_ring)

    # take two circles bigger is charcterized by outer radius,
    # other is this radius minus the ring dr, and calculates area of ring
    ring_areas = np.pi * ( outer_rings**2 - (outer_rings - dr)**2)

    # find quntatity densities
    surface_lum_density = surface_lums_per_ring / ring_areas
    surface_number_density =  counts_per_ring / ring_areas
    surface_mass_density = masses_per_ring / ring_areas

    avg_star_mass_per_ring = masses_per_ring / counts_per_ring
    #print(counts_per_ring)

    err_surface_mass_density = np.sqrt(counts_per_ring) * (avg_star_mass_per_ring  / ring_areas )
    #err_surface_mass_density = np.sqrt(counts_per_ring*avg_star_mass_per_ring / ring_areas )

    clstr_mass = np.sum(masses_per_ring)

    return (
        outer_rings,
        surface_mass_density,
        np.squeeze(err_surface_mass_density.T),
        clstr_mass
        )

