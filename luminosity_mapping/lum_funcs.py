import numpy as np
import pandas as pd
#import yt
#from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#import seaborn as sns
import matplotlib.cm as cm
import matplotlib.patches as patches
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

def star_luminosity_plot(
        proj_width,
        star_positions,
        scaled_stellar_lums,
        time,
        snapshot_num,
        pi_multiple,
        sfc_positions=None,
        psc_positions=None,
        ):
    
    lums, xedges, yedges = np.histogram2d(
        star_positions[:,0],
        star_positions[:,1],
        bins=2000,
        weights=scaled_stellar_lums,
        normed=False,
        range= [[-proj_width/2, proj_width/2], [-proj_width/2, proj_width/2]]
    )
    """
    currently makes prjection plot along projected on z axis
    """
    lums = lums.T

    fig = plt.figure(figsize=(14,12),dpi=200)
    ax = fig.add_subplot(111, facecolor=cm.inferno(0))
    rectbin = plt.imshow(
               lums,
               cmap='inferno',
               interpolation='gaussian',
               origin='lower',
               extent=[-proj_width/2,
                       proj_width/2,
                       -proj_width/2,
                       proj_width/2],
               norm=LogNorm(vmin=3e+32, vmax=3e+36)
               )
    if sfc_positions is not None:
        sfc_x = sfc_positions[:,0]
        sfc_y = sfc_positions[:,1]
        sfc_tags = sfc_positions[:,3]

        plt.scatter(
            sfc_x,
            sfc_y,
            marker='.',
            s=5
            )


        for i, txt in enumerate(sfc_tags):
            plt.annotate(int(txt), (sfc_x[i], sfc_y [i]), fontsize=7, ha='center')


        plt.xlim(-proj_width/2, proj_width/2)
        plt.ylim(-proj_width/2, proj_width/2)


    if psc_positions is not None:
        psc_x = psc_positions[:,0]
        psc_y = psc_positions[:,1]
        psc_tags = psc_positions[:,3]

        plt.scatter(
            psc_positions[:,0],
            psc_positions[:,1],
            marker='.',
            s=5,
            c='lime'
            )


        for i, txt in enumerate(psc_tags):
            plt.annotate(int(txt), (psc_x[i], psc_y [i]), fontsize=7, ha='center')

        plt.xlim(-proj_width/2, proj_width/2)
        plt.ylim(-proj_width/2, proj_width/2)



    # annotate with time
    ax.text(
        -proj_width*0.375,
        -proj_width*0.45,
        't = %.2f Myr'%(time),
        size=12,
        ha='center',
        va='center',
        color='white')
    # add color bar
    # to the side
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes('bottom', size='1%', pad=0.0)
    # cbar = fig.colorbar(
    #     rectbin,
    #     pad = 0,
    #     aspect=60,
    #     )
    # to the bottom
    fig.subplots_adjust(wspace=0, hspace=0, bottom=.1)
    cbar_ax = fig.add_axes([.178, .090, 0.67, 0.010])
    cbar = fig.colorbar(rectbin,
                 cax=cbar_ax,
                 orientation='horizontal',
                 pad=0
                )
    cbar.set_label(
        label=r"$\lambda = 1500\;\AA$ Projected Monochromatic Luminosity" +
        r" $\left(erg\;s^{-1}\AA^{-1} \right)$",
        size=12
        )
    # fig.suptitle(
    #     r"$\lambda = 1500\;\AA$ Projected Monochromatic Luminosity" +
    #     r" $\left(erg\;s^{-1}\AA^{-1} \right)$",
    #     y=.91,
    #     size=12)

    # add scale bar
    rect = patches.Rectangle(
            xy=(-proj_width*0.125, proj_width*0.45),
            width=proj_width*0.25,
            height=proj_width*0.005,
            linewidth=0,
            edgecolor='white',
            facecolor='white')
    ax.add_patch(rect)
    ax.text(
        0,
        proj_width*0.475,
        '{}pc'.format(int(proj_width/4)),
        size=12,
        ha='center',
        va='center',
        color='white'
        )

    #ax.set_axis_off()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.add_artist(ax.patch)
    ax.patch.set_zorder(-1)


def look_up_table(stellar_ages, table_link, column_idx:int, log=True):
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
    #print(np.shape(residuals))
    closest_match_idxs = np.argmin(residuals, axis=1)
    #print(np.shape( closest_match_idxs), np.min(closest_match_idxs), np.max(closest_match_idxs))
    luminosities = look_up_lumi[closest_match_idxs]
    return luminosities

def get_cluster(
        xpos,
        ypos,
        zpos,
        ctr_at,
        cluster_radius,
        lums,
        masses,
        trns_coord
        ):
    """
    Mask out a cluster based on its center, radius.
    Returns luminosities masses and transformed coordinates,
    the ctr_at becomes the origin.
    """
    all_positions = np.vstack((xpos, ypos, zpos)).T
    distances = np.sqrt(np.sum(np.square(all_positions - ctr_at), axis=1))
    mask = distances <= cluster_radius
    #print(distances.size)
    masked_positions = all_positions[mask]
    masked_lums = lums[mask]
    masked_masses = masses[mask]

    x,y,z = (masked_positions[:,0], masked_positions[:,1], masked_positions[:,2])

    if trns_coord is True:
        x_recentered = masked_positions[:,0] - np.median(x)
        y_recentered = masked_positions[:,1] - np.median(y)
        z_recentered = masked_positions[:,2] - np.median(z)

        return x_recentered, y_recentered, z_recentered, masked_lums, masked_masses
    else:
        return x, y, z, masked_lums, masked_masses

def ring_2d_mask(xpos, ypos, ctr_at, lums, masses, outer_radius, inner_radius):
    """
    Gets stars within a 2d ring.
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

