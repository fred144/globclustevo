import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from lum_funcs import look_up_table, get_cluster, star_luminosity_plot
from scipy.optimize import curve_fit

mpl.rc('font', family='serif')
plt.style.use('dark_background')


def king_model(r, k, r_c, r_t):
    """
    https://articles.adsabs.harvard.edu/pdf/1962AJ.....67..471K
    """
    f = k *( (1 / np.sqrt(1 + (r / r_c)**2)) - (1 / np.sqrt(1 + (r_t /r_c )**2)) )**2
    return f

def unpack_pop_ii_data(
        path:str, 
        lum_scaling=1e-5, 
        lum_link='https://www.stsci.edu/science/starburst99/data/l1500_inst_e.dat'
    ): 
    """
    Parameters
    ----------
    path
        path to file
    lum_scaling
        scaling factor for luminosity, see stsci tables
    lum_link
        link to the lookup table

    Returns
    -------
        (x,y,z) positions of stars
        corresponding stellar luminosities
        masses in M_sun
        current time in Myr
    """
    
    pop_2_data = np.loadtxt(path)
    # birth_epochs = pop_2_data[:,0] *1e6
    ages = pop_2_data[:,1] *1e6             # convert to myr
    ages [ages < 1e6 ] = 1e6                # set minimum age
    t_myr = pop_2_data[0,6]
    masses = pop_2_data[:,5]                #msun
    
    # use look up table
    # current bottle neck 
    stellar_lums = look_up_table(
        stellar_ages=ages,
        table_link=lum_link,
        column_idx=1,
        log=True
        )
    
    scaled_stellar_lums = stellar_lums*lum_scaling 
    star_positions = pop_2_data[:,2:5]      # (x,y,z)

    
    return star_positions, scaled_stellar_lums, masses, t_myr

def projected_surf_densities(
    x_coord, y_coord, lums, masses, radius, num_bins, log_bins=True, dr=None        
    ):
    
    starting_point = .04 #pc might have to tweak this. 
    
    # stack two-1d arrays
    all_positions = np.vstack((x_coord, y_coord)).T

    if log_bins is True:
        r = np.geomspace(starting_point, radius, num=num_bins, endpoint=True)
        #r_inner = np.geomspace(0, radius, num=num_bins, endpoint=False)
    else: 
        r = np.arange(0, radius+dr, dr)
    
    # assume that the cluster fed to this has already been translated to (0,0)
    #ctr_at=np.array([0,0])
    
    distances = np.sqrt(np.sum(np.square(all_positions), axis=1))    

    mass_per_bin, bin_edges = np.histogram(distances, bins=r, weights=masses)
    lum_per_bin, _ = np.histogram(distances, bins=r, weights=lums)
    count_per_bin, _ = np.histogram(distances, bins=r)
    
    # getting bin properties
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]
    bin_ctrs = 0.5*( left_edges + right_edges)
    ring_areas = np.pi * (right_edges**2 - left_edges**2)
    
    # calculate densities
    surf_mass_density = mass_per_bin / ring_areas
    surf_lum_density = lum_per_bin / ring_areas
    surf_number_density =  count_per_bin / ring_areas
    
    # characterize what the tupical mass is for a bin 
    avg_star_masses = mass_per_bin / count_per_bin
    err_surf_mass_density = np.sqrt(count_per_bin)*(avg_star_masses/ring_areas)
    total_clust_m = np.sum(mass_per_bin)
    
    return bin_ctrs, surf_mass_density, err_surf_mass_density, total_clust_m
    
        
def king_profiler(star_pos, lums, masses, gc_ctr, gc_rad, bins=25):
    """
    depends on projected_surf_densities
    """
    # get a cluster given a center --> (0,0,0) and spherically mask around it
    if gc_ctr.size == 2:
        clust_x, clust_y, clust_lums, clust_masses = get_cluster(
            xpos=star_pos[:,0],
            ypos=star_pos[:,1],
            zpos=None,
            ctr_at=gc_ctr,
            masses=masses,
            cluster_radius=gc_rad,
            lums=lums,
            trns_coord = True
            )
    else:
        clust_x, clust_y, clust_z, clust_lums, clust_masses = get_cluster(
            xpos=star_pos[:,0],
            ypos=star_pos[:,1],
            zpos=star_pos[:,2],
            ctr_at=gc_ctr,
            masses=masses,
            cluster_radius=gc_rad,
            lums=lums,
            trns_coord = True
            )

    
    # given an isolated cluster, find projected density quantities
    r, rho, err, tot_m = projected_surf_densities(
        x_coord=clust_x, 
        y_coord=clust_y, 
        lums=clust_lums, 
        masses=clust_masses, 
        radius=gc_rad, 
        num_bins=bins, 
        log_bins=True, 
        dr=None        
        )
    
    # plot the data
    plt.figure(figsize = (8,8), )
    plt.errorbar(
        r,
        rho,
        yerr=err,
        fmt='.',
        capsize=4,
        label= r'$M = {:.1e} \: M_{{\odot}}$'.format(tot_m)
        )
    
    # do the fit
    # fit_params, cov_matrix = curve_fit(
    #     f=king_model,
    #     xdata=r,
    #     ydata=rho,
    #     sigma=err,
    #     absolute_sigma=True,
    #     p0 = [1e4, 0.2, 10],
    #     bounds=([0,0,0], [np.inf,np.inf,np.inf]),

    #     )
    # fit_sigma = np.sqrt(np.diag(cov_matrix))
    
    # plot the fit
    # legend = [item for sublist in zip(fit_params[:-1], fit_sigma[:-1]) for item in sublist]
    # legend.append(tot_m)
    # plt.plot(
    #     r,
    #     king_model(r,*fit_params),
    #     label=(r'$k = {:.0f} \pm {:.0f}$'
    #            '\n'
    #            r'$r_c = {:.2f} \pm {:.2f}$ pc'
    #            '\n'
    #            r'$M = {:.1e} \: M_{{\odot}}$'.format(*legend)
    #            )
    #     )
    
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'Surface Mass Density ($M_{\odot} \; pc^{-2}$)', fontsize=14)
    plt.xlabel('Radius (pc)', fontsize=14)
    #plt.xlim(.9*ring_width,gc_rad)
    plt.grid(visible=True, which='both', axis='y', ls='--')
    plt.legend(fontsize=12)
    
    return r, rho, err, tot_m
    
#%%

#test_ctr = np.array([0.8662, -78.5067])
#test_ctr = np.array([peak_x, peak_y])
#test_ctr = np.array([3.781448, 14.74236, 25.41285])
test_rad = 10
test_proj_width = 200
bins = 4000
star_positions, scaled_stellar_lums, masses, t_myr= unpack_pop_ii_data(
    r"./pop_2_data/pos_00446_467_92_myr.txt"
    )
# generate luminosity plot and get peaks based on counts
peak_x, peak_y = star_luminosity_plot(
    proj_width=test_proj_width ,
    star_positions=star_positions,
    scaled_stellar_lums=scaled_stellar_lums,
    time=t_myr,
    snapshot_num=590,
    pi_multiple=0,
    bins=bins,
    plt_type='luminosity',
    annotate_ctrs=True
    )

test_ctrs = np.array([peak_x, peak_y]).T
# iterate over x,y maximas and plot
for ctr in test_ctrs:
    print(ctr)
    r, rho, err, tot_m = king_profiler(
                star_pos=star_positions, 
                lums=scaled_stellar_lums, 
                masses=masses, 
                gc_ctr=ctr, 
                gc_rad=test_rad, 
                bins=50
                )

#%% sigma = bg + (peak)/( 1 + (r/r_c)^alpha) scale background by half the peak, etc.
# =============================================================================
# modify the king profile rt-> inf
# add free power and constant background
# total mass from 100 to 600 myr for pop2
# increase radius
# multiplot, rotationnot zoomed in.
# 
# set to 1.5bg =  bg + (peak)/( 1 + (r/r_c)^alpha)
# # play with threshold
# #count mass up to the radius
# #mass is 2pi r_core^2 sigma_peak ln(rt/rc) from core radius out. 
# # Mcore = pi r_core^2 sigma_peak
# # integrate sigma
#mass as function of trunc radius 
# =============================================================================
