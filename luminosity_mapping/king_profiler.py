import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from lum_funcs import look_up_table, get_cluster 
from scipy.optimize import curve_fit

mpl.rc('font', family='serif')
plt.style.use('dark_background')
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })

def king_model(r, k, r_c, r_t):
    """
    https://articles.adsabs.harvard.edu/pdf/1962AJ.....67..471K
    """
    f = k *( (1 / np.sqrt(1 + (r / r_c)**2)) - (1 / np.sqrt(1 + (r_t /r_c )**2)) )**2
    return f

def modified_king_model(r, sigma_naught, r_c, alpha, bg):

    sigma = bg + (sigma_naught / (1 + (r/r_c)**alpha) )
    return sigma

def trunc_radius(sigma_0, r_c, alpha, sigma_bg):
    """
    set to 1.5bg =  bg + (peak)/( 1 + (r/r_c)^alpha)
    0.5bg = (peak)/( 1 + (r/r_c)^alpha)
    """
    #trunc_r = (r_c**alpha * ((sigma_0/(.5*sigma_bg)) - 1) )**(1/alpha)
    r_trunc = (r_c**alpha * ((sigma_0/ ((1.5-1)*sigma_bg ) - 1 ) ) )**(1/alpha)
    return r_trunc
    

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
    #print(mass_per_bin.size, lum_per_bin.size, count_per_bin.size)
    # mask zero count bins
    mask = count_per_bin > 0
    count_per_bin = count_per_bin[mask]
    mass_per_bin = mass_per_bin[mask]
    lum_per_bin = lum_per_bin[mask]
    
    # getting bin properties
    right_edges = bin_edges[1:] 
    left_edges = bin_edges[:-1]
    bin_ctrs = 0.5*( left_edges + right_edges)[mask]
    ring_areas = np.pi * (right_edges**2 - left_edges**2)[mask]
    

    
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
        fmt='o',
        capsize=5,
        elinewidth=2,
        label= r'$M_{{total}}= {:.2e} \: M_{{\odot}}$'.format(tot_m) 
        )
    
    # do the fit
    try:
        fit_params, cov_matrix = curve_fit(
            f=modified_king_model,
            xdata=r,
            ydata=rho,
            sigma=err,
            absolute_sigma=True,
            p0 = [1e4, 0.2, 2, 10],
            bounds=([0,0,0,0], [np.inf,np.inf,100,np.inf]),
            )
        #r, sigma_naught, r_c, alpha, bg
        fit_sigma = np.sqrt(np.diag(cov_matrix))
        # print(fit_params)
        # print(fit_sigma)
        
        fit_sigma_naught = fit_params[0]
        fit_r_c = fit_params[1]
        fit_alpha = fit_params[2]
        fit_sigma_bg = fit_params[3]
        
        truncation_radius = trunc_radius(
            r_c=fit_r_c, 
            alpha=fit_alpha, 
            sigma_naught=fit_sigma_naught,
            sigma_bg=fit_sigma_bg
            )
        # set to 1.5bg =  bg + (peak)/( 1 + (r/r_c)^alpha)
        
        #plot the fit
    
        plt.plot(
            r,
            modified_king_model(r,*fit_params),
            linewidth=2,
            label=(r'$R_{{trunc}} = {:.2f} \: pc$'
                   '\n'
                   r'$R_{{core}} = {:.2f} \: pc$').format(truncation_radius,fit_r_c)
            )
        
    except:
        plt.title("can't fit")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'Surface Mass Density ($M_{\odot} \; pc^{-2}$)', fontsize=16)
    plt.xlabel(r'Radius ($pc$)', fontsize=16)
    #plt.xlim(.9*ring_width,gc_rad)
    plt.grid(visible=True, which='both', axis='y', ls='--')
    plt.legend(fontsize=16)

    return r, rho, err, tot_m
    
#%%
from lum_plotting_lib import star_luminosity_plot

# test_ctr = np.array([0.8662, -78.5067])
# #test_ctr = np.array([peak_x, peak_y])
# test_ctr = np.array([3.781448, 14.74236, 25.41285])
test_rad = 10
test_proj_width = 400
bins = 4000
star_positions, scaled_stellar_lums, masses, t_myr= unpack_pop_ii_data(
    r"./pop_2_data/pos_00446_467_92_myr.txt"
    )

# generate luminosity plot and get peaks based on counts
# peak_x, peak_y = star_luminosity_plot(
#     proj_width=test_proj_width ,
#     star_positions=star_positions,
#     scaled_stellar_lums=scaled_stellar_lums,
#     time=t_myr,
#     snapshot_num=590,
#     pi_multiple=0,
#     bins=bins,
#     plt_type='luminosity',
#     annotate_ctrs=True
#     )



peak_x, peak_y = star_luminosity_plot(
    proj_width=test_proj_width,
    star_positions=star_positions,
    scaled_stellar_lums=scaled_stellar_lums,
    time=t_myr,
    snapshot_num=446,
    pi_multiple=0,
    plt_bins=bins,
    get_ctr=(True, 'potential', 0.04, True),
    masses=masses,
    ) 

gc_masses = []
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
                bins=25
                )
    gc_masses.append(tot_m)
#%%
plt.hist(np.gc_masses, bins=10) 
#plt.xscale('log')

