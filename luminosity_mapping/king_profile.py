import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from lum_funcs import look_up_table, get_cluster, surface_2d_brightness


def king_model(r, k, r_c, r_t): 
    """
    https://articles.adsabs.harvard.edu/pdf/1962AJ.....67..471K
    """
    f = k *( (1 / np.sqrt(1 + (r / r_c)**2)) - (1 / np.sqrt(1 + (r_t /r_c )**2)) )**2
    return f

pop_2_data = np.loadtxt(r"./pop_2_data/pos_00500_479_16_myr.txt")   
birth_epochs = pop_2_data[:,0] *1e6 
ages = pop_2_data[:,1] *1e6 
ages [ages < 1e6 ] = 1e6 # set minimum age
current_time_myr = pop_2_data[0,6]
masses = pop_2_data[:,5] #msun 


stellar_lums = look_up_table(
    stellar_ages=ages, 
    table_link='https://www.stsci.edu/science/starburst99/data/l1500_inst_e.dat', 
    column_idx=1,
    log=True)

scaled_stellar_lums = stellar_lums*1e-5 
star_positions = pop_2_data[:,2:5] # (x,y,z)

gc_ctr = np.array([3.781448, 14.74236, 25.41285]) # pc
#gc_ctr = np.array([0.8662809, -78.50676, -66.55415])
ring_width = .05
gc_rad = 15

cluster_x, cluster_y, cluster_z, cluster_lums, cluster_masses = get_cluster(
    xpos=star_positions[:,0], 
    ypos=star_positions[:,1], 
    zpos=star_positions[:,2],
    masses=masses,
    ctr_at=gc_ctr, 
    cluster_radius=gc_rad, 
    lums=scaled_stellar_lums,
    trns_coord = True
    )

r, surface_mass_density, err_surface_mass_density =  surface_2d_brightness(
    xpos=cluster_x, 
    ypos=cluster_y, 
    lums=cluster_lums, 
    masses=cluster_masses, 
    clust_radius=gc_rad,
    dr=ring_width, 
    log_bins=True,
    num_bins=50

)
#%% Fitting 


#%%
plt.figure(figsize = (8,8), )
plt.errorbar(
    r, 
    surface_mass_density, 
    yerr=err_surface_mass_density,
    fmt='.',
    capsize=4,
    c='tab:red'
    )
plt.xscale('log')
plt.yscale('log')
plt.grid(visible=True, which='both', axis='y', ls='--')