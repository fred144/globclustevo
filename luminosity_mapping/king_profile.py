import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from lum_funcs import look_up_table, spherical_mask , get_cluster, surface_2d_brightness



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
gc_ctr = np.array([0.8662809, -78.50676, -66.55415])
ring_width = .2
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

d_r_2d, surface_lum_2d, counts_2d, masses_2d = surface_2d_brightness(
    xpos=cluster_x, 
    ypos=cluster_y, 
    lums=cluster_lums, 
    masses=cluster_masses, 
    dr=ring_width, 
    clust_radius=gc_rad
)