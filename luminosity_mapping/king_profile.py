import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from lum_funcs import look_up_table, spherical_mask 


pop_2_data = np.loadtxt(r"./pop_2_data/pos_00500_479_16_myr.txt")   
birth_epochs = pop_2_data[:,0] *1e6 
ages = pop_2_data[:,1] *1e6 
ages [ages < 1e6 ] = 1e6 # set minimum age
current_time_yr = pop_2_data[0,]

#stellar_lums = look_up_table(ages, lookuptime, lum_imf_2_35_m_up_100msun) 
stellar_lums = look_up_table(
    stellar_ages=ages, 
    table_link='https://www.stsci.edu/science/starburst99/data/l1500_inst_e.dat', 
    column_idx=1)

scaled_stellar_lums = stellar_lums*1e-5 

star_positions = pop_2_data[:,2:5] 