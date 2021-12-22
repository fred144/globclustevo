from cluster_evolution_fs07.macros import code_age_to_yr
import numpy as np
import os
import yt

pop_2_data = np.loadtxt(r'star_positions_out400.txt') 
x_pos = pop_2_data[:,1]
y_pos = pop_2_data[:,2] 

