import numpy as np
import os
#import yt
#from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
import matplotlib as mpl
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.transform import Rotation as R 
from lum_funcs import star_luminosity_plot, look_up_table


proj_width = 200 # pc

mpl.rc('font', family='serif') 
plt.style.use('dark_background') 

directory = r"./pop_2_data/"
files = sorted(os.listdir(directory)) #[-5:-1]  

#rotation_interval = np.arange(0,2,.002) # times pi
rotating_timelapse = True
seq_folder_name = "rot_about_y_axis"
rotation_interval = np.linspace(0,2,np.size(files)) 

for i,file_name in enumerate(files, start=0):
    
    print("# read in:", file_name)
    time_str = file_name[10:16].replace('_','.') #in myr
    time = float(time_str)
    snapshot_num = int(file_name[4:9])
    #print(file_name)

    pop_2_data = np.loadtxt('./pop_2_data/' + file_name) 
    birth_epochs = pop_2_data[:,0] *1e6 
    ages = pop_2_data[:,1] *1e6 
    # if the age <10^6 yr, set the age to 10^6 year.
    ages [ages < 1e6 ] = 1e6
    
    
    #stellar_lums = look_up_table(ages, lookuptime, lum_imf_2_35_m_up_100msun) 
    stellar_lums = look_up_table(
        stellar_ages=ages, 
        table_link='https://www.stsci.edu/science/starburst99/data/l1500_inst_e.dat', 
        column_idx=1)
    
    scaled_stellar_lums = stellar_lums*1e-5 
   
    star_positions = pop_2_data[:,2:5] 
    
    # x_pos = pop_2_data[:,2]
    # y_pos = pop_2_data[:,3] 
    # z_pos = pop_2_data[:,4]
    
    if rotating_timelapse is True:
        pi_multiple = rotation_interval[i]
        rotation_angle = pi_multiple*np.pi
        # along (x,y,z) axis
        r = R.from_rotvec(rotation_angle * np.array([0,1,0]))
        rotation_matrix = r.as_matrix()
        rotated_star_positions = np.dot(star_positions, rotation_matrix .T)
        star_positions = rotated_star_positions 
    else:
        pi_multiple = 0

    star_luminosity_plot(proj_width=200,
                         star_positions=star_positions, 
                         scaled_stellar_lums=scaled_stellar_lums, 
                         time=time, 
                         snapshot_num=snapshot_num, 
                         pi_multiple=pi_multiple) 
    
    save_name = './sequences/{}/rot_{}_t_{}_pi_{}.png'.format(
        seq_folder_name,
        str(snapshot_num).zfill(4), 
        str(time).ljust(6, '0').replace('.','_'),
        str(np.round(pi_multiple,3)).ljust(5, '0').replace('.','_'))
    plt.savefig( 
        save_name, 
        dpi=200,
        bbox_inches='tight',
        pad_inches=0.05
        )
    # plt.show() 
    plt.close() 
    print("# saved:", save_name)   
    
    # after rotating as the stars are forming, rotate the rest of the set
    # rotation interval vector using the last snapshot-- frozen in time
    
    if (np.size(files) - 1) == i:
        print("# time frozen")
        
        # frozen rotation 
        frozen_rotation_interval = np.linspace(2,4,np.size(files)) 
        
        for remaining_pi_multiple in frozen_rotation_interval[1:]: # stutter
            # for remaining_pi_multiple in rotation_interval[i+1:]: 
            
            stellar_lums = look_up_table(
                stellar_ages=ages, 
                table_link='https://www.stsci.edu/science/starburst99/data/l1500_inst_e.dat', 
                column_idx=1)
            scaled_stellar_lums = stellar_lums*1e-5 
            
            #pi_multiple = remaining_pi_multiple
            rotation_angle = remaining_pi_multiple*np.pi
            star_positions = pop_2_data[:,2:5] 
            
            # along (x,y,z) axis
            r = R.from_rotvec(rotation_angle * np.array([0,1,0]))
            rotation_matrix = r.as_matrix()
            rot_star_positions = np.dot(star_positions, rotation_matrix .T)
            
            star_luminosity_plot(proj_width=200,
                                 star_positions=rot_star_positions, 
                                 scaled_stellar_lums=scaled_stellar_lums, 
                                 time=time, 
                                 snapshot_num=snapshot_num, 
                                 pi_multiple=remaining_pi_multiple)
            
            save_name = './sequences/{}/rot_{}_t_{}_pi_{}.png'.format(
                seq_folder_name,
                str(snapshot_num).zfill(4), 
                str(time).ljust(6, '0').replace('.','_'),
                str(np.round(remaining_pi_multiple,3)).ljust(5, '0').replace('.','_'))
            plt.savefig( 
                save_name, 
                dpi=200,
                bbox_inches='tight',
                pad_inches=0.05
                )
            # plt.show() 
            plt.close()  
            print("# saved:", save_name)  


