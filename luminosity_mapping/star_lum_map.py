import sys
sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/cosmology-clean/lib/python3.7/site-packages"
    ) 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os 
from scipy.spatial.transform import Rotation as R
from lum_funcs import look_up_table
from lum_plotting_lib import star_luminosity_plot

# ==========================Render Parameters==================================
proj_width = 400 # pc

mpl.rc('font', family='serif')
plt.style.use('dark_background')

data_directory = r"./pop_2_data/"
# seq_parent = r"./test/"
seq_parent = r"./test_sequence/"
seq_folder_name = "dynamic_pause_rotate"
if not os.path.exists(seq_parent+seq_folder_name):
    print("# Creating new sequence directory",seq_parent+seq_folder_name)
    os.makedirs(seq_parent+seq_folder_name)
# test is 
# num_ctr=50, 
# ctr_dist_thresh=100, 
# ctr_rel_thresh=.08,

# test 1 is
# num_ctr=50, 
# ctr_dist_thresh=10, 
# ctr_rel_thresh=.1,

# enable discrete selection of time range based on snapshot number
strt_snapshot = "00113"
end_snapshot = "00694"
files = sorted(os.listdir(data_directory))  #[-2:-1]  [300:400:2]
strt_idx = [i for i, s in enumerate(files) if strt_snapshot in s][0]
end_idx = [i for i, s in enumerate(files) if end_snapshot in s][0]
filtered_files = files [strt_idx:end_idx]

# rotate flags
rotating_timelapse = False
rotate_at_the_end = False
pause_and_rotate = True

# rate of rotation times pi
timelapse_rotation_interval = np.linspace(0,2,np.size(files))
frozen_rotation_interval = np.linspace(2,4,np.size(files))
mid_pause_rotation_interval = np.linspace(0,2,100)

# pause at a specific frame and rotate
pause_at = [405,500]

# gc centering 
annotate_gc = False 

# =============================================================================
for i,file_name in enumerate(filtered_files, start=0):

    print("# read in:", file_name)
    time_str = file_name[10:16].replace('_','.') #in myr
    time = float(time_str)
    snapshot_num = int(file_name[4:9])
    #print(file_name)

    pop_2_data = np.loadtxt(data_directory+file_name)
    birth_epochs = pop_2_data[:,0] *1e6
    ages = pop_2_data[:,1] *1e6
    # if the age <10^6 yr, set the age to 10^6 year.
    ages [ages < 1e6 ] = 1e6
    masses = pop_2_data[:,5]

    stellar_lums = look_up_table(
        stellar_ages=ages,
        table_link='https://www.stsci.edu/science/starburst99/data/l1500_inst_e.dat',
        column_idx=1)
    scaled_stellar_lums = stellar_lums*1e-5

    # get positions from general data
    star_positions = pop_2_data[:,2:5]

    # x_pos = pop_2_data[:,2]
    # y_pos = pop_2_data[:,3]
    # z_pos = pop_2_data[:,4]

    if rotating_timelapse is True:
        pi_multiple = timelapse_rotation_interval[i]
        rotation_angle = pi_multiple*np.pi
        # along (x,y,z) axis
        r = R.from_rotvec(rotation_angle * np.array([0,1,0]))
        rotation_matrix = r.as_matrix()
        # rotate stars
        rotated_star_positions = np.dot(star_positions, rotation_matrix.T)
        star_positions = rotated_star_positions
        # rotate test particles 
    elif pause_and_rotate is True and snapshot_num in pause_at:
        # pause the movie at a snapshot and rotate it 
        for pause_rot_angle in mid_pause_rotation_interval:
            # reset the star positions every loop
            star_positions = pop_2_data[:,2:5]
            masses = pop_2_data[:,5]
            # new angle each loop
            rotation_angle = pause_rot_angle*np.pi
            # along (x,y,z) axis
            r = R.from_rotvec(rotation_angle * np.array([0,1,0]))
            rotation_matrix = r.as_matrix()
            # rotate stars
            rotated_star_positions = np.dot(star_positions, rotation_matrix.T)
            star_positions = rotated_star_positions
# =============================================================================            
            _, _ = star_luminosity_plot(
                proj_width=proj_width,
                star_positions=star_positions,
                scaled_stellar_lums=scaled_stellar_lums,
                time=time,
                snapshot_num=snapshot_num,
                pi_multiple=pause_rot_angle,
                plt_bins=4000,
                lum_scale=('dynamic', 3e+32, 3e+36),
                get_ctr=(True, 'potential', 0.04, annotate_gc),
                masses=masses,
                ) 
# =============================================================================
            save_name = '{}{}/rot_{}_t_{}_pi_{}.png'.format(
                seq_parent,
                seq_folder_name,
                str(snapshot_num).zfill(4),
                str(time).ljust(6, '0').replace('.','_'),
                str(np.round(pause_rot_angle,3)).ljust(5, '0').replace('.','_'))
            plt.savefig(
                save_name,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.05
                )
            plt.show()
            #plt.close()
            print("# saved:", save_name)
    else:
        pi_multiple = 0

# =============================================================================
 
    _, _ = star_luminosity_plot(
        proj_width=proj_width,
        star_positions=star_positions,
        scaled_stellar_lums=scaled_stellar_lums,
        time=time,
        snapshot_num=snapshot_num,
        pi_multiple=pi_multiple,
        plt_bins=4000,
        get_ctr=(True, 'potential', 0.04, annotate_gc),
        lum_scale=('dynamic', 3e+32, 3e+36),
        masses=masses,
    ) 
# =============================================================================
    save_name = '{}{}/rot_{}_t_{}_pi_{}.png'.format(
        seq_parent,
        seq_folder_name,
        str(snapshot_num).zfill(4),
        str(time).ljust(6, '0').replace('.','_'),
        str(np.round(pi_multiple,3)).ljust(5, '0').replace('.','_'))
    plt.savefig(
        save_name,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.05
        )
    plt.show()
    #plt.close()
    print("# saved:", save_name)

    # after rotating as the stars are forming, rotate the rest of the set
    # rotation interval vector using the last snapshot-- frozen in time

    if rotate_at_the_end is True and (np.size(files) - 1) == i:
        print("# at the end, time frozen and will rotate")


        for end_rot_angle in frozen_rotation_interval[1:]: # stutter
            # for remaining_pi_multiple in rotation_interval[i+1:]:

            stellar_lums = look_up_table(
                stellar_ages=ages,
                table_link='https://www.stsci.edu/science/starburst99/data/l1500_inst_e.dat',
                column_idx=1)
            scaled_stellar_lums = stellar_lums*1e-5

            #pi_multiple = remaining_pi_multiple
            rotation_angle = end_rot_angle*np.pi
            star_positions = pop_2_data[:,2:5]
            masses = pop_2_data[:,5]
            # along (x,y,z) axis
            r = R.from_rotvec(rotation_angle * np.array([0,1,0]))
            rotation_matrix = r.as_matrix()
            rotated_star_positions = np.dot(star_positions, rotation_matrix .T)
# =============================================================================
            _, _ = star_luminosity_plot(
                proj_width=proj_width,
                star_positions=star_positions,
                scaled_stellar_lums=scaled_stellar_lums,
                time=time,
                snapshot_num=snapshot_num,
                pi_multiple=pause_rot_angle,
                plt_bins=4000,
                lum_scale=('dynamic', 3e+32, 3e+36),
                get_ctr=(True, 'potential', 0.01, annotate_gc),
                masses=masses,
            ) 
# =============================================================================
            save_name = '{}{}/rot_{}_t_{}_pi_{}.png'.format(
                seq_parent,
                seq_folder_name,
                str(snapshot_num).zfill(4),
                str(time).ljust(6, '0').replace('.','_'),
                str(np.round(end_rot_angle,3)).ljust(5, '0').replace('.','_'))
            plt.savefig(
                save_name,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.5
                )
            # plt.show()
            plt.close()
            print("# saved:", save_name)


