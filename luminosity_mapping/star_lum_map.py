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
sfc_dir = r"./sfc_data/"
psc_dir = r"./psc_data/"

files = sorted(os.listdir(directory))   [387:400]
sfc_files = sorted(os.listdir(sfc_dir)) [387:400]
psc_files = sorted(os.listdir(psc_dir)) [387:400]

plot_test_parts = True

#rotation_interval = np.arange(0,2,.002) # times pi
rotating_timelapse = True
seq_folder_name = "test"
rotation_interval = np.linspace(0,2,np.size(files))

for i,file_name in enumerate(files, start=0):

    print("# read in:", file_name)
    time_str = file_name[10:16].replace('_','.') #in myr
    time = float(time_str)
    snapshot_num = int(file_name[4:9])
    #print(file_name)

    pop_2_data = np.loadtxt(directory + file_name)
    birth_epochs = pop_2_data[:,0] *1e6
    ages = pop_2_data[:,1] *1e6
    # if the age <10^6 yr, set the age to 10^6 year.
    ages [ages < 1e6 ] = 1e6

    # read in the psc and sfc files if not empty
    sfc_filepath = sfc_dir + sfc_files[i]
    if os.stat(sfc_filepath).st_size != 0:
        sfc_exists = True
        sfc_data = np.loadtxt(sfc_filepath)
        sfc_pos = sfc_data[:,0:3]
        sfc_tags = sfc_data[:,4]
    else:
        sfc_exists = False

    psc_filepath = psc_dir + psc_files[i]
    if os.stat(psc_filepath).st_size != 0:
        psc_exists = True
        psc_data = np.loadtxt(psc_filepath)
        psc_pos = psc_data[:,0:3]
        psc_tags = psc_data[:,4]
    else:
        psc_exists = False

    # if sfc_exists is True and psc_exists is True:
    #     test_particles_pos = np.vstack((sfc_pos, psc_pos))
    # elif sfc_exits is True and psc_exists is not True:
    #     test_particles_pos = sfc_pos

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
        pi_multiple = rotation_interval[i]
        rotation_angle = pi_multiple*np.pi
        # along (x,y,z) axis
        r = R.from_rotvec(rotation_angle * np.array([0,1,0]))
        rotation_matrix = r.as_matrix()
        # rotate stars
        rotated_star_positions = np.dot(star_positions, rotation_matrix.T)
        star_positions = rotated_star_positions
        # rotate test particles
        if psc_exists is True:
           psc_pos = np.dot(psc_pos, rotation_matrix.T)
           psc_pos = np.hstack((psc_pos, np.expand_dims(psc_tags,axis=1)))
        if sfc_exists is True:
           sfc_pos = np.dot(sfc_pos, rotation_matrix.T)
           sfc_pos = np.hstack((psc_pos, np.expand_dims(psc_tags,axis=1)))
    else:
        pi_multiple = 0

    if plot_test_parts is True:
        if sfc_exists is True:
           star_luminosity_plot(proj_width=proj_width ,
                                 star_positions=star_positions,
                                 scaled_stellar_lums=scaled_stellar_lums,
                                 time=time,
                                 snapshot_num=snapshot_num,
                                 pi_multiple=pi_multiple,
                                 sfc_positions=sfc_pos
                                 #need the tag, untrimmed array
                                 )
        if psc_exists is True:
           star_luminosity_plot(proj_width=proj_width ,
                                 star_positions=star_positions,
                                 scaled_stellar_lums=scaled_stellar_lums,
                                 time=time,
                                 snapshot_num=snapshot_num,
                                 pi_multiple=pi_multiple,
                                 psc_positions=psc_pos
                                 )
    else:
        star_luminosity_plot(proj_width=proj_width ,
                              star_positions=star_positions,
                              scaled_stellar_lums=scaled_stellar_lums,
                              time=time,
                              snapshot_num=snapshot_num,
                              pi_multiple=pi_multiple
                              )

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
    plt.show()
    #plt.close()
    print("# saved:", save_name)

    # after rotating as the stars are forming, rotate the rest of the set
    # rotation interval vector using the last snapshot-- frozen in time
# =============================================================================
#
#     if (np.size(files) - 1) == i:
#         print("# time frozen")
#
#         # frozen rotation
#         frozen_rotation_interval = np.linspace(2,4,np.size(files))
#
#         for remaining_pi_multiple in frozen_rotation_interval[1:]: # stutter
#             # for remaining_pi_multiple in rotation_interval[i+1:]:
#
#             stellar_lums = look_up_table(
#                 stellar_ages=ages,
#                 table_link='https://www.stsci.edu/science/starburst99/data/l1500_inst_e.dat',
#                 column_idx=1)
#             scaled_stellar_lums = stellar_lums*1e-5
#
#             #pi_multiple = remaining_pi_multiple
#             rotation_angle = remaining_pi_multiple*np.pi
#             star_positions = pop_2_data[:,2:5]
#
#             # along (x,y,z) axis
#             r = R.from_rotvec(rotation_angle * np.array([0,1,0]))
#             rotation_matrix = r.as_matrix()
#             rot_star_positions = np.dot(star_positions, rotation_matrix .T)
#
#             star_luminosity_plot(proj_width=proj_width ,
#                                  star_positions=rot_star_positions,
#                                  scaled_stellar_lums=scaled_stellar_lums,
#                                  time=time,
#                                  snapshot_num=snapshot_num,
#                                  pi_multiple=remaining_pi_multiple)
#
#             save_name = './sequences/{}/rot_{}_t_{}_pi_{}.png'.format(
#                 seq_folder_name,
#                 str(snapshot_num).zfill(4),
#                 str(time).ljust(6, '0').replace('.','_'),
#                 str(np.round(remaining_pi_multiple,3)).ljust(5, '0').replace('.','_'))
#             plt.savefig(
#                 save_name,
#                 dpi=200,
#                 bbox_inches='tight',
#                 pad_inches=0.05
#                 )
#             # plt.show()
#             plt.close()
#             print("# saved:", save_name)
# =============================================================================


