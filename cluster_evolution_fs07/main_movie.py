import sys
sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/cosmology-clean/lib/python3.7/site-packages"
    ) 
import warnings
import os  
import pathlib
import yt
import numpy as np
from yt.funcs import mylog
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from macros import code_age_to_yr, succ_distance

mylog.setLevel(40)
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

#---------------------------------data directory/info file---------------------
# datadir = os.path.realpath('/home/fabg/cosm_test_data/refine')
datadir = os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine') 

#---------------------------------save path------------------------------------
# local save path 
# parent_folder = 'C:/Users/144/Desktop/AstroSimulationResearch/cluster_evolution_fs07'
# parent_folder = '.'
# sequence_folder = 'test_frames'

#####------------------------DT2 save path-------------------------------######
parent_folder = '/homes/fgarcia4/analysis/cluster_evolution_fs07/sequences/new_refine'
sequence_folder = 'projden_z_rev3_113_475'
newpath = parent_folder + '/' + sequence_folder
if not os.path.exists(newpath):
    os.makedirs(newpath)
    
#------------------------------------plot params-------------------------------

sequence_title = 'z_proj_rev3'
width = (400,'pc')
slice_axis = 'z'
start_step = 113
end_step = 475
#ctr_shift_thresh = 0.00060 #code length
ctr_shift_thresh =  0.000001 #code length
max_density_coords = []

# snapshot 115 to 452 roughly spans 340 to 470 myr
star_map = cm.get_cmap('autumn')
time_range = (339,480) #Myr
evenly_spaced_times = np.arange(time_range[0], time_range[1]  + 1)
cmap = star_map(np.linspace(0, 1, time_range[1] - time_range[0]))

#cosmetics
mpl.rc('font', family='serif') 

#---------------------------------MAIN LOOP-----------------------------------
for loop_num, output_num in enumerate(range(start_step, end_step + 1)) :
    print ("-------------------------------------------------------")
    infofile = os.path.abspath(
        datadir + f"/output_{output_num:05}/info_{output_num:05}.txt"
        )
    print ("# reading in", infofile )  

    #cell fields
    FIELDS = ['Density',
              'x-velocity', 
              'y-velocity', 
              'z-velocity',
              'Pressure',
              'Metallicity',
              'dark_matter_density',
              'xHI', 
              'xHII', 
              'xHeII', 
              'xHeIII'
              ]
    #extra particle fields
    EPF= [('particle_family', 'b'),      
          ('particle_tag', 'b'),         
          ('particle_birth_epoch', 'd'), 
          ('particle_metallicity', 'd')
          ] 
    
    ds = yt.load(infofile, fields=FIELDS, extra_particle_fields=EPF)

    redshft = ds.current_redshift
    current_hubble = ds.hubble_constant  
    current_time = float(ds.current_time.in_units('Myr'))
    # extract all data fields
    ad = ds.all_data()
    
    # get SFC/PSC positions and other important fields
    pos_SFCs = ad['SFC', 'particle_position']
    pos_PSCs = ad['PSC', 'particle_position']
    be_star = ad['star', 'particle_birth_epoch']
    
    # centering using max density 
    max_den = ad.argmax(('gas', 'density'))
    max_density_coord = yt.YTArray(max_den).in_units('code_length') 
    max_density_coord = np.array(max_density_coord)
   
    # keep center at density max
    # if loop_num == 0:
    #     max_density_coords.append(max_density_coord)
    # distance = succ_distance(max_density_coord, max_density_coords[-1])
    # print('\n> distance b/w current and previously used max density:', distance)
    # if distance < ctr_shift_thresh: 
    #     p = yt.ProjectionPlot(
    #         ds, slice_axis, "density", width = width, center = max_density_coord
    #         )
    #     x_pos = np.array(ad['star', 'particle_position_x']) - max_density_coord[0]
    #     y_pos = np.array(ad['star', 'particle_position_y']) - max_density_coord[1]
    #     z_pos = np.array(ad['star', 'particle_position_z']) - max_density_coord[2]
    #     # if the plot center migrates, annotate previous center
    #     # p.annotate_marker(
    #     #     max_density_coords[-1],
    #     #     marker="x",
    #     #     coord_system="data",
    #     #     plot_args={"color": "lime", "s": 30},)
    #     # appened new center to list
    #     max_density_coords.append(max_density_coord)
        
    #     print('> plot centered at {}'. format(max_density_coord)) 
    # else: 
    #     center = max_density_coords[-1]
    #     p = yt.ProjectionPlot(
    #         ds, slice_axis, "density", width = width, center = max_density_coords[-1]
    #         )
    #     x_pos = np.array(ad['star', 'particle_position_x']) - max_density_coords[-1] [0]
    #     y_pos = np.array(ad['star', 'particle_position_y']) - max_density_coords[-1] [1]
    #     z_pos = np.array(ad['star', 'particle_position_z']) - max_density_coords[-1] [2]
        
    #     print('> using old center at {}'. format(center)) 

 
    # keep centered at mean particle positons
    x_pos = np.array(ad['star', 'particle_position_x']) 
    y_pos = np.array(ad['star', 'particle_position_y']) 
    z_pos = np.array(ad['star', 'particle_position_z']) 
    
    x_center = np.mean(x_pos)
    y_center = np.mean(y_pos)
    z_center = np.mean(z_pos)
    
    plt_ctr = np.array([x_center, y_center, z_center])

    p = yt.ProjectionPlot(
                          ds, 
                          slice_axis, 
                          ('gas', 'density'), 
                          width = width, 
                          center = plt_ctr
                          )

    #aesthetics
    p.set_font({'family': 'serif', 'size': 14,})
    p.annotate_timestamp(corner='upper_left', 
                          time_format='t = {time:.2f} {units}', 
                          time_unit= 'Myr', 
                          redshift=True) 
    p.annotate_scale(corner='lower_right', 
                     coeff=100, 
                     unit='pc',
                     text_args={'size':12, 'family':'serif'}
                     )
    p.set_cmap('density', 'inferno') 
    p.set_zlim('density', 0.005, .34)       
    p.set_log(('gas', 'density'), False)
    p.set_colorbar_label(
        ('gas', 'density'), r'Projected Gas Density (g cm$^{-2}$)'
        )
    p.hide_axes(draw_frame=True)
    # width = (1.0, "unitary")
    # res = [1000, 1000]
    # frb = p.to_frb(width, res)
    
    
    print('> annotating', np.array(be_star).size, 'star particles')
    #p.annotate_particles(width=width, ptype='star', p_size=10.0,marker='.',col='r') 
    # particle clumps by age; converts code age to relative ages  
    unique_birth_epochs = code_age_to_yr(
        ad['star', 'particle_birth_epoch'], current_hubble
        ) 
    # gets the clump ages, treats all clumps within 1 Myr as the from same  
    unique_birth_epochs = np.unique(np.round_(unique_birth_epochs, 0))
    
    # all the birth epochs of the stars 
    converted_unfiltered = code_age_to_yr(
        ad['star', 'particle_birth_epoch'], current_hubble, unique=False
        )
    # treats all clusters within 1 Myr birth epoch as same birth epoch
    # the first output with star in it was t = 339.562
    # have yet to figure out how to calculate absolute times
    # just relative for now 
    converted_unfiltered_rounded = np.round_(converted_unfiltered, 0) + 339.562
    
    # pop II annotate loop
    x_pos = x_pos - plt_ctr[0] 
    y_pos = y_pos - plt_ctr[1] 
    z_pos = z_pos - plt_ctr[2] 
    
    for i,unique_age in enumerate(unique_birth_epochs + 339.562 ):
        print(unique_age)
        mask = np.array(converted_unfiltered_rounded) == unique_age 
        filtered_x = ds.arr(x_pos, 'code_length').to('pc') [mask] 
        filtered_y = ds.arr(y_pos, 'code_length').to('pc') [mask] 
        filtered_z = ds.arr(z_pos, 'code_length').to('pc') [mask] 
        
        idx_of_nearest_c = np.argmin(np.abs(evenly_spaced_times - unique_age))
        #print(idx_of_nearest_c)
        color = cmap[idx_of_nearest_c]
        color = color.reshape(1,-1)
        
        if slice_axis == 'z': 
            p['gas', 'density'].axes.scatter(
                filtered_x, filtered_y, marker='.', c=color, s=.2, alpha=0.1
                ) 
        elif slice_axis == 'x':
            p['gas', 'density'].axes.scatter(
                filtered_y, filtered_z, marker='.', c=color, s=.2, alpha=0.1
                ) 
        elif slice_axis == 'y':
            p['gas', 'density'].axes.scatter(
                filtered_z, filtered_x, marker='.', c=color, s=.2, alpha=0.1
                ) 
        else:
            print('Invalid slice axis.')
            
   
    #cbar_fig.style.use('dark_background') 
    
    # pop II age color bar 
    cbar_fig = p.plots[('gas', 'density')].figure
    ax = cbar_fig.add_axes([0.31, 0.91, 0.3, 0.015])
    cb = mpl.colorbar.ColorbarBase(
        ax,  
        norm = mpl.colors.Normalize(time_range[0], time_range[1]),
        #ticks = [340,405,470], 
        orientation='horizontal', 
        cmap='autumn', 
        #label='Birth Epoch (Myr)'
        ) 
    cb.ax.tick_params(colors='white', labelsize=6)
    for t in cb.ax.xaxis.get_ticklabels():
        t.set_family("serif")
    ax.set_title(
        "Pop II Birth Time (Myr) | Epochs: {}".format(len(unique_birth_epochs)),                 
        c='white', 
        fontsize=9,
        fontfamily='serif'
        )
    
    # axis guide
    p_ax = p.plots[('gas', 'density')].axes 
    if slice_axis == 'z':
        p_ax.text(-150, -185, 
                'x', 
                size=7, 
                ha='center', 
                va='center', 
                color='white')
        p_ax.text(-185, -150, 
                'y', 
                size=7, 
                ha='center', 
                va='center', 
                color='white')   
    elif slice_axis == 'x': 
        p_ax.text(-150, -185, 
                'y', 
                size=7, 
                ha='center', 
                va='center', 
                color='white')
        p_ax.text(-185, -150, 
                'z', 
                size=7, 
                ha='center', 
                va='center', 
                color='white') 
    elif slice_axis == 'y':
        p_ax.text(-150, -185, 
                'z', 
                size=7, 
                ha='center', 
                va='center', 
                color='white')
        p_ax.text(-185, -150, 
                'x', 
                size=8, 
                ha='center', 
                va='center', 
                color='white')
    else:
        print('Invalid slice axis.')
    p_ax.arrow(-185, -185, 30, 0,    
              head_width=3, 
              head_length=3, 
              linewidth=0.5, 
              color='w', 
              length_includes_head=True)
    p_ax.arrow(-185, -185, 0, 30,    
              head_width=3, 
              head_length=3, 
              linewidth=0.5, 
              color='w', 
              length_includes_head=True) 
    
    
    # luminosity mappping data extraction 
    
    # get star positons 
    abs_birth_epochs = np.round(converted_unfiltered + 339.562, 3)
    current_ages = np.round(current_time, 3) - np.round(abs_birth_epochs, 3)
    t_myr = np.array([current_time]) 
    t_myr.resize(np.size(current_ages))
    star_info = np.array(
          [
          abs_birth_epochs,
          current_ages,
          ds.arr(x_pos, 'code_length').to('pc'), 
          ds.arr(y_pos, 'code_length').to('pc'), 
          ds.arr(z_pos, 'code_length').to('pc'), 
          ds.arr(ad['star', 'particle_mass'], 'code_mass').to('msun'),
          t_myr
          ]
          )
    
    # luminosity mappping save
    star_info = np.array(star_info).T
    save_path_star_pos = str(pathlib.Path(os.getcwd()).parents[0])
    save_time = str(format(current_time, '.2f')).replace('.', '_')
    save_name = "/luminosity_mapping/pop_2_data/pos_{:05d}_{}_myr.txt".format(
          output_num,save_time)
    np.savetxt(fname=save_path_star_pos+save_name, X=star_info)
     
    
    
    # if pos_SFCs.size > 0:
    #     p.annotate_particles(width = width,
    #                           ptype='SFC', 
    #                           p_size=10,
    #                           marker='x', col='b') 
    # if pos_PSCs.size > 0: 
    #     p.annotate_particles(width = width,
    #                           ptype='PSC', 
    #                           p_size=10,
    #                           marker='x', col='r')

    # from yt.analysis_modules.halo_analysis.api import HaloCatalog

    # hc = HaloCatalog(data_ds=ds, finder_method='hop',
    #                   finder_kwargs={"threshold": 100, #default: 160
    #                                   "ptype":'DM',
    #                                   "dm_only":False})
    
    # hc = HaloCatalog(data_ds=ds, finder_method='fof',
    #                   finder_kwargs={"ptype": 'DM',
    #                                 "link": 0.2,
    #                                 "dm_only":False})
    
    # hc.create()
    # hc_ad = hc.halos_ds.all_data()
    # p.annotate_halos(hc, 
    #                   width=width,
    #                   factor = 0.03) 


    save_path = str (
        "{}/{}/out-{}-z-{}-{}.png".format(
            parent_folder,
            sequence_folder,
            str(output_num).zfill(5),
            str(format(redshft, '.4f')).replace('.', '_'), 
            sequence_title.replace(' ','-')
            )
                      )
    p.save(save_path, 
           mpl_kwargs={
                       'bbox_inches': 'tight', 
                       'dpi':250, 
                       'pad_inches':0.1
                       #'facecolor': 'black'
                       }
           )
    
    print('# saved:', save_path)
