import warnings
import os
import yt
import numpy as np
from yt.funcs import mylog
from matplotlib import cm
from macros import code_age_to_yr, succ_distance

mylog.setLevel(40)
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

#---------------------------------data directory/info file---------------------
# datadir = os.path.expanduser(
#     'G:/My Drive/Research/AstrophysicsSimulation/DesktopEnvironment/data_globular_cluster/refine')  
datadir = os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine') 

# local save path 
# parent_folder = 'C:/Users/144/Desktop/AstroSimulationResearch/cluster_evolution_fs07'
# sequence_folder = 'test_frames'

#---------------------------------save path---------------------
##### cluster save path ######
parent_folder = '/homes/fgarcia4/analysis/cluster_evolution_fs07/sequences/new_refine'
sequence_folder = 'continued-zlim-changed-113021'
newpath = parent_folder + '/' + sequence_folder
if not os.path.exists(newpath):
    os.makedirs(newpath)
#----------------------------------------------------------------
# plot params
sequence_title = 'fs07-200-208'
width = (610,'pc')
axis = 'z'
start_step = 200
end_step = 208
#ctr_shift_thresh = 0.00060 #code length
ctr_shift_thresh =  0.000005 #code length
max_density_coords = []

star_map = cm.get_cmap('jet')
cmap = star_map(np.linspace(0, 1, 10))
cmap = np.flip(cmap, axis=0)
#---------------------------------MAIN LOOP-----------------------------------
for loop_num, output_num in enumerate(range(start_step, end_step + 1)) :
    print ("----------------------------------------------------------------------------------")
    infofile = os.path.abspath (datadir + "/output_%05d/info_%05d.txt" % (output_num,output_num))
    print ("# reading in: %s" %infofile)  
    
    #cell fields
    FIELDS = ["Density",
              "x-velocity", 
              "y-velocity", 
              "z-velocity",
              "Pressure",
              "Metallicity",
              "dark_matter_density",
              "xHI", "xHII", "xHeII", "xHeIII"]
    #extra particle fields
    EPF= [('particle_family', 'b'),      
          ('particle_tag', 'b'),         
          ('particle_birth_epoch', 'd'), 
          ('particle_metallicity', 'd')] 
    
    ds = yt.load(infofile, fields=FIELDS, extra_particle_fields=EPF)

    redshft = ds.current_redshift
    current_hubble = ds.hubble_constant  
    
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
    if loop_num == 0:
        max_density_coords.append(max_density_coord)
    distance = succ_distance(max_density_coord, max_density_coords[-1])
    print('\n> distance b/w current and previously used max density:', distance)
    if distance < ctr_shift_thresh: 
        p = yt.ProjectionPlot(
            ds, axis, "density", width = width, center = max_density_coord
            )
        x_pos = np.array(ad['star', 'particle_position_x']) - max_density_coord[0]
        y_pos = np.array(ad['star', 'particle_position_y']) - max_density_coord[1]
        z_pos = np.array(ad['star', 'particle_position_z']) - max_density_coord[2]
        # if the plot center migrates, annotate previous center
        # p.annotate_marker(
        #     max_density_coords[-1],
        #     marker="x",
        #     coord_system="data",
        #     plot_args={"color": "lime", "s": 30},)
        # appened new center to list
        max_density_coords.append(max_density_coord)
        
        print('> plot centered at {}'. format(max_density_coord)) 
    else: 
        center = max_density_coords[-1]
        p = yt.ProjectionPlot(
            ds, axis, "density", width = width, center = max_density_coords[-1]
            )
        x_pos = np.array(ad['star', 'particle_position_x']) - max_density_coords[-1] [0]
        y_pos = np.array(ad['star', 'particle_position_y']) - max_density_coords[-1] [1]
        z_pos = np.array(ad['star', 'particle_position_z']) - max_density_coords[-1] [2]
        
        print('> using old center at {}'. format(center)) 
        

 
    p.annotate_timestamp(corner='lower_left', 
                          time_format='t = {time:.2f} {units}', 
                          time_unit= 'Myr', 
                          redshift=True) 
    p.annotate_scale(corner='lower_right')
    p.set_cmap('density', 'magma')
    p.set_zlim('density', 0.01, .15)
    p.set_log(("gas", "density"), False)
    print('annotating', np.array(be_star).size, 'star particles')
    #p.annotate_particles(width=width, ptype='star', p_size=10.0,marker='.',col='r') 
    
    # particle clumps by age; converts code age to relative ages  
    unique_birth_epochs = code_age_to_yr(
        ad['star', 'particle_birth_epoch'], current_hubble
        ) 
    unique_birth_epochs = np.unique(np.round_(unique_birth_epochs, 0))
    converted_unfiltered = code_age_to_yr(
        ad['star', 'particle_birth_epoch'], current_hubble, unique=False
        )
    # treats all clusters within 1 Myr birth epoch as same birth epoch
    converted_unfiltered = np.round_(converted_unfiltered, 0)
    
    p.annotate_text((0.68, 0.92), 
                    "Birth Epochs: {}".format(len(unique_birth_epochs)), 
                    coord_system="figure")
    
    for i,unique_age in enumerate(unique_birth_epochs):
        print(unique_age)
        mask = np.array(converted_unfiltered) == unique_age 
        filtered_x = ds.arr(x_pos, "code_length").to('pc') [mask] 
        filtered_y = ds.arr(y_pos, "code_length").to('pc') [mask]
        color = cmap[i]
        color = color.reshape(1,-1)
        p["gas", "density"].axes.scatter(filtered_x, 
                                          filtered_y, 
                                          marker=".", 
                                          c=color,
                                          s=.25) 

  
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


     
    
    # import matplotlib as mpl
    
    # cmap = mpl.cm.cool
    # norm = mpl.colors.Normalize(vmin=5, vmax=10)
    # cbaxes = inset_axes(p["gas", "density"].axes, width="30%", height="3%", loc=3)  
    # p["gas", "density"].colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #              cax=cbaxes, orientation='horizontal', label='Some Units')
    
    # cbaxes = inset_axes(p["gas", "density"].axes, width="30%", height="3%", loc=3) 
    # plt.colorbar(cax=cbaxes, ticks=[0.,1], orientation='horizontal')

    # fig, ax = plt.subplots()
    # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
    # fig.colorbar(p["gas", "density"].axes, cax=cax, orientation='horizontal')

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
    p.save(save_path, mpl_kwargs=dict(dpi=250))
    print('#saved:', save_path)