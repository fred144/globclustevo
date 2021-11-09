import warnings
import os
import yt
import numpy as np
import sys
import functools
from yt.funcs import mylog
from clump_filters import clump_filters
from macros import code_age_to_yr, succ_distance
from matplotlib import cm

mylog.setLevel(40)
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
namespace = sys._getframe(0).f_globals

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
sequence_folder = 'hop_halo_annotated_1kpc'
newpath = parent_folder + '/' + sequence_folder
if not os.path.exists(newpath):
    os.makedirs(newpath)
    
#plot params
sequence_title = 'hop'
width = (1000,'pc')
axis = 'z'
start_step = 100
end_step = 310
ctr_shift_thresh = 0.00060 #code length

max_density_coords = []

star_map = cm.get_cmap('rainbow')
cmap = star_map (np.linspace(0, 1, 20))

#---------------------------------MAIN LOOP-----------------------------------
for loop_num, output_num in enumerate(range(start_step, end_step + 1)) :
    print ("----------------------------------------------------------------------------------")
    infofile = os.path.abspath (datadir + "/output_%05d/info_%05d.txt" % (output_num,output_num))
    print ("# reading in info file: %s" %infofile)  
    
    #cell fields
    FIELDS = ["Density",
              "x-velocity", "y-velocity", "z-velocity",
              "Pressure",
              "Metallicity",
              "dark_matter_density",
              "xHI", "xHII", "xHeII", "xHeIII"]
    #extra particle fields
    EPF= [('particle_family', 'b'),      
          ('particle_tag', 'b'),         
          ('particle_birth_epoch', 'd'), 
          ('particle_metallicity', 'd')] 
    
    print('# reading fields...')
    
    ds = yt.load(infofile, fields=FIELDS, extra_particle_fields=EPF)
    
    clump_filters(ds)
    
    redshft = ds.current_redshift
    current_hubble = ds.hubble_constant  
    
    # extract all data fields
    ad = ds.all_data()
    
    # get SFC/PSC positions and other important fields
    pos_SFCs = ad['SFC', 'particle_position']
    pos_PSCs = ad['PSC', 'particle_position']
    be_star = ad['star', 'particle_birth_epoch']
    raw_birth_epochs = ad['star', 'particle_birth_epoch'] 
    
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
        p = yt.ProjectionPlot(ds, axis, "density", width = width, center = max_density_coord)
        # if the plot center migrates, annotate previous center
        p.annotate_marker(
            max_density_coords[-1],
            marker="x",
            coord_system="data",
            plot_args={"color": "lime", "s": 30},)
        # appened new center to list
        max_density_coords.append(max_density_coord)
        print('> plot centered at {}'. format(max_density_coord)) 
    else: 
        center = max_density_coords[-1]
        p = yt.ProjectionPlot(ds, axis, "density", width = width, center = max_density_coords[-1])
        print('> using old center at {}'. format(center)) 
        
    
    # clump_tracker(ds=ds, 
    #               birth_epochs=unique_birth_epochs, 
    #               width=width, 
    #               plot_object=p)
    
    try:
        # particle clumps by age 
        unique_birth_epochs = code_age_to_yr(raw_birth_epochs, current_hubble)
        print('annotating', np.array(be_star).size, 'star particles')
        #p.annotate_particles(width=width, ptype='star', p_size=10.0,marker='.',col='r') 
        for clumpnum,age in enumerate(unique_birth_epochs, start=1):
            print('> annotating clump', age)
            clumpname = 'clump' + str(clumpnum) 
            color = cmap[clumpnum]
            color = color.reshape(1,-1)
            p.annotate_particles(width=width, 
                                  ptype=clumpname, 
                                  p_size=5.0, 
                                  marker='.',
                                  col=color) 
    except: 
        print('> no stars')
        pass
    
    

    from yt.analysis_modules.halo_analysis.api import HaloCatalog

    hc = HaloCatalog(data_ds=ds, finder_method='hop',
                      finder_kwargs={"threshold": 100,
                                      "ptype":'DM',
                                      "dm_only":False})
    
    # hc = HaloCatalog(data_ds=ds, finder_method='fof',
    #                   finder_kwargs={"ptype": 'DM',
    #                                 "link": 0.2,
    #                                 "dm_only":False})
    
    hc.create()
    hc_ad = hc.halos_ds.all_data()
    p.annotate_halos(hc, 
                     width=width,
                     factor = 0.03) 



    if pos_SFCs.size > 0:
        p.annotate_particles(width = width,
                              ptype='SFC', 
                              p_size=10,
                              marker='x', col='b') 
    if pos_PSCs.size > 0: 
        p.annotate_particles(width = width,
                              ptype='PSC', 
                              p_size=10,
                              marker='x', col='r')
 
    p.annotate_timestamp(corner='lower_left', 
                          time_format='t = {time:.2f} {units}', 
                          time_unit= 'Myr', 
                          redshift=True) 
    p.annotate_scale(corner='lower_right')
    p.set_cmap('density', 'magma')
    p.set_zlim('density', 0.01, .05)

    # plot_title = str( 
    # "{} {}| Red = Pop II, Blue = SFC, Black = PSC ".format(sequence_title, output_num) ) 
    #p.annotate_title(plot_title)
    
    save_path = str ("{}/{}/output-{}-z-{}-{}.png".format(parent_folder,
                                                sequence_folder,
                                                str(output_num).zfill(5),
                                                str(np.around(redshft, 2)).replace('.', '_'),
                                                sequence_title.replace(' ','-'))
                      )
    p.save(save_path, mpl_kwargs=dict(dpi=200))
    print('#saved:', save_path)

