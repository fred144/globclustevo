import warnings
import os
import yt
import numpy as np
import scipy as sc
from yt.funcs import mylog

mylog.setLevel(40)
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

def succ_distance (current, previous):
    dist = np.linalg.norm(np.array(current) - np.array(previous)) 
    return dist 

#data directory/info file
#datadir = os.path.expanduser('G:/My Drive/Research/AstrophysicsSimulation/DesktopEnvironment/data_globular_cluster/') 
#datadir = os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_rerun') 
datadir = os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine') 

parent_folder = '/homes/fgarcia4/analysis/cluster_evolution_fs07/sequences/new_refine'
sequence_folder = 'z_proj_den_cmap_2'

#make new folder
newpath = parent_folder + '/' + sequence_folder
if not os.path.exists(newpath):
    os.makedirs(newpath)

sequence_title = 'Z Projected Density'
width = (690,'pc')
axis = 'z'
start_step = 100
end_step = 235

ctr_shift_thresh =  0.00065 #code length

max_density_coords = []

for loop_num, output_num in enumerate(range(start_step, end_step + 1)) :
    print ("-")
    infofile = os.path.abspath (datadir + "/output_%05d/info_%05d.txt" % (output_num,output_num))
    print ("#reading in info file: %s" %infofile)  

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
    
    print('reading fields...')
    
    ds = yt.load(infofile, fields=FIELDS, extra_particle_fields=EPF)
    redshft = ds.current_redshift
    
    #get SFC/PSC positions and other important fields
    ad = ds.all_data()
    pos_SFCs = ad['SFC', 'particle_position']
    pos_PSCs = ad['PSC', 'particle_position']
    be_star = ad['star', 'particle_birth_epoch']
    
    max_den = ad.argmax(('gas', 'density'))
    # x = x.in_units('pc')
    # y = y.in_units('pc')
    # z = z.in_units('pc')
    max_density_coord = yt.YTArray(max_den).in_units('code_length') 
    #max_density_coord = np.array(max_density_coord)
    max_density_coord = np.array(max_density_coord)
    #print('coords', max_density_coord)
    

    #keep center of plots relatively stable
    if loop_num == 0:
        max_density_coords.append(max_density_coord)
    
    distance = succ_distance(max_density_coord, max_density_coords[-1])
    
    print('> distance b/w current and previously used max density:', distance)
    
    if distance < ctr_shift_thresh: 
        p = yt.ProjectionPlot(ds, axis, "density", width = width, center = max_density_coord)
        
        # if the plot center migrates, annotate previous center
        p.annotate_marker(
            max_density_coords[-1],
            marker="x",
            coord_system="data",
            plot_args={"color": "lime", "s": 30},)

        # appen new center to list
        max_density_coords.append(max_density_coord)

        print('> plot centered at {}'. format(max_density_coord)) 
    else: 
        center = max_density_coords[-1]
        p = yt.ProjectionPlot(ds, axis, "density", width = width, center = max_density_coords[-1])
        print('> using old center at {}'. format(center)) 
        
        
    print('annotating', np.array(be_star).size, 'star particles')
    if pos_SFCs.size > 0:
#         first_sfc_center = pos_SFCs[0]  
#         p = yt.SlicePlot(ds, 'z', "density", width = width, center = first_sfc_center) #turn this on to track center of sfc
        p.annotate_particles(width = width,
                             ptype='SFC', 
                             p_size=100.0,
                             marker='x',col='b') 
    if pos_PSCs.size > 0: 
        p.annotate_particles(width = width,
                             ptype='PSC', 
                             p_size=100.0,
                             marker='x',col='k')
#         if pos_SFCs.size == 0: 
#             first_psc_center = pos_PSCs[0] 
#             p = yt.SlicePlot(ds, 'z', "density", width = width, center = first_psc_center) #turn this on to track center of psc
        
            
    p.annotate_particles(width=width, ptype='star', p_size=20.0,marker='.',col='r')    
    
    p.annotate_timestamp(corner='lower_left', 
                         time_format='t = {time:.2f} {units}', 
                         time_unit= 'Myr', 
                         redshift=True) 
                         #draw_inset_box=True)
    p.annotate_scale(corner='lower_right')#, draw_inset_box= True)

    #p.annotate_cell_edges() 
    p.set_cmap('density', 'magma')
    p.set_zlim('density', 0.01, .5)

    plot_title = str( "{} {}| Red = Pop II, Blue = SFC, Black = PSC ".format(sequence_title, output_num) ) 

    p.annotate_title(plot_title)
    
    save_path = str ("{}/{}/output-{}-z-{}-{}.png".format(parent_folder,
                                                sequence_folder,
                                                str(output_num).zfill(5),
                                                str(np.around(redshft, 2)).replace('.', '_'),
                                                sequence_title.replace(' ','-'))
                      )
    p.save(save_path, mpl_kwargs=dict(dpi=200))
    print('#saved:', save_path)

    
