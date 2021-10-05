import warnings
import os
import yt
import numpy as np
from yt.funcs import mylog

mylog.setLevel(40)
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

def succ_distance (current, previous): 
    current = np.array(current) 
    previous = np.array(previous) 
    dist = np.linalg.norm(current - previous) 
    return dist 

#data directory/info file
#datadir = os.path.expanduser('G:/My Drive/Research/AstrophysicsSimulation/DesktopEnvironment/data_globular_cluster/') 
#datadir = os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_rerun') 
datadir = os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine') 

parent_folder = '/homes/fgarcia4/analysis/cluster_evolution_fs07/sequences'
sequence_folder = 'cl_re_centered_refine'
sequence_title = 'New Centering - Z Density'
width = (310,'pc')
slice_axis = 'z'
start_step = 116
end_step = 130

ctr_shift_thresh = 200000 #pc

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
    
    # x, y, z = ad.argmax(('gas', 'density'))
    # x = x.in_units('pc')
    # y = y.in_units('pc')
    # z = z.in_units('pc')
    max_density_coord = yt.YTArray(ad.argmax('density')).in_units('code_length')
    #max_density_coord = (x, y, z) 

    #keep center of plots relatively stable
    if loop_num == 0:
        max_density_coords.append(max_density_coord)
    
    succ_distance_pc = succ_distance(max_density_coord, max_density_coords[-1])
    
    print('distance b/w current and previously used max density:', succ_distance_pc)
    
    if succ_distance_pc < ctr_shift_thresh: 
        p = yt.SlicePlot(ds, slice_axis, "density", width = width, center = max_density_coord)
        max_density_coords.append(max_density_coord)
        print('centered at {}'. format(max_density_coord)) 
    else: 
        center = max_density_coords[-1] 
        p = yt.SlicePlot(ds, slice_axis, "density", width = width, center = center)
        print('centered at {}'. format(center)) 
        
        
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
                         redshift=True, 
                         draw_inset_box=True)
    p.annotate_scale(corner='lower_right', draw_inset_box= True)

    #p.annotate_cell_edges() 
    p.set_cmap("density", "magma")

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

    
