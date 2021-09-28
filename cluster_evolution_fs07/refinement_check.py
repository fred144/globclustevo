import warnings
import os
import yt
import numpy as np
from yt.funcs import mylog

mylog.setLevel(40)
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

#----------------------------------------------------------------------------
#data directory/info file
#datadir = os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_rerun') 
datadir = os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine') 
#datadir = os.path.expanduser('G:/My Drive/Research/AstrophysicsSimulation/DesktopEnvironment/data_globular_cluster') 
#----------------------------------------------------------------------------

start_step = 125
end_step = 125
for output_num in range (start_step, end_step + 1):
    
    infofile = os.path.abspath (datadir + "/output_%05d/info_%05d.txt" % (output_num,output_num))
    print ("#Reading in info file: %s" %infofile)    

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

    ds = yt.load(infofile, fields=FIELDS, extra_particle_fields=EPF)
    z = ds.current_redshift
    
    #get SFC/PSC positions
    ad = ds.all_data()
    pos_SFCs = ad['SFC', 'particle_position']
    pos_PSCs = ad['PSC', 'particle_position']
    
    width = (20,'pc')
    p = yt.SlicePlot(ds, 'z', "density", width = width, center = ('max','density'))
    
    if pos_SFCs.size > 0:
        first_sfc_center = pos_SFCs[0]  
        p = yt.SlicePlot(ds, 'z', "density", width = width, center = first_sfc_center)
        p.annotate_particles(width = width,
                             ptype='SFC', 
                             p_size=100.0,
                             marker='x',col='b') 
    else:
        p = yt.SlicePlot(ds, 'z', "density", width = width, center = ('max','density'))
        
    if pos_PSCs.size > 0: 
        p.annotate_particles(width = width,
                             ptype='PSC', 
                             p_size=100.0,
                             marker='x',col='k')   
    
    p.annotate_particles(width=width, ptype='star', p_size=20.0,marker='.',col='r')    
    
    p.annotate_timestamp(corner='lower_left', 
                         time_format='t = {time:.3f} {units}', 
                         time_unit= 'Myr', 
                         redshift=True, 
                         draw_inset_box=True)
    p.annotate_scale(corner='lower_right', draw_inset_box= True)

    p.annotate_cell_edges() 
    p.set_cmap("density", "magma")

    sequence_folder = "refinement_check"
    sequence_title = "New Refinement"
    parent_folder = '/homes/fgarcia4/analysis/cluster_evolution_fs07/sequences/'
    
    p.annotate_title(sequence_title, 
                     str(output_num), 
                     '| Red = Pop II, Blue = SFC, Black = PSC ')
    
    save_path = "{}{}/output-{}-z-{}-{}.png".format(parent_folder,
                                                sequence_folder,
                                                str(output_num).zfill(5),
                                                str(round(z, 2)).replace('.', '_'),
                                                sequence_title )
    p.save(save_path)
    print('Saved:', save_path)


    
    
    
