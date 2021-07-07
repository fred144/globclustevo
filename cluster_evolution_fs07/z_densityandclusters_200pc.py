# import library
import warnings
import sys
import os
import matplotlib 
import yt
import time
import pylab as pl
from IPython import display

import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import shutil

from yt.funcs import mylog
mylog.setLevel(40)
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

#----------------------------------------------------------------------------
#data directory/info file
datadir=os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_rerun') 
#----------------------------------------------------------------------------

startSlice = 116
endSlice = 227
for outputNum in range (startSlice, endSlice + 1):
    
    infofile = os.path.abspath (datadir + "/output_%05d/info_%05d.txt" % (int(outputNum), int(outputNum)))
    print ("#Reading in info file: %s" %infofile)    
    '''
    #create and move to analysis directory
    anldir = os.path.abspath(datadir+"/analysis_%05d"%(int(step)))
    if not os.path.isdir(anldir):
        print ("create %s"%anldir)
        os.makedirs(anldir)
    print ("move to %s"%anldir)                          
    os.chdir(anldir)                          
    '''
    #cell fields
    FIELDS = ["Density",
              "x-velocity", "y-velocity", "z-velocity",
              "Pressure",
              "Metallicity",
              "dark_matter_density",
              "xHI", "xHII", "xHeII", "xHeIII"]
  
    #extra particle fields
    EPF= [('particle_family', 'b'),      #byte size
          ('particle_tag', 'b'),         #byte size
          ('particle_birth_epoch', 'd'), #double size
          ('particle_metallicity', 'd')] #double size


    ds = yt.load(infofile, fields=FIELDS, extra_particle_fields=EPF)
    
    #plot                                                                                                                                                                                                     
    width = (200,'pc') #plot width                                                                                                                                                                                
    
    from yt.analysis_modules.halo_analysis.api import HaloCatalog

    hc = HaloCatalog(data_ds=ds, finder_method='fof',
                 finder_kwargs={"ptype": "star",
                                 "padding": .2,
                                "link": 0.0002,
                                 "dm_only":False})
    
    #get SFC/PSC positions
    ad = ds.all_data()
    pos_SFCs = ad['SFC', 'particle_position']
    pos_PSCs = ad['PSC', 'particle_position']

    p = yt.SlicePlot(ds, 'z', "density", width = width, center = ('max','density'))
    
    if pos_SFCs.size > 0 or pos_PSCs.size > 0:

        if pos_PSCs.size > 0:
            center = pos_PSCs[0]
            p = yt.SlicePlot(ds, 'z', "density", width = width, center = center)

            p.annotate_particles(width=width,ptype='PSC', p_size=100.0,marker='x',col='k') #passive stellar clusters (test particles) 
            p.annotate_particles(width=width, ptype='star', p_size=15.0,marker='.',col='r') #Pop II stars
            p.annotate_particles(width=width,ptype='SFC', p_size=100.0,marker='x',col='b') #star forming clouds (test particles)
        
        if pos_SFCs.size > 0: 
            center = pos_SFCs[0]
            p = yt.SlicePlot(ds, 'z', "density", width = width, center = center)

            p.annotate_particles(width=width,ptype='PSC', p_size=100.0,marker='x',col='k') #passive stellar clusters (test particles)
            p.annotate_particles(width=width, ptype='star', p_size=15.0,marker='.',col='r') #Pop II stars
            p.annotate_particles(width=width,ptype='SFC', p_size=100.0,marker='x',col='b') #star forming clouds (test particles)      

        hc.create()
        hc_ad = hc.halos_ds.all_data()
        p.annotate_halos(hc, width=width)
        p.annotate_text((0.175, 0.2), str( len(hc_ad['particle_mass'])) + " Halos", coord_system="figure")
        
               
    p.annotate_timestamp(corner='lower_left', time_format='t = {time:.3f} {units}', time_unit= 'Myr', redshift=True, draw_inset_box=True)
    p.annotate_scale(corner='lower_right', draw_inset_box= True)

    #p.set_cmap("density", "viridis")

    p.annotate_title(' Z Density and Star Clusters | '  + str(outputNum) + ' of ' + str(endSlice) + 
                     ' | Red = Pop II, Blue = SFC, Black = PSC ')
    

    
    #---------------------------------------------------------------------------------------------------------
    p.save('/homes/fgarcia4/analysis/cluster_evolution_fs07/sequences/halos_z_density_200pc/')
    print ('saved frame #' + str(outputNum) + ' to: /homes/fgarcia4/analysis/cluster_evolution_fs07/sequences/halos_z_density_200pc/')
    #---------------------------------------------------------------------------------------------------------
    
    
    
    
