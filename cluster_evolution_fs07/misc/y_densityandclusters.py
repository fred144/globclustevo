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

step = 1
while step <= 227:
    infofile = os.path.abspath (datadir + "/output_%05d/info_%05d.txt" % (int(step),int(step)))
    print ("#Reading in info file: %s" %infofile)

    #cell fields
    FIELDS = ["Density",
              "x-velocity", "y-velocity", "z-velocity",
              "Pressure",
              "Metallicity",
              "dark_matter_density",
              "xHI", "xHII", "xHeII", "xHeIII"]
  
    #extra particle fields
    EPF = [('particle_family', 'b'),      #byte size
          ('particle_tag', 'b'),         #byte size
          ('particle_birth_epoch', 'd'), #double size
          ('particle_metallicity', 'd')] #double size


    ds = yt.load(infofile, fields=FIELDS, extra_particle_fields=EPF)
        #plot                                                                                                                                                                                                     
    width = (500,'pc') #plot width                                                                                                                                                                                
    
    
    #get SFC/PSC positions
    ad = ds.all_data()
    pos_SFCs=ad['SFC','particle_position']
    pos_PSCs=ad['PSC','particle_position']


    p = yt.SlicePlot(ds, 'y', "density", width = width, center = ('max','density'))
    
    if pos_SFCs.size > 0:
        center = pos_SFCs[0] #set the center of plot to the poisition of the first SFC, for example
        p = yt.SlicePlot(ds, 'y', "density", width = width, center = center )
        p.annotate_particles(width=width,ptype='star', p_size=20.0,marker='.',col='r') #Pop II stars
        p.annotate_particles(width=width,ptype='SFC', p_size=100.0,marker='x',col='b') #star forming clouds (test particles)
        p.annotate_particles(width=width,ptype='PSC', p_size=100.0,marker='x',col='k') #passive stellar clusters (test particles)

    p.annotate_timestamp(corner='lower_left', time_format='t = {time:.3f} {units}', time_unit= 'Myr', redshift=True, draw_inset_box=True)
    p.annotate_scale(corner='lower_right', draw_inset_box= True)

    #p.set_cmap("density", "viridis")

    p.annotate_title(' Y Density and Star Clusters | '  + str(step) + ' of 227 ' +
                     ' | [Red = Pop II, BlackX = PSC, BlueX = SFC]')
    
    #---------------------------------------------------------------------------------------------------------
    p.save('/homes/fgarcia4/analysis/cluster_evolution_fs07/sequences/rerun_y_density_500pc/')
    
    print ('saved frame #' + str(step) + ' to: /homes/fgarcia4/analysis/cluster_evolution_fs07/sequences/rerun_y_density_500pc/')

    #time.sleep(.001)

    step += 1
