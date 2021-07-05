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

#data directory/info file
datadir=os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_rerun')

startSlice = 50
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
    
    sp = ds.sphere('max', (50, "pc"))

    L = sp.quantities.angular_momentum_vector()

    print("Angular momentum vector: {0}".format(L))

    p = yt.OffAxisSlicePlot( ds,  L, "density", center = sp.center,
                             width = (300, "pc"), north_vector = [0,1,0])
    #p.annotate_particles((100, 'Mpc'),ptype='star', p_size=20.0,marker='.',col='r')
    p.annotate_timestamp(corner='upper_left', time_format='t = {time:.3f} {units}', time_unit= 'Myr', 
                         redshift=True, draw_inset_box=True)
    p.annotate_scale(corner='lower_right', draw_inset_box= True)
    p.annotate_title('Density Face-on View | '+ str(outputNum) + " of 227")

    #p.show()






    p.save('/homes/fgarcia4/analysis/cluster_evolution_fs07/sequences/face-on/' + str(outputNum).zfill(3) )

    print('Plot' + str(outputNum) + "saved.")
    
