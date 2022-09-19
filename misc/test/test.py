### import library
import warnings
import numpy as np
import sys
import os
import matplotlib
import yt
#%matplotlib inline
import time
import pylab as pl
from IPython import display


from yt.funcs import mylog
mylog.setLevel(40)
warnings.simplefilter(action = "ignore", category = RuntimeWarning)


#data directory/info file
datadir=os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_rerun') 

#your data directory (parent directory of output_***)
step = 125
while step <= 125:
    infofile = os.path.abspath (datadir + "/output_%05d/info_%05d.txt" % (int(step),int(step)))
    print ("info file: %s" %infofile)

    #cell fields
    FIELDS = ["Density",
              "x-velocity", "y-velocity", "z-velocity",
              "Pressure",
              "Metallicity",
              "xHI", "xHII", "xHeII", "xHeIII"]
    
    #extra particle fields
    EPF= [('particle_family', 'b'),      #byte size
          ('particle_tag', 'b'),         #byte size
          ('particle_birth_epoch', 'd'), #double size
          ('particle_metallicity', 'd')] #double size
    

    ds = yt.load(infofile, fields=FIELDS, extra_particle_fields=EPF)

# density slice plot
    p = yt.SlicePlot(ds,'z', 'H_nuclei_density',center="max",width=(600, 'pc'))
    p.annotate_timestamp(corner='upper_left', time_format='t = {time:.3f} {units}', time_unit= 'Myr', redshift=True, draw_inset_box=True)
    p.annotate_scale(corner='lower_right', draw_inset_box= True)
    p.annotate_particles((100, 'Mpc'), ptype='star', p_size = 15.0,marker ='.', col='r')
    #p.annotate_quiver('velocity_x', 'velocity_y', factor=32, plot_args={"color": "black"})
    
    #p.annotate_velocity(factor = 32 )
    #p.annotate_title('H Nuclei Density with Velocity and Pop II Stars')

    p.save('/homes/fgarcia4/test')
    
    #display.clear_output(wait=True)
    time.sleep(1)
    step += 1
