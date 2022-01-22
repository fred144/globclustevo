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



step = 113
while step <= 115:
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

    #----------------------------------------------------------------------------    

    #p = yt.SlicePlot(ds,'z', "Metallicity" , center=("max", 'Metallicity'), width=(500, 'pc') )
    #p.annotate_particles((100, 'Mpc'),ptype='star', p_size=20.0,marker='.',col='r')
    #p.annotate_contour('Metallicity', plot_args={"colors": "k","linewidths": 1})
    #p.annotate_timestamp(corner='upper_left', time_format='t = {time:.3f} {units}', time_unit= 'Myr', redshift=True, draw_inset_box=True)
    #p.annotate_scale(corner='lower_right', draw_inset_box= True)
    #p.annotate_title('Metallicity Slice Plot '+ str(step) + " of 125")



    #----------------------------------------------------------------------------
#Line plots
#    import yt.units as u
#    from matplotlib.pyplot import figure
#
#    figure(figsize=(6, 6), dpi=100)
#    my_sphere = yt.create_profile(ds, 'radius', ('DM','particle_mass'),
                         #units = {'radius': 'kpc'},
                         #extrema = {'radius': ((0.1, 'kpc'), (1000.0, 'kpc'))})
#    radius = my_sphere.x
#    profile = yt.create_profile(DM, ["radius"],
#                            "particle_mass", weight_field=None,
#                             n_bins=(128, 128))

#    
#    my_sphere = ds.sphere('c', (1, "Mpc"))
#    
#    
#    DM = my_sphere['DM','particle_mass'].in_units('Msun') 
#    
#    
#    pop2 = my_sphere['star','particle_mass'].in_units('Msun') 
#    
#    prof_gas = yt.create_profile(my_sphere, 'radius', ('gas','cell_mass'), units = {'radius': 'kpc'}, n_bins = 128, weight_field='cell_mass')
#    gas = prof_gas['gas','cell_mass'].in_units('Msun') 
#    
#
#    
#    radius_gas = prof_gas.x
#
    #plotting        
#    plt.loglog(radius_gas, gas, label = 'Gas')
    #plt.loglog(radius_gas, DM)
#    plt.xlabel('r (pc)')
#    plt.ylabel('Mass (Msun)')
#    plt.axis([1e-1, 1e4, 1e1, 1e10])
#    plt.title('Mass Profile of a Sphere Centered at Center| z = ' + str (round(ds.current_redshift, 5)))
#    plt.legend()
#    
    
#----------------------------------------------------------------------------    
##nT phase plot

#    varX_yt = 'H_nuclei_density'
#    varY_yt = 'temperature'

#    varX_min = 1e-6
#    varX_max = 1e4
#    varY_min = 1e-1
#    varY_max = 1e9

#    ad = ds.all_data()
#    units = dict(cell_mass='Msun')
#    extrema = {varX_yt:(varX_min,varX_max), varY_yt:(varY_min,varY_max)}
#    profile = yt.create_profile(ad, [varX_yt,varY_yt],
#                            n_bins=[164, 164], fields=['cell_mass'],
#                            weight_field=None, units=units, extrema=extrema)

#    p = yt.PhasePlot.from_profile(profile)


#---------------------------------------------------------------------
## density slice plot


    from yt.analysis_modules.halo_analysis.api import HaloCatalog

 
    hc = HaloCatalog(data_ds=ds, finder_method='fof',
                    finder_kwargs={"ptype": "star",
                                "padding": 0.2,
                               "link": 0.0002,
                                "dm_only":False})
    hc.create()
    hc_ad = hc.halos_ds.all_data()
    print ("# of halos:",  len(hc_ad['particle_mass']))    
    print ("halo masses:", hc_ad['particle_mass'].in_units('Msun'))
    print ("halo radii:", hc_ad['virial_radius'].in_units('pc'))
    print ( hc_ad['particle_position'][0])
   
#    #plot
    width = (600,'pc')
    
    
    center = hc_ad['particle_position'][0].in_units('pc') 
   
    p=yt.SlicePlot(ds, 'z', 'density', width = width, center = center)
    p.set_unit ('density', 'g/cm**3') 
    p.annotate_particles(width=width,ptype='star', p_size=20.0,marker='.',col='r')
    p.annotate_halos(hc,width=width)
     
    p.annotate_title('Nuclei Density and Star Cluster') 
    p.annotate_timestamp(corner='upper_right', time_format='t = {time:.3f} {units}', time_unit= 'Myr', redshift=True, draw_inset_box=True)
    p.annotate_scale(corner='lower_right', draw_inset_box= True)






#    plt.savefig('/homes/fgarcia4/analysis/cluster_evolution_fs07/z_density_cluster_highlighted')
    p.save('/homes/fgarcia4/analysis/cluster_evolution_fs07/z_density_cluster/' + str(step).zfill(3) )
    
    
    time.sleep(.001)
    step += 1
