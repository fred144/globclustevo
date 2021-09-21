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

from clump_filters import clump_filters
#----------------------------------------------------------------------------
#data directory/info file
datadir=os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine') 
#----------------------------------------------------------------------------

#some colors https://www.rapidtables.com/web/color/RGB_Color.html
maroon = np.array([128,0,0])/255.; maroon = maroon.reshape(1,-1)
tomato = np.array([255,99,71])/255.; tomato = tomato.reshape(1,-1)
orange_red = np.array([255,69,0])/255.; orange_red = orange_red.reshape(1,-1)
dark_orange = np.array([255,140,0])/255.; dark_orange = dark_orange.reshape(1,-1)
dark_golden_rod =  np.array([184,134,11])/255.; dark_golden_rod = dark_golden_rod.reshape(1,-1)
olive =  np.array([128,128,0])/255.; olive = olive.reshape(1,-1)
lime  =  np.array([0,255,0])/255.; lime  = lime .reshape(1,-1)     
teal =  np.array([0,128,128])/255.; teal  = teal .reshape(1,-1)              
indigo = np.array([75,0,130])/255.; indigo  = indigo .reshape(1,-1)  
thistle = np.array([216,191,216])/255.; thistle  = thistle .reshape(1,-1)  
plum = np.array([221,160,221])/255.; plum  = plum .reshape(1,-1)  
violet = np.array([238,130,238])/255.; violet  = violet .reshape(1,-1)  
deep_pink = np.array([255,20,147])/255.; deep_pink  = deep_pink .reshape(1,-1) 
light_pink = np.array([255,182,193])/255.; light_pink   = light_pink  .reshape(1,-1) 
chocolate = np.array([210,105,30])/255.; chocolate    = chocolate   .reshape(1,-1) 
beige = np.array([245,245,220])/255.; beige= beige.reshape(1,-1) 
wheat = np.array([245,222,179])/255.; wheat= wheat.reshape(1,-1) 
saddle_brown = np.array([139,69,19])/255.; saddle_brown= saddle_brown.reshape(1,-1)
honeydew = np.array([240,255,240])/255.; honeydew = honeydew .reshape(1,-1)
ivory = np.array([255,255,240])/255.; ivory = ivory .reshape(1,-1)
azure = np.array([240,255,255])/255.; azure = azure .reshape(1,-1)
mintcream = np.array([245,255,250])/255.; mintcream = mintcream .reshape(1,-1)

startSlice = 120
endSlice = 131
for outputNum in range (startSlice, endSlice + 1):
    
    infofile = os.path.abspath (datadir + "/output_%05d/info_%05d.txt" % (int(outputNum), int(outputNum)))
    print ("#Reading in info file: %s" %infofile)    

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
    clump_filters(ds)
    #plot                                                                                                                                                                                                     
    width = (410,'pc') #plot width                                                                                                                                                                                
    
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
        #annotate the star clumps
        p.annotate_particles(width=width, ptype="clump1", p_size=20.0, marker='.',col='b') 
        p.annotate_particles(width=width, ptype="clump2", p_size=20.0, marker='.',col='g') 
        p.annotate_particles(width=width, ptype="clump3", p_size=20.0, marker='.',col='r') 
        p.annotate_particles(width=width, ptype="clump4", p_size=20.0, marker='.',col='c') 
        p.annotate_particles(width=width, ptype="clump5", p_size=20.0, marker='.',col='m') 
        p.annotate_particles(width=width, ptype="clump6", p_size=20.0, marker='.',col='y') 
        p.annotate_particles(width=width, ptype="clump7", p_size=20.0, marker='.',col='k') 
        p.annotate_particles(width=width, ptype="clump8", p_size=20.0, marker='.',col=maroon) 
        p.annotate_particles(width=width, ptype="clump9", p_size=20.0, marker='.',col=tomato) 
        p.annotate_particles(width=width, ptype="clump10", p_size=20.0, marker='.',col=orange_red) 
        p.annotate_particles(width=width, ptype="clump11", p_size=20.0, marker='.',col=dark_orange) 
        p.annotate_particles(width=width, ptype="clump12", p_size=20.0, marker='.',col=dark_golden_rod) 
        p.annotate_particles(width=width, ptype="clump13", p_size=20.0, marker='.',col=olive) 
        p.annotate_particles(width=width, ptype='clump14', p_size=20.0, marker='.',col=lime) 
        p.annotate_particles(width=width, ptype='clump15', p_size=20.0, marker='.',col=teal) 
        p.annotate_particles(width=width, ptype='clump16', p_size=20.0, marker='.',col=indigo) 
        p.annotate_particles(width=width, ptype='clump17', p_size=20.0, marker='.',col=thistle) 
        p.annotate_particles(width=width, ptype='clump18', p_size=20.0, marker='.',col=plum) 
        p.annotate_particles(width=width, ptype='clump19', p_size=20.0, marker='.',col=violet) 
        p.annotate_particles(width=width, ptype='clump20', p_size=20.0, marker='.',col=deep_pink) 
        p.annotate_particles(width=width, ptype='clump21', p_size=20.0, marker='.',col=light_pink) 
        p.annotate_particles(width=width, ptype='clump22', p_size=20.0, marker='.',col=chocolate) 
        p.annotate_particles(width=width, ptype='clump23', p_size=20.0, marker='.',col=beige) 
        p.annotate_particles(width=width, ptype='clump24', p_size=20.0, marker='.',col=wheat) 
        p.annotate_particles(width=width, ptype='clump25', p_size=20.0, marker='.',col=saddle_brown) 
        p.annotate_particles(width=width, ptype='clump26', p_size=20.0, marker='.',col=honeydew) 
        p.annotate_particles(width=width, ptype='clump27', p_size=20.0, marker='.',col=ivory) 
        p.annotate_particles(width=width, ptype='clump28', p_size=20.0, marker='.',col=azure) 
        p.annotate_particles(width=width, ptype='clump29', p_size=20.0, marker='.',col=mintcream) 
        
        if pos_PSCs.size > 0:
            #center = pos_PSCs[0]
            #p = yt.SlicePlot(ds, 'z', "density", width = width, center = center)

            p.annotate_particles(width=width,ptype='PSC', p_size=100.0,marker='x',col='k') #passive stellar clusters (test particles)
            p.annotate_particles(width=width,ptype='SFC', p_size=100.0,marker='x',col='b') #star forming clouds (test particles)
        
        if pos_SFCs.size > 0: 
            #center = pos_SFCs[0]
            #p = yt.SlicePlot(ds, 'z', "density", width = width, center = center)

            p.annotate_particles(width=width,ptype='PSC', p_size=100.0,marker='x',col='k') #passive stellar clusters (test particles)
            p.annotate_particles(width=width,ptype='SFC', p_size=100.0,marker='x',col='b') #star forming clouds (test particles)      
        
        
        hc.create()
        hc_ad = hc.halos_ds.all_data()
        #p.annotate_halos(hc, width=width)
        p.annotate_text((0.175, 0.2), str( len(hc_ad['particle_mass'])) + " Halos", coord_system="figure")
        
    

           
    p.annotate_timestamp(corner='lower_left', time_format='t = {time:.3f} {units}', time_unit= 'Myr', redshift=True, draw_inset_box=True)
    p.annotate_scale(corner='lower_right', draw_inset_box= True)
    p.set_cmap("density", "gnuplot2")
    p.annotate_title(' Z Density and Star Clusters | '  + str(outputNum) + ' of ' + str(endSlice) + 
                     ' | Dots = Pop II, Blue = SFC, Black = PSC ')
    p.set_buff_size(5000)
    #---------------------------------------------------------------------------------------------------------
    p.save('/homes/fgarcia4/analysis/cluster_evolution_fs07/sequences/refine_clump_tracked/', mpl_kwargs=dict(dpi=200))
    print ('saved frame #' + str(outputNum) + ' to: /homes/fgarcia4/analysis/cluster_evolution_fs07/sequences/refine_clump_tracked/')
    #---------------------------------------------------------------------------------------------------------
    
    
    
    
