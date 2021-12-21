from cluster_evolution_fs07.macros import code_age_to_yr
import numpy as np
import os
import yt

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

datadir = os.path.expanduser(
    'G:/My Drive/Research/AstrophysicsSimulation/DesktopEnvironment/data_globular_cluster/refine/output_00250/info_00250.txt'
    ) 

# datadir = os.path.expanduser(
#     '/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine/output_00250/info_00250.txt'
#     ) 
ds = yt.load(datadir, fields=FIELDS, extra_particle_fields=EPF)

ad = ds.all_data()
current_hubble = ds.hubble_constant

current_time = float(ds.current_time.in_units('Myr')) 
redshft = ds.current_redshift
