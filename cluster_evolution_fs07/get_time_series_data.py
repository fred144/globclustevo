import os
import yt
import numpy as np
from macros import code_age_to_yr


# path = 'halo_catalogs/catalog/catalog.0.h5'
# data = h5py.File(path, 'r')
# particle_id = np.array(data.get('particle_identifier'))
# particle_mass = np.array(data.get('particle_mass'))

#datadir = os.path.expanduser(
#    'G:/My Drive/Research/AstrophysicsSimulation/DesktopEnvironment/data_globular_cluster/refine') 

#datadir = os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_rerun') 
datadir = os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine') 

# local save path 
#parent_folder = 'C:/Users/1.44/Desktop/AstroSimulationResearch/cluster_evolution_fs07'
#sequence_folder = 'test_frames'
#---------------------------------save path---------------------
##### cluster save path ######
clust_save_path = '/homes/fgarcia4/analysis/cluster_evolution_fs07/sequences/new_refine/data'

# make new folder
if not os.path.exists(clust_save_path ):
    os.makedirs(clust_save_path)


start_step = 250
end_step = 255 #345

mass_data = []
ages = []

#---------------------------------MAIN LOOP-----------------------------------
for loop_num, output_num in enumerate(range(start_step, end_step + 1)) :
    infofile = os.path.abspath (datadir + "/output_%05d/info_%05d.txt" % (output_num,output_num))
    #print ("#reading in info file: %s" %infofile)  
    
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
    ad = ds.all_data()
    current_hubble = ds.hubble_constant
    
    current_time = float(ds.current_time.in_units('Myr')) 
    redshft = ds.current_redshift
    total_pop2_mass = float(
        ad.quantities.total_quantity([('star', 'particle_mass')]).in_units("Msun")
        )
    total_dm_mass = float(
        ad.quantities.total_quantity([('DM', 'particle_mass')]).in_units("Msun") 
        )
    raw_birth_epochs = ad['star', 'particle_birth_epoch'] 
    unique_birth_epochs = np.array(
        code_age_to_yr(raw_birth_epochs, current_hubble))
    
    time_step_mass_data = np.array(
        [redshft, current_time, total_pop2_mass, total_dm_mass]
        )
    mass_data.append(time_step_mass_data)
    ages.append(unique_birth_epochs)
    print('> reading output:', output_num)

mass_data = np.array(mass_data)

name = clust_save_path + '/timeseries_mass_data_11_09.txt'
name_1 = clust_save_path + '/timeseries_age_data_11_09.txt'

np.savetxt(fname=name, X=mass_data)
np.savetxt(fname=name_1, X=ages)


    



    

    

  
