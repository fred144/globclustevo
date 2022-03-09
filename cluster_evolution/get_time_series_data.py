import sys
sys.path.insert(
    1, "/homes/fgarcia4/py-virtual-envs/cosmology-clean/lib/python3.7/site-packages"
    )
import warnings
import os
import yt
import numpy as np
from yt.funcs import mylog


mylog.setLevel(40)
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

#---------------------------------local test-----------------------------------
# datadir = os.path.realpath('/home/fabg/cosm_test_data/refine')
# parent_folder = 'C:/Users/144/Desktop/AstroSimulationResearch/cluster_evolution'
# parent_folder = '.'
# sequence_folder = 'test_frames'

#---------------------------------DT2 Paths------------------------------------
# lustre data path
datadir = os.path.expanduser(
    '/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine'
    )
# save path
sequence_folder = 'data'
parent_folder = './time_series_data/fs07_ms10'
newpath = parent_folder + '/' + sequence_folder
if not os.path.exists(newpath):
    os.makedirs(newpath)


# datadir = os.path.expanduser(
#     '/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs035_ms10'
#     )
# # save path
# sequence_folder = 'data'
# parent_folder = '/homes/fgarcia4/analysis/cluster_evolution/sequences/fs035_ms10'
# newpath = parent_folder + '/' + sequence_folder
# if not os.path.exists(newpath):
#     os.makedirs(newpath)




ctr_at_max_den = False
radius = (500, 'pc')


start_step = 695
end_step = 710 

total_counts = []
total_masses = []

#---------------------------------MAIN LOOP-----------------------------------
for loop_num, output_num in enumerate(range(start_step, end_step + 1)) :
    print ("#________________________________________________________________")
    infofile = os.path.abspath(
        datadir + f"/output_{output_num:05}/info_{output_num:05}.txt"
        )
    print ("# reading in", infofile )

    # read fields explicitly, not recognized by YT from this ver of RAMSES
    cell_fields = [
        'Density',
        'x-velocity',
        'y-velocity',
        'z-velocity',
        'Pressure',
        'Metallicity',
        'dark_matter_density',
        'xHI',
        'xHII',
        'xHeII',
        'xHeIII'
        ]
    epf = [
        ('particle_family', 'b'),
        ('particle_tag', 'b'),
        ('particle_birth_epoch', 'd'),
        ('particle_metallicity', 'd')
        ]

    # read in RAMSES data set
    ds = yt.load(
        infofile, fields=cell_fields, extra_particle_fields=epf
        )
    redshft = ds.current_redshift
    current_time = float(ds.current_time.in_units('Myr'))
    
    if ctr_at_max_den is True:
        ad = ds.all_data()
        max_den = ad.argmax(('gas', 'density'))
        max_density_coord = yt.YTArray(max_den).in_units('code_length') 
        region = ds.sphere(max_density_coord, radius)
        ad = region
    else:
        ad = ds.all_data()
        

    
    # get indivudual masses
    dark_matter = ad['DM','particle_mass'].in_units('Msun')       
    pop_ii = ad['star','particle_mass'].in_units('Msun')    
    # living  pop three stars    
    pop_iii = ad['POPIII','particle_mass'].in_units('Msun')     
    # Pop III stars taking place SNe
    sn = ad['supernova','particle_mass'].in_units('Msun')  
    # Pop III stars after SNe
    dead = ad['dead','particle_mass'].in_units('Msun') 
    # Pop III remnant BHs           
    black_hole = ad['BH','particle_mass'].in_units('Msun') 
    # Star-Forming Clouds test particles, should be 0               
    star_forming = ad['SFC','particle_mass'].in_units('Msun')      
    # Passive Stellar Clusters, should be 0               
    passive = ad['PSC','particle_mass'].in_units('Msun')               
    
    # gas 
    m_gas = ad['gas','mass'].in_units('Msun').sum()         
    
    # number of particles
    n_dark_matter = len(dark_matter)
    n_pop_ii = len(pop_ii)
    n_pop_iii = len(pop_iii)
    n_sn = len(sn)
    n_dead = len(dead)
    n_black_hole = len(black_hole)
    n_star_forming = len(star_forming)
    n_passive = len(passive)    
    
    # total mass
    m_dark_matter = dark_matter.sum()
    m_pop_ii = pop_ii.sum()
    m_pop_iii = pop_iii.sum()
    m_sn = sn.sum()
    m_dead = dead.sum()
    m_black_hole = black_hole.sum() 
    m_star_forming = star_forming.sum()
    m_passive = passive.sum() 
    

    # save format
    # [redshift, current_time, dm, pop2, pop3, sn, dead, BH, sfc, psc]
    counts = np.array(
    [redshft,current_time,n_dark_matter,n_pop_ii,n_pop_iii,n_sn,n_dead,n_black_hole,n_star_forming,n_passive]
    )
    # [redshift, current_time, dm, pop2, pop3, sn, dead, BH, sfc, psc,gas mass]
    masses = np.array(
    [redshft,current_time,m_dark_matter,m_pop_ii,m_pop_iii,m_sn,m_dead,m_black_hole,m_star_forming,m_passive,m_gas]
    )
    
    total_counts.append(counts)
    total_masses.append(masses)
    

total_counts = np.array(total_counts)
total_masses = np.array(total_masses)

if ctr_at_max_den is True:
    n_save_path = str (
        "{}/{}/n_maxden-{}-{}.txt".format(
            parent_folder,
            sequence_folder,
            str(start_step).zfill(5),
            str(end_step).zfill(5),
            )
        ) 
    m_save_path = str (
        "{}/{}/m_maxden-{}-{}.txt".format(
            parent_folder,
            sequence_folder,
            str(start_step).zfill(5),
            str(end_step).zfill(5),
            )
        )  
else:
    n_save_path = str (
        "{}/{}/n_tot-{}-{}.txt".format(
            parent_folder,
            sequence_folder,
            str(start_step).zfill(5),
            str(end_step).zfill(5),
            )
        ) 
    m_save_path = str (
        "{}/{}/m_tot-{}-{}.txt".format(
            parent_folder,
            sequence_folder,
            str(start_step).zfill(5),
            str(end_step).zfill(5),
            )
        )  

    
np.savetxt(fname=n_save_path, X=total_counts)
np.savetxt(fname=m_save_path, X=total_masses) 
#%% sfc psc testing
# for field in dir(ds.fields.BH):
#     print(field)
