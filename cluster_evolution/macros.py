import yt
import warnings
import numpy as np
from yt.funcs import mylog 
#reduces some of the outputs when reading in yt data
mylog.setLevel(40) 
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

def code_age_to_yr (all_star_ages, hubble_const, unique = True): 
    r"""
    Returns an array with unique birth epochs in Myr given  
    raw_birth_epochs = ad['star', 'particle_birth_epoch'] 
    AND 
    hubble = ds.hubble_constant  
    Youngest is 0 Myr, all others are relative to the youngest.
    """
    cgs_yr = 3.1556926e7              # 1yr (in s)
    cgs_pc = 3.08568e18               # pc (in cm)
    h_0 =  hubble_const*100           # hubble parameter (km/s/Mpc)
    h_0_invsec = h_0*1e5/(1e6*cgs_pc) # hubble constant h [km/s Mpc-1]->[1/sec]
    h_0inv_yr = 1/h_0_invsec/cgs_yr   # 1/h_0 [yr]
    
    if unique == True:
        #process to unique birth epochs only
        be_star_processed = np.array(sorted( list( set(all_star_ages.to_ndarray()) ) ) ) 
        star_age_myr = be_star_processed * h_0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
    else: 
        all_stars  =  all_star_ages
        star_age_myr = all_stars * h_0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
    return relative_ages

def succ_distance (current, previous):
    dist = np.linalg.norm(np.array(current) - np.array(previous)) 
    return dist 

def max_density_center_stable(
        ds, ad, loop_num, ctr_shift_thresh, slice_axis, width, prev_max_den
        ):
    r"""
    Center at the most dense point, but keep it relatively stable. 
    
    """
    # centering using max density 
    max_den = ad.argmax(('gas', 'density'))
    max_density_coord = yt.YTArray(max_den).in_units('code_length') 
    max_density_coord = np.array(max_density_coord)
    #keep center at density max
    if loop_num == 0:
        prev_max_den = max_density_coord
        distance = succ_distance(max_density_coord, prev_max_den)
        print('\n> distance b/w current and previously used max density:', distance)
        
    distance = succ_distance(max_density_coord, prev_max_den)  
    
    if distance < ctr_shift_thresh: 
        p = yt.ProjectionPlot(
            ds, slice_axis, "density", width = width, center = max_density_coord
            )
        x_pos = np.array(ad['star', 'particle_position_x']) - max_density_coord[0]
        y_pos = np.array(ad['star', 'particle_position_y']) - max_density_coord[1]
        z_pos = np.array(ad['star', 'particle_position_z']) - max_density_coord[2]
        # if the plot center migrates, annotate previous center
        # p.annotate_marker(
        #     max_density_coords[-1],
        #     marker="x",
        #     coord_system="data",
        #     plot_args={"color": "lime", "s": 30},)
        # appened new center to list
        print('> new plot centered at {}'. format(max_density_coord)) 
        return p, max_density_coord, x_pos, y_pos, z_pos
               
    else: 
        #center = prev_max_den
        p = yt.ProjectionPlot(
            ds, slice_axis, "density", width = width, center = prev_max_den
            )
        x_pos = np.array(ad['star', 'particle_position_x']) - prev_max_den[0]
        y_pos = np.array(ad['star', 'particle_position_y']) - prev_max_den[1]
        z_pos = np.array(ad['star', 'particle_position_z']) - prev_max_den[2]
        print('> using old center at {}'. format(prev_max_den)) 
        return p, max_density_coord, x_pos, y_pos, z_pos

# #generate colors test
# colors = np.zeros([50,3])
# for i in range (0,np.shape(colors)[0]):
#     #color = np.random.rand(3,1).T 
    
#     color = np.random.rand(3,1).T
#     colors[i,:]= color
# colors = list(colors)
