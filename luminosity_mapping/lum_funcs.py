import numpy as np
import pandas as pd

def look_up_table(stellar_ages, table_link, column:int, log=True):
    df = pd.read_csv(
        table_link, 
        delim_whitespace=True, 
        header=None
        )
    data = df.to_numpy().astype(float)
    look_up_times = data[:,0] # yr
    
    if log is True:
        look_up_lumi = 10**data[:,column] 
    else: 
        look_up_lumi = data[:,column] 
    
    residuals = np.abs(look_up_times - stellar_ages[:, np.newaxis]) 
    #print(np.shape(residuals))
    closest_match_idxs = np.argmin(residuals, axis=1) 
    #print(np.shape( closest_match_idxs), np.min(closest_match_idxs), np.max(closest_match_idxs))
    luminosities = look_up_lumi[closest_match_idxs]
    return luminosities

def spherical_mask(xpos, ypos, zpos, ctr_at, cluster_radius, lums):
    all_positions = np.vstack((xpos, ypos, zpos)).T 
    distances = np.sqrt(np.sum(np.square(all_positions - ctr_at), axis=1)) 
    mask = distances <= cluster_radius
    #print(distances.size)
    masked_positions = all_positions[mask]
    masked_lums = lums[mask]
    return masked_positions[:,0], masked_positions[:,1], masked_positions[:,2], masked_lums
