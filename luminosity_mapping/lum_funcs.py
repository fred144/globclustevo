import numpy as np
import pandas as pd
#import yt
#from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
#import seaborn as sns
import matplotlib.cm as cm
import matplotlib.patches as patches
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm 

def star_luminosity_plot(
        proj_width,
        star_positions, 
        scaled_stellar_lums,
        time,
        snapshot_num, 
        pi_multiple,
        ):
    lums, xedges, yedges = np.histogram2d(
        star_positions[:,0], 
        star_positions[:,1], 
        bins=2000, 
        weights=scaled_stellar_lums, 
        normed=False, 
        range= [[-proj_width/2, proj_width/2], [-proj_width/2, proj_width/2]]
    )
    lums = lums.T
    
    fig = plt.figure(figsize=(14,12),dpi=200) 
    ax = fig.add_subplot(111, facecolor=cm.inferno(0))
    rectbin = plt.imshow(
               lums, 
               cmap='inferno',
               interpolation='gaussian', 
               origin='lower',
               extent=[-proj_width/2, 
                       proj_width/2,
                       -proj_width/2, 
                       proj_width/2],
               norm=LogNorm(vmin=3e+32, vmax=3e+36)
               )
    # annotate with time 
    ax.text(
        -proj_width*0.375, 
        -proj_width*0.45, 
        't = %.2f Myr'%(time), 
        size=12, 
        ha='center', 
        va='center', 
        color='white')
    # add color bar 
    # to the side
    #divider = make_axes_locatable(ax) 
    #cax = divider.append_axes('bottom', size='1%', pad=0.0)
    # cbar = fig.colorbar(
    #     rectbin, 
    #     pad = 0,
    #     aspect=60,
    #     )
    # to the bottom  
    fig.subplots_adjust(wspace=0, hspace=0, bottom=.1)
    cbar_ax = fig.add_axes([.178, .090, 0.67, 0.010]) 
    cbar = fig.colorbar(rectbin, 
                 cax=cbar_ax, 
                 orientation='horizontal', 
                 pad=0
                )
    cbar.set_label(
        label=r"$\lambda = 1500\;\AA$ Projected Monochromatic Luminosity" +
        r" $\left(erg\;s^{-1}\AA^{-1} \right)$", 
        size=12
        )
    # fig.suptitle(
    #     r"$\lambda = 1500\;\AA$ Projected Monochromatic Luminosity" +
    #     r" $\left(erg\;s^{-1}\AA^{-1} \right)$",
    #     y=.91,
    #     size=12)
    
    # add scale bar 
    rect = patches.Rectangle(   
            xy=(-proj_width*0.125, proj_width*0.45), 
            width=proj_width*0.25, 
            height=proj_width*0.005, 
            linewidth=0, 
            edgecolor='white', 
            facecolor='white')
    ax.add_patch(rect)
    ax.text(
        0, 
        proj_width*0.475, 
        '{}pc'.format(int(proj_width/4)), 
        size=12, 
        ha='center', 
        va='center', 
        color='white'
        )
    #ax.set_axis_off()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none')
    ax.add_artist(ax.patch)
    ax.patch.set_zorder(-1) 
    
  
def look_up_table(stellar_ages, table_link, column_idx:int, log=True):
    df = pd.read_csv(
        table_link, 
        delim_whitespace=True, 
        header=None
        )
    data = df.to_numpy().astype(float)
    look_up_times = data[:,0] # yr
    
    if log is True:
        look_up_lumi = 10**data[:,column_idx] 
    else: 
        look_up_lumi = data[:,column_idx] 
    
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
