import numpy as np
import os
#import yt
#from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
import pandas as pd
#import seaborn as sns
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm 
from scipy.spatial.transform import Rotation as R 

def look_up_table(stellar_ages, look_up_times, look_up_lumi):
    residuals = np.abs(look_up_times - stellar_ages[:, np.newaxis]) 
    #print(np.shape(residuals))
    closest_match_idxs = np.argmin(residuals, axis=1) 
    #print(np.shape( closest_match_idxs), np.min(closest_match_idxs), np.max(closest_match_idxs))
    luminosities = look_up_lumi[closest_match_idxs]
    return luminosities

df = pd.read_csv('https://www.stsci.edu/science/starburst99/data/l1500_inst_e.dat', delim_whitespace=True, header=None)
data = df.to_numpy().astype(float)
lookuptime = data[:,0] # yr
# 150 nm (UV) monocrhomatric Luminosity (erg/s/Å)
lum_imf_2_35_m_up_100msun = 10**data[:,1] 
lum_imf_3_30_m_up_100msun = 10**data[:,2]
lum_imf_2_35_m_up_30msun = 10**data[:,3]  


mpl.rc('font', family='serif') 
plt.style.use('dark_background') 

directory = r"./pop_2_data/"
files = sorted(os.listdir(directory))   

#rotation_interval = np.arange(0,2,.005) 
rotation_interval = np.linspace(0,2,np.size(files))

for frame,(file_name,pi_multiple) in enumerate(zip(files,rotation_interval), start=113):

    print(file_name)
    time_str = file_name[10:16].replace('_','.') #in myr
    time = float(time_str)
    #print(file_name)

    pop_2_data = np.loadtxt('./pop_2_data/' + file_name) 
    birth_epochs = pop_2_data[:,0] *1e6 
    ages = pop_2_data[:,1] *1e6 
    # If the age <10^6 yr, set the age to 10^6 year.
    ages [ages < 1e6 ] = 1e6
    
    # x_pos = pop_2_data[:,2]
    # y_pos = pop_2_data[:,3] 
    # z_pos = pop_2_data[:,4]

    #for i,angle_multiple in enumerate(rotation_interval):
        
    star_positions = pop_2_data[:,2:5] 
    #counter clockwise along y axis
    r = R.from_rotvec(pi_multiple*np.pi * np.array([0, 1, 0]))
    rotation_matrix = r.as_matrix()
    rot_star_positions = np.dot(star_positions, rotation_matrix .T)
    
    stellar_lums = look_up_table(ages, lookuptime, lum_imf_2_35_m_up_100msun) 
    scaled_stellar_lums = stellar_lums*1e-5 
    
    lums, xedges, yedges = np.histogram2d(
        rot_star_positions[:,0], 
        rot_star_positions[:,1], 
        bins=2000, 
        weights=scaled_stellar_lums, 
        normed=False, 
        range= [[-200, 200], [-200, 200]]
    )
    lums = lums.T
    
    fig = plt.figure(figsize=(14,12),dpi=200) 
    ax = fig.add_subplot(111, facecolor=cm.inferno(0))
    rectbin = plt.imshow(
               lums, 
               cmap='inferno',
               interpolation='gaussian', 
               origin='lower',
               extent=[-200, 200,-200, 200],
               norm=LogNorm(vmin=3e+32, vmax=3e+36)
               )
    # annotate with time 
    ax.text(
        -150, 
        -180, 
        't = %.2f Myr'%(time), 
        size=12, 
        ha='center', 
        va='center', 
        color='white')
    
    # and color bar 
    #divider = make_axes_locatable(ax) 
    #cax = divider.append_axes('bottom', size='1%', pad=0.0)
    cbar = fig.colorbar(
        rectbin, 
        pad = 0,
        aspect=60,
        #orientation='horizontal',
        #cax=cax
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
    
    # and scale bar 
    rect = patches.Rectangle(   
            xy=(75, 180), 
            width=100, 
            height=2, 
            linewidth=0, 
            edgecolor='white', 
            facecolor='white')
    ax.add_patch(rect)
    ax.text(
        125, 190, '100 pc', size=12, ha='center', va='center', color='white'
        )
    # ax.set_axis_off()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none')
    ax.add_artist(ax.patch)
    ax.patch.set_zorder(-1) 
    #plt.show() 
    
    save_name = './test_sequence/rotation/rot_{}_{}_pi{}.png'.format(
        str(frame).zfill(3), 
        str(time).ljust(6, '0').replace('.','_'),
        str(np.round(pi_multiple,3)).ljust(5, '0').replace('.','_'))
    
    plt.savefig( 
        save_name, 
        dpi=200,
        bbox_inches='tight',
        pad_inches=0.05
        )
    
    print(save_name)   
    
    plt.close() 