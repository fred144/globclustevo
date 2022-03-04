import numpy as np
import os
#import yt
#from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.transform import Rotation as R
from lum_funcs import star_luminosity_plot, look_up_table
from matplotlib.colors import LogNorm
import matplotlib.cm as cm



mpl.rc('font', family='serif')
plt.style.use('default')

directory = r"./pop_2_data/"
sfc_dir = r"./sfc_data/"
psc_dir = r"./psc_data/" 

files = sorted(os.listdir(directory))   [-111:-110] #[310:315:5]
# sfc_files = sorted(os.listdir(sfc_dir)) [-111:-110] #[310:315:5]
# psc_files = sorted(os.listdir(psc_dir)) [-111:-110] #[310:315:5]

test_widht = 200 
test_bins =  6000


for i,file_name in enumerate(files, start=0):
    
    print("# read in:", file_name)
    time_str = file_name[10:16].replace('_','.') #in myr
    time = float(time_str)
    snapshot_num = int(file_name[4:9])
    # sfc_filepath = sfc_dir + sfc_files[i]
    # psc_filepath = psc_dir + psc_files[i]
    
    pop_2_data = np.loadtxt(directory + file_name)
    # sfc_data = np.loadtxt(sfc_filepath)
    # psc_data = np.loadtxt(psc_filepath)
    
    x_star = pop_2_data[:,2]
    y_star = pop_2_data[:,3]
    
    # plt.figure(figsize=(8,8), dpi=200)
    # plt.scatter(x_star, y_star, s=.2, alpha=.2)
    # plt.scatter(sfc_data[:,0], sfc_data[:,1], s=5, c='red')
    # plt.scatter(psc_data[:,0], psc_data[:,1], s=5, c='red')
    # plt.xlim(-100,100)
    # plt.ylim(-100,100)
    
    counts, xedges, yedges = np.histogram2d(
        x_star,
        y_star,
        bins=test_bins,
        normed=False,
        range= [[-test_widht,test_widht], [-test_widht,test_widht]]
    )
    counts = counts.T 
    x_ctr = 0.5*(xedges[1:] + xedges[:-1])
    y_ctr = 0.5*(yedges[1:] + yedges[:-1])
    
    plt.figure(figsize=(8,8))
    plt.imshow(
               counts,
               cmap='inferno',
               #interpolation='gaussian',
               origin='lower',
               extent=[-test_widht,
                       test_widht,
                       -test_widht,
                       test_widht],
               #norm=LogNorm(np.min(counts),np.max(counts))
               )
   
star_positions = pop_2_data[:,2:5]
masses = pop_2_data[:,5]
#%%
from pytreegrav import Accel, Potential
phi = Potential(pos=star_positions, m=masses, method='bruteforce')
#%%
grav, xedges, yedges = np.histogram2d(
    x_star,
    y_star,
    bins=10000,
    weights=np.abs(phi),
    normed=False,
    range= [[-test_widht,test_widht], [-test_widht,test_widht]]
)
x_ctr = 0.5*(xedges[1:] + xedges[:-1])
y_ctr = 0.5*(yedges[1:] + yedges[:-1])
plt.hist(phi )
#%%
grav = grav.T
plt.imshow(
            grav,
            cmap='inferno',
            #interpolation='gaussian',
            origin='lower',
            extent=[-test_widht,
                    test_widht,
                    -test_widht,
                    test_widht],
            #norm=LogNorm(np.min(counts),np.max(counts))
            )
#ind = np.argpartition(np.abs(phi), -4)[-4:]
#top_phi = star_positions[ind]
# x_peak = top_phi[:,0]
# y_peak = top_phi[:,1]
#%%
from skimage.feature import peak_local_max
from scipy.ndimage import center_of_mass, label 


# from findpeaks import findpeaks
# # Initialize
# fp = findpeaks(method='topology')
# # Fit
# results = fp.fit(counts)
# # # Plot the pre-processing steps
# # fp.plot_preprocessing()
# # # Plot all
# # fp.plot()



# peaks returns (row_idx,col_idx)
peaks = peak_local_max(grav, num_peaks=90, min_distance=55, threshold_rel=.15) 
col_idx =  peaks[:,1]
row_idx = peaks[:,0] 


x = x_ctr[col_idx] 
y = y_ctr[row_idx]
#%%



plt.figure(figsize=(8,8))
plt.imshow(
           counts,
           cmap='inferno',
           #interpolation='gaussian',
           origin='lower',
           extent=[-test_widht,
                   test_widht,
                   -test_widht,
                   test_widht],
           #norm=LogNorm(np.min(counts),np.max(counts))
           )
#%%
test_widht = 100
plt.figure(figsize=(8,8),dpi=400)
plt.scatter(x_star,y_star,s=.5,alpha=.05,c='red')

plt.scatter(x,y, color='lime',s=.8)
plt.xlim(-test_widht, test_widht)
plt.ylim(-test_widht, test_widht)



