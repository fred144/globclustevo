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
proj_width = 400 # pc

mpl.rc('font', family='serif')
plt.style.use('default')

directory = r"./pop_2_data/"
sfc_dir = r"./sfc_data/"
psc_dir = r"./psc_data/" 

files = sorted(os.listdir(directory))   [-2:-1] #[310:315:5]
sfc_files = sorted(os.listdir(sfc_dir)) [-2:-1] #[310:315:5]
psc_files = sorted(os.listdir(psc_dir)) [-2:-1] #[310:315:5]
test_widht = 200 
for i,file_name in enumerate(files, start=0):
    
    print("# read in:", file_name)
    time_str = file_name[10:16].replace('_','.') #in myr
    time = float(time_str)
    snapshot_num = int(file_name[4:9])
    sfc_filepath = sfc_dir + sfc_files[i]
    psc_filepath = psc_dir + psc_files[i]
    
    pop_2_data = np.loadtxt(directory + file_name)
    sfc_data = np.loadtxt(sfc_filepath)
    psc_data = np.loadtxt(psc_filepath)
    
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
        bins=4000,
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
   
    
#%%
# from findpeaks import findpeaks
# # Initialize
# fp = findpeaks(method='topology')
# # Fit
# results = fp.fit(counts)
# # # Plot the pre-processing steps
# # fp.plot_preprocessing()
# # # Plot all
# # fp.plot()

from skimage.feature import peak_local_max
from scipy.ndimage.measurements import center_of_mass, label

# peaks returns (row_idx,col_idx)
peaks = peak_local_max(counts, threshold_abs=20) 

col_idx =  peaks[:,1]
row_idx = peaks[:,0]


#%%

x = x_ctr[col_idx] 
y = y_ctr[row_idx]

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

plt.scatter(x,y, color='green',s=1)
plt.xlim(-test_widht, test_widht)
plt.ylim(-test_widht, test_widht)



