import numpy as np
import matplotlib.pyplot as plt 


mtot_1 = np.loadtxt(r'./sequences/new_refine/data/m_tot-00001-00100.txt')
mtot_2 = np.loadtxt(r'./sequences/new_refine/data/m_tot-00101-00200.txt') 
mtot_3 = np.loadtxt(r'./sequences/new_refine/data/m_tot-00201-00660.txt')
mtot = np.vstack((mtot_1,mtot_2,mtot_3))  
#[redshift, current_time, dm, pop2, pop3, sn, dead, BH, sfc, psc]
plt.figure(figsize = (6,6), dpi=200 ) 
plt.plot(mtot[:,1], mtot[:,3]) 
plt.xlim(100,600)
#plt.scatter(mtot[:,1], mtot[:,5]) 
#%%
ntot_1 = np.loadtxt(r'./sequences/new_refine/data/n_tot-00001-00100.txt')
ntot_2 = np.loadtxt(r'./sequences/new_refine/data/n_tot-00101-00200.txt') 
ntot_3 = np.loadtxt(r'./sequences/new_refine/data/n_tot-00201-00660.txt')
ntot = np.vstack((ntot_1,ntot_2,ntot_3))  
plt.figure(figsize = (6,6), dpi=200 ) 
plt.plot(ntot[:,1], ntot[:,3])  