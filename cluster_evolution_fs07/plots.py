import numpy as np
import matplotlib.pyplot as plt 

time_series_data = np.loadtxt('timeseriesdata.txt')
be = np.array(
    [ 0., 18.52773269, 35.60895952, 68.6774688,  68.96389576, 69.09889334,
     69.34419314, 70.17392563, 70.19719182, 70.2027757,  70.30222356, 70.41787201,
     70.79423501, 88.28128775, 89.38194823, 90.51360277, 95.28010249, 98.17317232]
    )
birth_epochs = be + 339.562

#mass return by SN 
#number of super novas = total mass over 150
    # efficiency is total stellar mass over  largest dm halo mass
plt.figure(figsize = (12,8))
plt.scatter(time_series_data[:,1], time_series_data[:,2])
for birth_epoch in birth_epochs:
    plt.axvline(x=birth_epoch, color = 'k', linestyle='--') 
plt.xlim(435,400)
plt.xlabel('Time (Myr)')
plt.ylabel('Total Pop 2 Mass (Msun)')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.legend()