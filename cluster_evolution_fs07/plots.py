import numpy as np
import matplotlib.pyplot as plt 

time_series_data = np.loadtxt('timeseriesdata.txt')

plt.figure(figsize = (12,8))
plt.scatter(time_series_data[:,1], time_series_data[:,2])

plt.xlabel('Time (Myr)')
plt.ylabel('Total Pop 2 Mass (Msun)')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))