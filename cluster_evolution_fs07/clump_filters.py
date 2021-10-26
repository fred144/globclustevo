# import libraries not sure yet which are vital
import yt
import warnings
#import sys
#import os
#import shutil
#import time
#import glob
#import argparse
import numpy as np
#import pylab as pl
#import matplotlib.pyplot as plt
#from IPython import display
#from mpl_toolkits.axes_grid1 import AxesGrid
from yt.funcs import mylog #reduces some of the outputs 

mylog.setLevel(40) #reduces some of the outputs when reading in yt data
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

#[ 0.         18.52773269 35.60895952 68.6774688  68.96389576 69.09889334
# 69.34419314 70.17392563 70.19719182 70.2027757  70.30222356 70.41787201
# 70.79423501]
def clump_filters(ds): 
    
    def clump1(pfilter, data):
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        hubble = ds.hubble_constant     #hubble = H0/100
        cgs_yr = 3.1556926e7          # 1yr (in s)
        cgs_pc = 3.08568e18           # pc (in cm)
        H0=  hubble*100 #hubble parameter (km/s/Mpc)
        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
        star_age_myr = code_ages*H0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
        wanted_age = 0 #0 corresponds to the oldest star clump
        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
        return filter
    yt.add_particle_filter("clump1",function=clump1,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump1") 
    
    def clump2(pfilter, data):
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        hubble = ds.hubble_constant     #hubble = H0/100
        cgs_yr = 3.1556926e7          # 1yr (in s)
        cgs_pc = 3.08568e18           # pc (in cm)
        H0=  hubble*100 #hubble parameter (km/s/Mpc)
        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
        star_age_myr = code_ages*H0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
        wanted_age = 18.52773269 #0 corresponds to the oldest star clump
        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
        return filter
    yt.add_particle_filter("clump2",function=clump2,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump2") 
    
    
    def clump3(pfilter, data):
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        hubble = ds.hubble_constant     #hubble = H0/100
        cgs_yr = 3.1556926e7          # 1yr (in s)
        cgs_pc = 3.08568e18           # pc (in cm)
        H0=  hubble*100 #hubble parameter (km/s/Mpc)
        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
        star_age_myr = code_ages*H0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
        wanted_age = 35.60895952 #0 corresponds to the oldest star clump
        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
        return filter
    yt.add_particle_filter("clump3",function=clump3,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump3") 

    def clump4(pfilter, data):
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        hubble = ds.hubble_constant     #hubble = H0/100
        cgs_yr = 3.1556926e7          # 1yr (in s)
        cgs_pc = 3.08568e18           # pc (in cm)
        H0=  hubble*100 #hubble parameter (km/s/Mpc)
        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
        star_age_myr = code_ages*H0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
        wanted_age = 68.6774688 #0 corresponds to the oldest star clump
        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
        return filter
    yt.add_particle_filter("clump4",function=clump4,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump4") 

    def clump5(pfilter, data):
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        hubble = ds.hubble_constant     #hubble = H0/100
        cgs_yr = 3.1556926e7          # 1yr (in s)
        cgs_pc = 3.08568e18           # pc (in cm)
        H0=  hubble*100 #hubble parameter (km/s/Mpc)
        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
        star_age_myr = code_ages*H0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
        wanted_age = 68.96389576  #0 corresponds to the oldest star clump
        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
        return filter
    yt.add_particle_filter("clump5",function=clump5,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump5") 

    def clump6(pfilter, data):
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        hubble = ds.hubble_constant     #hubble = H0/100
        cgs_yr = 3.1556926e7          # 1yr (in s)
        cgs_pc = 3.08568e18           # pc (in cm)
        H0=  hubble*100 #hubble parameter (km/s/Mpc)
        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
        star_age_myr = code_ages*H0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
        wanted_age = 69.09889334 #0 corresponds to the oldest star clump
        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
        return filter
    yt.add_particle_filter("clump6",function=clump6,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump6") 

    def clump7(pfilter, data):
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        hubble = ds.hubble_constant     #hubble = H0/100
        cgs_yr = 3.1556926e7          # 1yr (in s)
        cgs_pc = 3.08568e18           # pc (in cm)
        H0=  hubble*100 #hubble parameter (km/s/Mpc)
        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
        star_age_myr = code_ages*H0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
        wanted_age = 69.34419314 #0 corresponds to the oldest star clump
        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
        return filter
    yt.add_particle_filter("clump7",function=clump7,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump7") 

    def clump8(pfilter, data):
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        hubble = ds.hubble_constant     #hubble = H0/100
        cgs_yr = 3.1556926e7          # 1yr (in s)
        cgs_pc = 3.08568e18           # pc (in cm)
        H0=  hubble*100 #hubble parameter (km/s/Mpc)
        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
        star_age_myr = code_ages*H0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
        wanted_age = 70.17392563 #0 corresponds to the oldest star clump
        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
        return filter
    yt.add_particle_filter("clump8",function=clump8,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump8") 

    def clump9(pfilter, data):
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        hubble = ds.hubble_constant     #hubble = H0/100
        cgs_yr = 3.1556926e7          # 1yr (in s)
        cgs_pc = 3.08568e18           # pc (in cm)
        H0=  hubble*100 #hubble parameter (km/s/Mpc)
        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
        star_age_myr = code_ages*H0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
        wanted_age = 70.19719182 #0 corresponds to the oldest star clump
        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
        return filter
    yt.add_particle_filter("clump9",function=clump9,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump9") 

    def clump10(pfilter, data):
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        hubble = ds.hubble_constant     #hubble = H0/100
        cgs_yr = 3.1556926e7          # 1yr (in s)
        cgs_pc = 3.08568e18           # pc (in cm)
        H0=  hubble*100 #hubble parameter (km/s/Mpc)
        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
        star_age_myr = code_ages*H0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
        wanted_age = 70.2027757 #0 corresponds to the oldest star clump
        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
        return filter
    yt.add_particle_filter("clump10",function=clump10,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump10") 
    
    def clump11(pfilter, data):
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        hubble = ds.hubble_constant     #hubble = H0/100
        cgs_yr = 3.1556926e7          # 1yr (in s)
        cgs_pc = 3.08568e18           # pc (in cm)
        H0=  hubble*100 #hubble parameter (km/s/Mpc)
        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
        star_age_myr = code_ages*H0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
        wanted_age = 70.30222356 #0 corresponds to the oldest star clump
        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
        return filter
    yt.add_particle_filter("clump11",function=clump11,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump11") 
    
    def clump12(pfilter, data):
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        hubble = ds.hubble_constant     #hubble = H0/100
        cgs_yr = 3.1556926e7          # 1yr (in s)
        cgs_pc = 3.08568e18           # pc (in cm)
        H0=  hubble*100 #hubble parameter (km/s/Mpc)
        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
        star_age_myr = code_ages*H0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
        wanted_age = 70.41787201 #0 corresponds to the oldest star clump
        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
        return filter
    yt.add_particle_filter("clump12",function=clump12,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump12") 
    
    def clump13(pfilter, data):
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        hubble = ds.hubble_constant     #hubble = H0/100
        cgs_yr = 3.1556926e7          # 1yr (in s)
        cgs_pc = 3.08568e18           # pc (in cm)
        H0=  hubble*100 #hubble parameter (km/s/Mpc)
        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
        star_age_myr = code_ages*H0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
        wanted_age = 70.79423501 #0 corresponds to the oldest star clump
        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
        return filter
    yt.add_particle_filter("clump13",function=clump13,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump13") 
        
#    def clump14(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 74.07321298 #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump14",function=clump14,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump14") 
#     
#    def clump15(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 74.15449704 #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump15",function=clump15,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump15") 
#         
#    def clump16(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 74.71621663 #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump16",function=clump16,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump16") 
#   
#    def clump17(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 74.80178142 #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump17",function=clump17,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump17") 
#    
#     
#    def clump18(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 74.82401217 #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump18",function=clump18,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump18") 
#   
#     
#    def clump19(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 82.55954734 #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump19",function=clump19,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump19") 
#        
#    def clump20(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 84.18673332 #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump20",function=clump20,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump20") 
#     
#    def clump21(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 86.09189412  #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump21",function=clump21,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump21") 
#     
#    def clump22(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 88.65704616 #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump22",function=clump22,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump22") 
#        
#    def clump23(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 88.83611467 #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump23",function=clump23,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump23") 
#    
#    def clump24(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 88.97242775 #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump24",function=clump24,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump24") 
#        
#    def clump25(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 89.70418426 #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump25",function=clump25,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump25") 
#   
#         
#    def clump26(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 90.29802408 #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump26",function=clump26,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump26") 
#    
#    def clump27(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 90.60260442 #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump27",function=clump27,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump27") 
#   
#         
#    def clump28(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 90.90566563 #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump28",function=clump28,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump28") 
#       
#    def clump29(pfilter, data):
#        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#        hubble = ds.hubble_constant     #hubble = H0/100
#        cgs_yr = 3.1556926e7          # 1yr (in s)
#        cgs_pc = 3.08568e18           # pc (in cm)
#        H0=  hubble*100 #hubble parameter (km/s/Mpc)
#        H0_invsec = H0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
#        H0inv_yr = 1/H0_invsec/cgs_yr   #1/H0 [yr]
#        star_age_myr = code_ages*H0inv_yr/1e6
#        relative_ages = star_age_myr - star_age_myr.min()
#        wanted_age = 90.94992565 #0 corresponds to the oldest star clump
#        filter = np.logical_and(np.isclose(relative_ages, wanted_age), relative_ages >= 0)
#        return filter
#    yt.add_particle_filter("clump29",function=clump29,filtered_type="star",requires=["particle_birth_epoch"])
#    ds.add_particle_filter("clump29") 
     
    
if __name__ == '__main__': 
    clump_filters()