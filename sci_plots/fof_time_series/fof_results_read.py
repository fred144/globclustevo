"""
Make a time series result a la profiler, but only use the fof results,
don't pass it throught the gc filter. Needs post processed halo finder
as well as a gc_profiler
"""

import sys

sys.path.append("../../")
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from modules.macros import filter_snapshots, common_filter_snapshots
import os


#%% get the gc results from halo finder only
halo_data_directory = r"../../halo_data/fs035_ms10/fof_best"
pop2_data_directory = r"../../particle_data/pop_2_data/fs035_ms10"
save_pth = "./fs035_fof_best_154_1316.txt"
strt = 154
end = 1316
step = 1
pop2 = filter_snapshots(pop2_data_directory, strt, end, step)
halos_ds = filter_snapshots(halo_data_directory, strt, end, step)
pop2_ds = common_filter_snapshots(pop2, halos_ds)
#%% fof bound and unbound

t = []
rdshift = []

fof_bound_mass = []
fof_field_mass = []

prof_bound_mass = []
prof_field_mass = []

fof_bound_lumi = []
fof_field_lumi = []

prof_bound_lumi = []
prof_field_lumi = []


for i, (ds, p2) in enumerate(zip(halos_ds, pop2_ds), start=strt):
    print(i)
    p_ii = np.loadtxt(p2)
    t_myr, z = p_ii[:2, 6]  # old pop 2 data, no luminosities, adjust idx accordingly
    # pure friends of friends algorithm
    bound = np.loadtxt(os.path.join(ds, "bound_stars.txt"))
    field = np.loadtxt(os.path.join(ds, "field_stars.txt"))
    # exclude if not fitted
    bound_fitted = np.loadtxt(os.path.join(ds, "fitted_bound_stars.txt"))
    field_fitted = np.loadtxt(os.path.join(ds, "fitted_field_stars.txt"))
    try:
        lum_bound = np.sum(bound[:, 2])
        m_bound = np.sum(bound[:, 6])

    except:
        lum_bound = 0
        m_bound = 0

    try:
        lum_field = np.sum(field[:, 2])
        m_field = np.sum(field[:, 6])
    except:
        lum_field = 0
        m_field = 0

    try:
        lum_bound_fitted = np.sum(bound_fitted[:, 2])
        m_bound_fitted = np.sum(bound_fitted[:, 6])
    except:
        lum_bound_fitted = 0
        m_bound_fitted = 0

    try:
        lum_field_fitted = np.sum(field_fitted[:, 2])
        m_field_fitted = np.sum(field_fitted[:, 6])
    except:
        lum_field_fitted = 0
        m_field_fitted = 0

    t.append(t_myr)
    rdshift.append(z)

    fof_bound_mass.append(m_bound)
    fof_field_mass.append(m_field)
    prof_bound_mass.append(m_bound_fitted)
    prof_field_mass.append(m_field_fitted)

    fof_bound_lumi.append(lum_bound)
    fof_field_lumi.append(lum_field)
    prof_bound_lumi.append(lum_bound_fitted)
    prof_field_lumi.append(lum_field_fitted)

# x, y, z = np.loadtxt(os.path.join(ds, "field_stars.txt"))
t = np.array(t)
rdshift = np.array(rdshift)

fof_bound_mass = np.array(fof_bound_mass)
fof_field_mass = np.array(fof_field_mass)

prof_bound_mass = np.array(prof_bound_mass)
prof_field_mass = np.array(prof_field_mass)

fof_bound_lumi = np.array(fof_bound_lumi)
fof_field_lumi = np.array(fof_field_lumi)

prof_bound_lumi = np.array(prof_bound_lumi)
prof_field_lumi = np.array(prof_field_lumi)
#%%
data = np.vstack(
    (
        t,
        rdshift,
        fof_bound_mass,
        fof_field_mass,
        fof_bound_lumi,
        fof_field_lumi,
        prof_bound_mass,
        prof_field_mass,
        prof_bound_lumi,
        prof_field_lumi,
    )
).T

h = (
    "\t\t\t time \t\t\t\t redshift \t\t\t"
    "\tfof_bound_mass \t\t fof_field_mass \t\t"
    "\tfof_bound_lumi \t\t fof_field_lumi \t\t"
    "\tprof_bound_mass \t\t profiler_field_mass \t\t"
    "\tprof_bound_lumi \t\t prof_field_lumi \t\t"
)

np.savetxt(save_pth, X=data, header=h)
#%% fof performance map scatter
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), dpi=300)
(x, y, z) = np.loadtxt(os.path.join(ds, "field_stars.txt"))[:, 3:6].T
ax.scatter(x, y, c="tab:blue", s=0.5, alpha=0.08)
(x, y, z) = np.loadtxt(os.path.join(ds, "bound_stars.txt"))[:, 3:6].T
ax.scatter(x, y, c="tab:red", s=0.5, alpha=0.08)
ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
