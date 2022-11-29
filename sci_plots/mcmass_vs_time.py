import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from modules.macros import t_myr_from_z
from matplotlib import cm
from scipy import interpolate

cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))

fs70_color = cmap[1]
fs35_color = cmap[2]

fs070_log_sfc = np.loadtxt("../sim_log_files/fs07_refine/logSFC")
redshft_fs070 = fs070_log_sfc[:, 2]
r_pc_cloud_fs070 = fs070_log_sfc[:, 4]
m_star_fs070 = fs070_log_sfc[:, 7]
m_cloud_fs070 = fs070_log_sfc[:, 5]
n_hydrogen_fs070 = fs070_log_sfc[:, 8]
metal_cloud_fs070 = fs070_log_sfc[:, 9]
t_myr_fs070 = t_myr_from_z(redshft_fs070)

fs035_log_sfc = np.loadtxt("../sim_log_files/fs035_ms10/logSFC")
redshft_fs035 = fs035_log_sfc[:, 2]
r_pc_cloud_fs035 = fs035_log_sfc[:, 4]
m_star_fs035 = fs035_log_sfc[:, 7]
m_cloud_fs035 = fs035_log_sfc[:, 5]
n_hydrogen_fs035 = fs035_log_sfc[:, 8]
metal_cloud_fs035 = fs035_log_sfc[:, 9]
t_myr_fs035 = t_myr_from_z(redshft_fs035)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 6), dpi=300)
ax[0].scatter(t_myr_fs070, m_cloud_fs070, s=1, label="70 per cent")
ax[0].scatter(t_myr_fs035, m_cloud_fs035, s=1, label="35 per cent")
ax[0].set(yscale="log", ylabel=r"MC Mass ($M_{\odot}$)", xlabel="Myr")

ax[0].legend()
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3), dpi=300)
ax[1].hist(t_myr_fs070, alpha=0.5)
ax[1].hist(t_myr_fs035, alpha=0.5)
ax[1].set(ylabel=r"Number of MCs Meeting Threshold", xlabel="Myr")
# ax[1].set(yscale="log")
