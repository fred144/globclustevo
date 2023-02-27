import sys


sys.path.append("../")


sys.path.insert(1, "/home/fgarcia4/.local/lib/python3.8/site-packages")
import numpy as np
import os
import glob
from modules.luminosity.lum_functions import lum_look_up_table
from modules.match_t_sims import find_matching_time, get_snapshots
from modules.macros import filter_snapshots
from scipy.ndimage import gaussian_filter


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib import colors

import yt
from modules.macros import filter_snapshots, ram_fields, t_myr_from_z, code_age_to_myr

# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib as mpl


yt.enable_parallelism()
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "font.size": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
    }
)

main_dir = os.path.relpath("../../cosm_test_data/fs035_ms10/")

cell_fields, epf = ram_fields()

pre_ds = yt.load(
    os.path.join(main_dir, "output_00560/info_00560.txt"),
    fields=cell_fields,
    extra_particle_fields=epf,
)
post_ds = yt.load(
    os.path.join(main_dir, "output_00561/info_00561.txt"),
    fields=cell_fields,
    extra_particle_fields=epf,
)

pre_ad = pre_ds.all_data()
post_ad = post_ds.all_data()

t_myr = float(post_ds.current_time.in_units("Myr"))
redshift = float(post_ds.current_redshift)
current_hubble = post_ds.hubble_constant
#%%
sfc_kazu_radii = np.abs(
    pre_ds.arr(pre_ad["SFC", "particle_metallicity"], "code_length").to("pc")
)

sfc_ages = code_age_to_myr(
    post_ad["SFC", "particle_birth_epoch"], current_hubble, unique_age=False
)
