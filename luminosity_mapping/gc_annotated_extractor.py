import sys

sys.path.append("..")
import os
from modules.macros import filter_snapshots
import glob
import shutil

savepath = "../rendering/luminosity/fs07_refine/gc_annotated"
if not os.path.exists(savepath):
    print("# Creating new sequence directory", savepath)
    os.makedirs(savepath)

gc_profiler_directory = "../gc_profiles/profile_runs/fs07_refine/fof_best_v3"

gc_prof = filter_snapshots(gc_profiler_directory, 113, 918, 1)

for gc_dir in gc_prof:
    halos_annotated = glob.glob(os.path.join(gc_dir, "annotated*.png"))[0]
    shutil.copy(src=halos_annotated, dst=savepath)
