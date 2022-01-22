import glob
from PIL import Image

# filepaths
# fp_in = '/homes/fgarcia4/analysis/cluster_evolution_fs07/sequences/new_refine/clump_tracked_110921/*.png'
# fp_out = '/homes/fgarcia4/analysis/cluster_evolution_fs07/image_processor/gifs/color_coded_clumps.gif'
fp_in = 'C:/Users/144/Desktop/AstroSimulationResearch/cluster_evolution_fs07/sequences/new_refine/movie_200_to_400/*.png'
fp_out = 'C:/Users/144/Desktop/AstroSimulationResearch/cluster_evolution_fs07/image_processor/gifs/snapshot_200_400.gif'
print(fp_in)
# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
print ('Processing...')
img.save(fp=fp_out, format='GIF', append_images=imgs,
                  save_all=True, duration=250, loop=0)
print ('Image saved to' , fp_out)
