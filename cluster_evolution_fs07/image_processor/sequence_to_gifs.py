import glob
from PIL import Image

# filepaths
fp_in = "/homes/fgarcia4/analysis/cluster_evolution_fs07/sequences/refine_clump_tracked/*.png"
fp_out = '/homes/fgarcia4/analysis/cluster_evolution_fs07/image_processor/gifs/refine_clump_tracked.gif'
print(fp_in)
# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
print ('Processing...')
img.save(fp=fp_out, format='GIF', append_images=imgs,
                  save_all=True, duration=300, loop=0)
print ('Image saved to' , fp_out)
