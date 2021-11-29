import os 
import yt 
import numpy as np
from yt.units import pc

FIELDS = ["Density",
          "x-velocity", "y-velocity", "z-velocity",
          "Pressure",
          "Metallicity",
          "dark_matter_density",
          "xHI", "xHII", "xHeII", "xHeIII"]
#extra particle fields
EPF= [('particle_family', 'b'),      
      ('particle_tag', 'b'),         
      ('particle_birth_epoch', 'd'), 
      ('particle_metallicity', 'd')] 

datadir = os.path.expanduser(
    'G:/My Drive/Research/AstrophysicsSimulation/DesktopEnvironment/data_globular_cluster/refine/output_00112'
    ) 
ds = yt.load(datadir, fields=FIELDS, extra_particle_fields=EPF)


dd = ds.sphere("max", (300, "pc"))
bounds = [[dd.center[i] - 250* pc, dd.center[i] + 250 * pc] for i in range(3)]

rho = np.mean(np.array(dd["gas", "density"]))
surf = ds.surface(dd, ("gas", "density"), rho)

surf.export_ply("output_112.ply", 
                color_field="temperature",
                color_map="rainbow",
                color_log=True,
                bounds=bounds) 
#%%

upload_id = surf.export_sketchfab(
    title = "RD0058 - 5e-27",
    description = "Extraction of Density (colored by Temperature) at 5e-27 " \
                + "g/cc from a galaxy formation simulation by Ryan Joung.",
    color_field = "temperature",
    color_map = "hot",
    color_log = True,
    bounds = bounds
)