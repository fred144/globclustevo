import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# make sure lenstronomy is installed, otherwise install the latest pip version
# try:
#     import lenstronomy
# except:
#     ! pip install lenstronomy

# lenstronomy module import
import lenstronomy.Util.data_util as data_util
import lenstronomy.Util.util as util
import lenstronomy.Plots.plot_util as plot_util
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.LightModel.Profiles.gaussian import GaussianEllipse
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST

LSST_g = LSST(band="g", psf_type="GAUSSIAN", coadd_years=10)
kwargs_g_band = LSST_g.kwargs_single_band()
LSST_r = LSST(band="r", psf_type="GAUSSIAN", coadd_years=10)
kwargs_r_band = LSST_r.kwargs_single_band()
LSST_i = LSST(band="i", psf_type="GAUSSIAN", coadd_years=10)
kwargs_i_band = LSST_i.kwargs_single_band()
gauss = GaussianEllipse()
kwargs_numerics = {"point_source_supersampling_factor": 1}


kwargs_model_postit = {
    "lens_model_list": ["SIE", "SHEAR"],  # list of lens models to be used
    "lens_light_model_list": ["INTERPOL"],  # list of unlensed light models to be used
    "source_light_model_list": [
        "INTERPOL"
    ],  # list of extended source models to be used
    "point_source_model_list": [
        "SOURCE_POSITION"
    ],  # list of point source models to be used
}

numpix = 64

sim_g = SimAPI(
    numpix=numpix, kwargs_single_band=kwargs_g_band, kwargs_model=kwargs_model_postit
)
sim_r = SimAPI(
    numpix=numpix, kwargs_single_band=kwargs_r_band, kwargs_model=kwargs_model_postit
)
sim_i = SimAPI(
    numpix=numpix, kwargs_single_band=kwargs_i_band, kwargs_model=kwargs_model_postit
)

# return the ImSim instance. With this class instance, you can compute all the
# modelling accessible of the core modules. See class documentation and other notebooks.
imSim_g = sim_g.image_model_class(kwargs_numerics)
imSim_r = sim_r.image_model_class(kwargs_numerics)
imSim_i = sim_i.image_model_class(kwargs_numerics)

# we make an 'image' based on a 2d gaussian
x_grid, y_grid = util.make_grid(numPix=10, deltapix=1)
flux = gauss.function(
    x_grid, y_grid, amp=1, sigma=2, e1=0.4, e2=0, center_x=0, center_y=0
)
image_gauss = util.array2image(flux)

# g-band

# lens light
kwargs_lens_light_mag_g = [
    {
        "magnitude": 17,
        "image": image_gauss,
        "scale": 0.1,
        "phi_G": 0,
        "center_x": 0.0,
        "center_y": 0,
    }
]
# source light
kwargs_source_mag_g = [
    {
        "magnitude": 22,
        "image": image_gauss,
        "scale": 0.03,
        "phi_G": 0.4,
        "center_x": 0.0,
        "center_y": 0,
    }
]


# and now we define the colors of the other two bands
kwargs_lens = [
    {"theta_E": 2.0, "e1": 0.4, "e2": -0.1, "center_x": 0, "center_y": 0},  # SIE model
    {"gamma1": 0.03, "gamma2": 0.01, "ra_0": 0, "dec_0": 0},  # SHEAR model
]

# lens light
kwargs_lens_light_mag_g = [
    {
        "magnitude": 14,
        "R_sersic": 0.6,
        "n_sersic": 4,
        "e1": 0.1,
        "e2": -0.1,
        "center_x": 0,
        "center_y": 0,
    }
]
# source light
kwargs_source_mag_g = [
    {
        "magnitude": 19,
        "R_sersic": 0.3,
        "n_sersic": 1,
        "e1": -0.3,
        "e2": -0.2,
        "center_x": 0,
        "center_y": 0,
    }
]
# point source
kwargs_ps_mag_g = [{"magnitude": 21, "ra_source": 0.03, "dec_source": 0}]

# r-band
g_r_source = 1  # color mag_g - mag_r for source
g_r_lens = -1  # color mag_g - mag_r for lens light
g_r_ps = 0
kwargs_lens_light_mag_r = copy.deepcopy(kwargs_lens_light_mag_g)
kwargs_lens_light_mag_r[0]["magnitude"] -= g_r_lens

kwargs_source_mag_r = copy.deepcopy(kwargs_source_mag_g)
kwargs_source_mag_r[0]["magnitude"] -= g_r_source

kwargs_ps_mag_r = copy.deepcopy(kwargs_ps_mag_g)
kwargs_ps_mag_r[0]["magnitude"] -= g_r_ps


# i-band
g_i_source = 2
g_i_lens = -2
g_i_ps = 0
kwargs_lens_light_mag_i = copy.deepcopy(kwargs_lens_light_mag_g)
kwargs_lens_light_mag_i[0]["magnitude"] -= g_i_lens

kwargs_source_mag_i = copy.deepcopy(kwargs_source_mag_g)
kwargs_source_mag_i[0]["magnitude"] -= g_i_source

kwargs_ps_mag_i = copy.deepcopy(kwargs_ps_mag_g)
kwargs_ps_mag_i[0]["magnitude"] -= g_i_ps

# turn magnitude kwargs into lenstronomy kwargs
kwargs_lens_light_g, kwargs_source_g, kwargs_ps_g = sim_g.magnitude2amplitude(
    kwargs_lens_light_mag_g, kwargs_source_mag_g, kwargs_ps_mag_g
)
kwargs_lens_light_r, kwargs_source_r, kwargs_ps_r = sim_r.magnitude2amplitude(
    kwargs_lens_light_mag_r, kwargs_source_mag_r, kwargs_ps_mag_r
)
kwargs_lens_light_i, kwargs_source_i, kwargs_ps_i = sim_i.magnitude2amplitude(
    kwargs_lens_light_mag_i, kwargs_source_mag_i, kwargs_ps_mag_i
)

image_g = imSim_g.image(kwargs_lens, kwargs_source_g, kwargs_lens_light_g, kwargs_ps_g)
image_r = imSim_r.image(kwargs_lens, kwargs_source_r, kwargs_lens_light_r, kwargs_ps_r)
image_i = imSim_i.image(kwargs_lens, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i)

# add noise
image_g += sim_g.noise_for_model(model=image_g)
image_r += sim_r.noise_for_model(model=image_r)
image_i += sim_i.noise_for_model(model=image_i)

# and plot it

img = np.zeros((image_g.shape[0], image_g.shape[1], 3), dtype=float)
img[:, :, 0] = plot_util.sqrt(image_g, scale_min=0, scale_max=1000)
img[:, :, 1] = plot_util.sqrt(image_r, scale_min=0, scale_max=1000)
img[:, :, 2] = plot_util.sqrt(image_i, scale_min=0, scale_max=1000)

plt.clf()
plt.imshow(img, aspect="equal", origin="lower")
plt.show()
