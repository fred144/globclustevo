import numpy as np
import numpy.random
import random
import math as m
import matplotlib.pyplot as plt
import scipy

# from sciplt.signal import fftconvolve
import os


# ----------------------------------------------------------------
# The function to calculate the new coordinates after a rotation
# ----------------------------------------------------------------


def xy_rotate(x, y, xcen, ycen, phi):

    """x, y: numpy ndarrays with same size
    giving coordinates in the old system
    xcen: old-system x coordinate of the new origin
    ycen: old-system y coordinate of the new origin
    phi: angle c.c.w. in degrees from old x to new x axis"""
    phirad = np.deg2rad(phi)
    xnew = (x - xcen) * np.cos(phirad) + (y - ycen) * np.sin(phirad)
    ynew = (y - ycen) * np.cos(phirad) - (x - xcen) * np.sin(phirad)
    return (xnew, ynew)


# ----------------------------------------------------------------
# A two dimensional Sersic profile
# ----------------------------------------------------------------


def sersic(x, y, par):
    """Parameters for Sersic profile
    par[0]: Effective intensity
    par[1]: Effective radius
    par[2]: x-center
    par[3]: y-center
    par[4]: axis ratio
    par[5]: counter clockwise major-axis rotation w.r.t. x-axis
    par[6]: sersic index"""

    b = 2.0 * par[6] - 0.3271
    n = 1.0 / par[6]
    (xnew, ynew) = xy_rotate(x, y, par[2], par[3], par[5])
    r_ell_sq = ((xnew**2) * (par[4] ** 2) + ynew**2) / np.abs(par[1])
    g = par[0] * np.exp(-b * (r_ell_sq**n - 1.0))
    return g


# ----------------------------------------------------------------
# Deflection angle due to Singular Isothermal Ellipsoid lens
# ----------------------------------------------------------------


def deflect_SIE(x, y, par):

    """Function to calculate the deflection angle due to Isothermal Ellipsoid lens potential & shear
    x, y: vectors or images of coordinates;
    par: vector of parameters defined as follows:
    par[0]: lens strength, or 'Einstein radius'
    par[1]: axis ratio (default=1.0)
    par[2]: major axis Position Angle in degrees c.c.w. of x axis"""

    b = np.abs(par[0])
    q = np.abs(par[1])
    phiq = par[2]
    ss = par[3]
    phis = par[4]

    # The case q > 1 :
    if q > 1.0:
        q = 1.0 / q
        phiq = phiq + 90.0

    # Go into rotated coordinates of the potential:
    phirad = np.deg2rad(phiq)
    xsie = x * np.cos(phirad) + y * np.sin(phirad)
    ysie = y * np.cos(phirad) - x * np.sin(phirad)

    r = np.sqrt(x**2 + y**2)
    t = np.arctan2(y, x)

    f = np.sqrt(1.0 - q**2.0)
    xg1 = np.sqrt(q) * b * np.arcsinh((f / q) * np.cos(t)) / f
    yg1 = np.sqrt(q) * b * np.arcsin(f * np.sin(t)) / f

    # External shear
    phisrad = np.deg2rad(phis)
    xg2 = ss * (x * np.cos(2.0 * phisrad) + y * np.sin(2.0 * phisrad))
    yg2 = ss * (y * np.cos(2.0 * phisrad) + x * np.sin(2.0 * phisrad))

    xg = xg1 + xg2
    yg = yg1 + yg2

    return xg, yg


# ------X-Y Grid of KiDS------------------------------------------

(nx, ny) = (101, 101)  # No. of pixels
(Lx, Ly) = (20, 20)  # Total field of view in arc-sec

grid_resolution = float(Lx) / float(nx)
X = np.linspace(-Lx / 2.0, Lx / 2.0, nx)
Y = np.linspace(-Ly / 2.0, Ly / 2.0, nx)
x, y = np.meshgrid(X, Y, sparse=True)


# ------------Caustic curve------------------------------------------------------


def caust_t(phi_src, beta, theta, phi):
    f = np.sqrt(1 - beta**2)
    df = np.sqrt(beta) / f
    r_t_x = theta * (df * np.arccosh(1 / beta) - np.sqrt(beta)) * np.cos(phi_src) ** 3
    r_t_y = theta * (df * np.arccos(beta) - 1 / np.sqrt(beta)) * np.sin(phi_src) ** 3
    (r_t_xx, r_t_yy) = xy_rotate(r_t_x, r_t_y, 0.0, 0.0, -phi)
    return r_t_xx, r_t_yy


# ----------Function to determine source position on the source plane------------


def src(beta, theta, r_eff, phi):
    phi_src = 2 * np.pi * np.random.uniform(0, 1)

    beta = q_l
    theta = b

    f = np.sqrt(1 - beta**2)
    df = np.sqrt(beta) / f
    r_x = (
        theta
        * (df * np.arccosh(1 / beta) - np.sqrt(beta) + 0.5 * r_eff)
        * np.cos(phi_src) ** 3
    )
    r_y = (
        theta
        * (df * np.arccos(beta) - 1 / np.sqrt(beta) - 0.5 * r_eff)
        * np.sin(phi_src) ** 3
    )

    R = np.sqrt(r_x**2 + r_y**2)
    r_src = R * np.sqrt(np.random.uniform(0, 1))
    phi_caustic = np.arctan2(r_y, r_x)

    r_xx = r_src * np.cos(phi_caustic)
    r_yy = r_src * np.sin(phi_caustic)

    (xx_src, yy_src) = xy_rotate(r_xx, r_yy, 0.0, 0.0, -phi)

    return (xx_src, yy_src)


# -------- Let's test the caustic curve for some random parameters---------------

b = 10 ** (np.random.uniform(np.log10(0.5), np.log10(5.0)))  # Einstein Radius
q_l = np.random.uniform(0.3, 0.99)  # Axis ratio
phi_l = np.random.uniform(0, 180)  # major-minor axis angle
r_eff = 10 ** (
    np.random.uniform(np.log10(0.2), np.log10(0.6))
)  # Effective radius of the main Sersic source


plt.figure()
plt.plot()
plt.title("Caustic on the source plane (red) and random source co-ordinates (green)")
plt.xlabel("arc second")
plt.ylabel("arc second")

for i in range(5000):
    (x_s, y_s) = src(q_l, b, r_eff, phi_l)
    # print ("Random source co-ordinate (x, y):", x_s, y_s)
    plt.plot(x_s, y_s, "o", markersize=3, color="g")

plt.plot(*caust_t(np.linspace(0, 2 * np.pi, 200), q_l, b, phi_l), color="r", lw=2)
plt.show()


# --------Power spectrum ------------------------------------------


def powspec(L, variance, Npix, Psum, power):
    if L <= 0:
        P = 0.0
    else:
        A = variance * (Npix**2.0) / (2.0 * Psum)
        P = A * L ** (power)
    return P


def Psum_calculator(nx, ny, Lx, Ly, power):

    resolution = float(Lx) / float(nx)
    lxaxis = np.fft.fftfreq(nx, resolution)
    lyaxis = np.fft.fftfreq(ny, resolution)

    lx = list(np.zeros([nx, 1]))
    ly = list(np.zeros([ny, 1]))

    for x in range(len(lx)):
        lx[x] = lxaxis

    for y in range(len(ly)):
        ly[y] = lyaxis

    lx = np.array(lx)
    ly = np.transpose(np.array(ly))
    l = np.sqrt(lx**2.0 + ly**2.0)

    summ = 0.0
    for y in range(np.shape(l)[0]):
        for x in range(np.shape(l)[1]):
            if l[y][x] == 0.0:
                summ += 0.0
            else:
                summ += l[y][x] ** (power)
    return summ


# ----- Function to simulate GRF on a Fourier Grid-------------------------------


def gauss_rand_2d(par):

    nx = par[0].astype(int)
    ny = par[1].astype(int)
    Lx = par[2]
    Ly = par[3]
    var = par[4]
    power = par[5]

    j = 0 + 1j  # Defining the complex number
    plane = np.zeros(
        [nx, ny], dtype="cfloat"
    )  # Empty matrix to be filled in for the Fourier plane

    # lxaxis = np.append (np.arange (0.,(nx/2.)/Lx, 1./Lx), np.arange ((-nx/2.)/Lx, 0.,1./Lx))
    # lyaxis = np.append (np.arange (0.,(ny/2.)/Ly, 1./Ly), np.arange ((-ny/2.)/Ly, 0.,1./Ly))

    resolution = float(Lx) / float(nx)
    lxaxis = np.fft.fftfreq(nx, resolution)
    lyaxis = np.fft.fftfreq(ny, resolution)

    Psum = Psum_calculator(nx, ny, Lx, Ly, power)

    for y in range(np.shape(plane)[0]):
        for x in range(np.shape(plane)[1]):
            # The coordinates centered at x = n/2, y = n/2
            i1 = x - nx // 2
            j1 = y - ny // 2

            # The coordinates in the Fourier plane
            lx = lxaxis[x]
            ly = lyaxis[y]
            # The magnitude of the l-vector
            l = np.sqrt(lx**2.0 + ly**2.0)

            # Polar Box-Muller transform
            sigma = m.sqrt(powspec(l, var, nx * ny, Psum, power))
            s = 1.1
            while s > 1.0:
                u = np.random.uniform(-1.0, 1.0)
                v = np.random.uniform(-1.0, 1.0)
                s = u**2.0 + v**2.0
            fac = m.sqrt(-2.0 * m.log(s) / s)
            z1 = u * fac * sigma
            z2 = v * fac * sigma

            # Filling in the grid
            if x == 0.0 and y == 0.0:  # average of the field
                plane[y][x] = 0.0

            # three points that need to be real valued to get a real image after FFT:
            elif x == 0 and y == ny // 2:
                plane[y][x] = z1
            elif x == nx // 2 and y == 0:
                plane[y][x] = z1
            elif x == nx // 2 and y == ny // 2:
                plane[y][x] = z1

            else:
                plane[y][x] = z1 + j * z2

            # Creating symmetry f(k) = f*(-k)
            y2 = -(j1 + ny // 2)
            x2 = -(i1 + nx // 2)
            plane[y2][x2] = plane[y][x].conjugate()

        if y > np.shape(plane)[0] // 2.0:
            break
    return np.fft.ifftshift(np.fft.ifft2(plane)).real


# Let's test the Gaussian Random Field generator code for some random parameters

# GRF parameters

var = random.choice(np.logspace(np.log10(1.000e-04), np.log10(1.000e-01), num=100))
pow = 6.0
grpar = np.asarray([nx, ny, Lx, Ly, var, -pow])

gauss_rand = gauss_rand_2d(grpar)  # GRF

print("Variance:", var)
print("Power law exponent:", -pow)


plt.figure()
plt.title("Gaussian Random Field")
plt.xlabel("arc second")
plt.ylabel("arc second")
plt.imshow(gauss_rand, interpolation="nearest", extent=[-10, 10, -10, 10])
plt.colorbar(fraction=0.046)
plt.show()

# -------------- Function to define the Lens plane parameters --------------


def lens_param(nx, ny, Lx, Ly, N):
    # -------- lens mass parameters -------------------------------
    b = 10 ** (np.random.uniform(np.log10(1.5), np.log10(5.0)))  # Einstein Radius
    q_l = np.random.uniform(0.3, 0.99)  # Axis ratio
    phi_l = np.random.uniform(0, 180)  # major-minor axis angle
    ss = np.random.uniform(0, 0.05)  # shear strength
    phi_ss = phi_l + np.random.uniform(0, 180)  # shear angle

    # --------- Lens parameter array -----------
    lpar = np.asarray([b, q_l, phi_l, ss, phi_ss])  # SIE parameters

    # ------------- GRF parameters --------------------------------
    var = random.choice(np.logspace(np.log10(1.000e-04), np.log10(1.000e-01), num=N))
    pow = 6.0
    grpar = np.asarray([nx, ny, Lx, Ly, var, -pow])
    return (lpar, grpar)


# -------------- Function to define the Source plane parameters --------------


def src_param(N_blob):

    # ------------- main Sersic source parameters ------------------------
    i_eff = 1.0
    r_eff = 10 ** (np.random.uniform(np.log10(0.2), np.log10(0.6)))
    q_s = np.random.uniform(0.3, 0.99)
    phi_s = phi_l + np.random.uniform(0, 180)
    n = np.random.uniform(0.5, 5)
    (x_s, y_s) = src(q_l, b, r_eff, phi_l)

    serpar = np.asarray([i_eff, r_eff, x_s, y_s, q_s, phi_s, n])  # main sersic profile

    # -------- parameters for the blob sub-structures in the source ---------

    # ------ common parameters of all the blobs
    i_eff_blob = (
        1.0  # we keep the initial effective intensity of the profile to unity as
    )
    # it will be taken care of by the blob_amp_param

    q_s_blob = 0.999  # We keep the blobs circular
    phi_s_blob = 0.0

    # we choose the centers of the blobs from a Gaussian PDF around the main blob.
    a = r_eff  # sigma of this PDF is a function of the main effective radius

    # maximum flux ratio of the individual blobs. We choose a uniform random number between 0 and f_rat
    f_rat = 0.2

    # -------------------------------------------------------------

    blob_sersic_params = []  # list of Sersic parameters of the blobs
    blob_amp_param = []  # list of amplitudes of the blobs

    for b_n in range(N_blob + 1):
        # Sersic parameters of the blob structures
        r_eff_blob = (
            np.random.uniform(0.01, 0.1) * r_eff
        )  # effective radius of the first blob
        n_blob = np.random.uniform(0.5, 5)  # sersic index
        x_s_blob = np.random.normal(x_s, a, 1)  # the (x,y) positions of the blobs
        y_s_blob = np.random.normal(y_s, a, 1)
        a_blob = np.random.uniform(0, f_rat)

        blob_sersic_params.append(
            [i_eff_blob, r_eff_blob, x_s_blob, y_s_blob, q_s_blob, phi_s_blob, n_blob]
        )
        blob_amp_param.append([a_blob])

    blob = np.asarray(blob_sersic_params)  # blob sersic profile parameters
    a_blob = np.asarray(blob_amp_param)  # amplitudes

    return (serpar, blob, a_blob)


(lpar, grpar) = lens_param(nx, ny, Lx, Ly, 1)

N_blob = np.random.randint(
    0, 5
)  # random number of blobs added with the main source (max. 5)
(serpar, blob, a_blob) = src_param(N_blob)
# -------------------------------------------------------------
particle_data = np.loadtxt(
    "../particle_data/pop_2_data/fs07_refine/pos_00471_474_01_myr.txt"
)


xy_lums, _, _ = np.histogram2d(
    particle_data[:, 2],
    particle_data[:, 3],
    bins=101,
    # weights=f7_star_lums,
    normed=False,
    range=[[-200, 200], [-200, 200]],
)
source = xy_lums.T

(xg, yg) = deflect_SIE(x, y, lpar)  # The NIE gradient field
gauss_rand = gauss_rand_2d(grpar)
(x_r, y_r) = np.gradient(gauss_rand, grid_resolution, grid_resolution)
# source = sersic(x, y, serpar)
image = sersic(
    x - xg - x_r, y - yg - y_r, serpar
)  # The lensed image using main Sersic + lens + GRF

for jj in range(N_blob + 1):
    source = source + a_blob[jj] * sersic(x, y, blob[jj])
    image = image + a_blob[jj] * sersic(x - xg - x_r, y - yg - y_r, blob[jj])

# -------------------------------------------------------------------

# In order to convolve with a PSF please uncomment the follwoing line
# image = sciplt.signal.fftconvolve (image, psf, mode = 'same')

# -------------------------------------------------------------------

# If you want to normalise the image w.r.t. the maximum pixel value
# image = image/image.max()

# -------------------------------------------------------------------

# plt.imshow()
# plt.imshow(
#     xy_lums.T,
#     extent=[-200, 200, -200, 200],
#     # norm=LogNorm(6e-8, 1e-5),
#     # cmap="inferno",
#     origin="lower",
#     # interpolation="gaussian",
# )
# path = os.path.join("/content/drive/My Drive/Strong_Lens_Simulation/", str(id))
# os.makedirs(path)

# pf.writeto(os.path.join(path, '%s.fits' %id), image) #write to a FITS file
# hu = pf.getheader(os.path.join(path,'%s.fits' %id)) # edit the FITS header
# hu['LENSER'] = lpar[0]
# hu['LENSAR'] = lpar[1]
# hu['LENSAA'] = lpar[2]
# hu['LENSSH'] = lpar[3]
# hu['LENSSA'] = lpar[4]

# hu['SRCER'] = serpar[1]
# hu['SRCX'] = serpar[2]
# hu['SRCY'] = serpar[3]
# hu['SRCAR'] = serpar[4]
# hu['SRCAA'] = serpar[5]
# hu['SRCSI'] = serpar[6]


# hu.comments['LENSER'] = 'Einstein radius of the lens in arc seconds'
# hu.comments['LENSAR'] = 'axis ratio of the lens potential'
# hu.comments['LENSAA'] = 'major-minor axis angle of the lens potential'
# hu.comments['LENSSH'] = 'external shear strength'
# hu.comments['LENSSA'] = 'external shear angle'

# hu.comments['SRCER'] = 'eff-radius of sersic profile in arc seconds'
# hu.comments['SRCX'] = 'x co-ordinate of source'
# hu.comments['SRCY'] = 'y co-ordinate of source'
# hu.comments['SRCAR'] = 'axis ratio of the source'
# hu.comments['SRCAA'] = 'major-minor axis angle of the source'
# hu.comments['SRCSI'] = 'sersic index of the source'

# pf.writeto(os.path.join(path, '%s.fits' %id), image, hu, clobber=True)

# u = u+1

# -----------------------------------------------------------------------

# Plots of the source and the simulated lensed galaxies
plt.figure(dpi=300)
plt.subplot(1, 2, 1)
plt.title("Source plane")
plt.xlabel("arc second")
plt.ylabel("arc second")
plt.imshow(source, interpolation="nearest", extent=[-10, 10, -10, 10])
plt.colorbar(fraction=0.046)
plt.plot(*caust_t(np.linspace(0, 2 * np.pi, 200), q_l, b, phi_l), color="r", lw=1)

plt.subplot(1, 2, 2)
plt.title("Image plane")
plt.xlabel("arc second")
plt.imshow(image, interpolation="nearest", extent=[-10, 10, -10, 10])
plt.colorbar(fraction=0.046)
plt.show()
