import sys

sys.path.append("../")
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
from modules.macros import (
    filter_snapshots,
    common_filter_snapshots,
    t_myr_from_z,
    z_from_t_myr,
)
from modules.match_t_sims import find_matching_time, get_snapshots
import matplotlib.lines as mlines
from scipy import stats

# can easily be done with seaborn linregress, but here is a confidence band scripts
# for matplotlib, scipy , and numpy only
# https://gist.github.com/rsnemmen/f2c03beb391db809c90f
def scatterfit(x, y, a=None, b=None):
    """
    Compute the mean deviation of the data about the linear model given if A,B
    (y=ax+b) provided as arguments. Otherwise, compute the mean deviation about
    the best-fit line.

    x,y assumed to be Numpy arrays. a,b scalars.
    Returns the float sd with the mean deviation.

    Author: Rodrigo Nemmen
    """

    if a == None:
        # Performs linear regression
        a, b, r, p, err = stats.linregress(x, y)

    # Std. deviation of an individual measurement (Bevington, eq. 6.15)
    N = np.size(x)
    sd = 1.0 / (N - 2.0) * np.sum((y - a * x - b) ** 2)
    sd = np.sqrt(sd)

    return sd


def confband(xd, yd, a, b, conf=0.6827, x=None):
    """
    Calculates the confidence band of the linear regression model at the desired confidence
    level, using analytical methods. The 2sigma confidence interval is 95% sure to contain
    the best-fit regression line. This is not the same as saying it will contain 95% of
    the data points.

    Arguments:
    - conf: desired confidence level, by default 0.95 (2 sigma)
    - xd,yd: data arrays
    - a,b: linear fit parameters as in y=ax+b
    - x: (optional) array with x values to calculate the confidence band. If none is provided, will
      by default generate 100 points in the original x-range of the data.

    Returns:
    Sequence (lcb,ucb,x) with the arrays holding the lower and upper confidence bands
    corresponding to the [input] x array.

    Usage:
    >>> lcb,ucb,x=nemmen.confband(all.kp,all.lg,a,b,conf=0.95)
    calculates the confidence bands for the given input arrays

    >>> pylab.fill_between(x, lcb, ucb, alpha=0.3, facecolor='gray')
    plots a shaded area containing the confidence band

    References:
    1. http://en.wikipedia.org/wiki/Simple_linear_regression, see Section Confidence intervals
    2. http://www.weibull.com/DOEWeb/confidence_intervals_in_simple_linear_regression.htm

    Author: Rodrigo Nemmen
    v1 Dec. 2011
    v2 Jun. 2012: corrected bug in computing dy
    """
    alpha = 1.0 - conf  # significance
    n = xd.size  # data sample size

    if x == None:
        x = np.linspace(xd.min() - 1.0, xd.max() + 1.0, 100)

    # Predicted values (best-fit model)
    y = a * x + b

    # Auxiliary definitions
    sd = scatterfit(xd, yd, a, b)  # Scatter of data about the model
    sxd = np.sum((xd - xd.mean()) ** 2)
    sx = (x - xd.mean()) ** 2  # array

    # Quantile of Student's t distribution for p=1-alpha/2
    q = stats.t.ppf(1.0 - alpha / 2.0, n - 2)

    # Confidence band
    dy = q * sd * np.sqrt(1.0 / n + sx / sxd)
    ucb = y + dy  # Upper confidence band
    lcb = y - dy  # Lower confidence band

    return lcb, ucb, x


cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))

f7_mc_imf_clr = cmap[0]
f7_bsc_mf_clr = cmap[1]
f3_mc_imf_clr = cmap[2]
f3_bsc_mf_clr = cmap[3]

# f7_strt = 113
# f7_end = 1318
# f3_strt = 154
# f3_end = 1469
# step = 1
f7_strt = 113
f7_end = 1196
f3_strt = 154
f3_end = 1368
step = 1
# sample the matched snapshots for plotting by indexing
# try snapshot 873
strt = 1083  # 800
end = 1084  # f3_matched_nums.size
st = 1

# f7_prof_dir = r"../gc_profiles/profile_runs/fs07_refine/fof_best"
# f3_prof_dir = r"../gc_profiles/profile_runs/fs035_ms10/fof_best"

f7_halo_dir = r"../halo_data/fs07_refine/fof_best"
f3_halo_dir = r"../halo_data/fs035_ms10/fof_best"

profiler_data = (
    "/home/fabg/g_drive/Research/AstrophysicsSimulation/DesktopEnvironment/"
    "data_globular_cluster/gc_profiles/profile_runs/"
)

f7_prof_dir = profiler_data + "fs07_refine/fof_best"
f3_prof_dir = profiler_data + "fs035_ms10/fof_best"

# using pop2 data to construct lookuptables
f7_pop2_ds = filter_snapshots(
    r"../particle_data/pop_2_data/fs07_refine", f7_strt, f7_end, step
)
f3_pop2_ds = filter_snapshots(
    r"../particle_data/pop_2_data/fs035_ms10", f3_strt, f3_end, step
)


_, f3_matched_nums = find_matching_time(
    sequence=f7_pop2_ds, look_up_sequence=f3_pop2_ds
)
f3_pro_ds = filter_snapshots(f3_prof_dir, f3_strt, f3_end, step)


f7_pro_ds = filter_snapshots(f7_prof_dir, f7_strt, f7_end, step)[strt:end:st]
f3_pro_ds = get_snapshots(f3_pro_ds, get_list=f3_matched_nums)[strt:end:st]

f7_halo_ds = filter_snapshots(f7_halo_dir, f7_strt, f7_end, step)[strt:end:st]
f3_halo_ds = get_snapshots(
    filter_snapshots(f3_halo_dir, f3_strt, f3_end, step), get_list=f3_matched_nums
)[strt:end:st]

f3_pop2_ds = get_snapshots(f3_pop2_ds, get_list=f3_matched_nums)[strt:end:st]
f7_pop2_ds = f7_pop2_ds[strt:end:st]


def metal_lookup(log_sfc_path, bsc_form_times):

    log = np.loadtxt(log_sfc_path)
    z_form = log[:, 2]
    t_form = t_myr_from_z(z_form)
    m_sun_form = log[:, 7]  # mass in stars
    z_sun = log[:, 9]  # solar metallicity

    residuals = np.abs(t_form - bsc_form_times[:, np.newaxis])
    closest_match_idxs = np.argmin(residuals, axis=1)
    bsc_metals = z_sun[closest_match_idxs]
    bsc_m_sun_form = m_sun_form[closest_match_idxs]
    return bsc_metals, bsc_m_sun_form


def lin_model(x, slope, intercept):
    return (slope * x) + intercept


cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))

fs70_color = cmap[1]
fs35_color = cmap[2]

for sn, (f7, f3) in enumerate(zip(f7_pro_ds, f3_pro_ds)):
    # if i == 0:
    #     continue
    f7_prof_data = np.loadtxt(os.path.join(f7, "info.txt"))
    f3_prof_data = np.loadtxt(os.path.join(f3, "info.txt"))

    f7_halo_data = np.loadtxt(os.path.join(f7_halo_ds[sn], "fof_info.txt"))

    # load the profiled BSCs data
    f7_t_myr = f7_prof_data[0, 0]
    f7_labels = f7_prof_data[:, 1]
    f7_ages = f7_prof_data[:, 2]
    f7_bes = f7_t_myr - f7_ages
    f7_mass = f7_prof_data[:, 3]
    f7_core_mass = f7_prof_data[:, 4]
    f7_vir_rad = f7_prof_data[:, 5]
    # derived qunatities
    f7_core_rad = f7_prof_data[:, 6]
    f7_core_err = f7_prof_data[:, 7]
    f7_alpha = f7_prof_data[:, 8]
    f7_alpha_err = f7_prof_data[:, 9]
    f7_sig_0 = f7_prof_data[:, 10]
    f7_sig_0_err = f7_prof_data[:, 11]
    f7_sig_bg = f7_prof_data[:, 12]
    f7_sig_bg_err = f7_prof_data[:, 13]
    f7_fit_pval = f7_prof_data[:, 14]
    f7_half_mass_rad = f7_prof_data[:, 15]
    f7_half_light_rad = f7_prof_data[:, 16]
    f7_tot_light = f7_prof_data[:, 17]
    f7_metal, f7_orig_mass = metal_lookup("../sim_log_files/fs07_refine/logSFC", f7_bes)
    f7_redshift = np.loadtxt("../sim_log_files/fs07_refine/logSFC")[:, 2]
    # half efficiency
    f3_t_myr = f3_prof_data[0, 0]
    f3_labels = f3_prof_data[:, 1]
    f3_ages = f3_prof_data[:, 2]
    f3_bes = f3_t_myr - f3_ages
    f3_mass = f3_prof_data[:, 3]
    f3_core_mass = f3_prof_data[:, 4]
    f3_vir_rad = f3_prof_data[:, 5]
    # derived qunatities
    f3_core_rad = f3_prof_data[:, 6]
    f3_core_err = f3_prof_data[:, 7]
    f3_alpha = f3_prof_data[:, 8]
    f3_alpha_err = f3_prof_data[:, 9]
    f3_sig_0 = f3_prof_data[:, 10]
    f3_sig_0_err = f3_prof_data[:, 11]
    f3_sig_bg = f3_prof_data[:, 12]
    f3_sig_bg_err = f3_prof_data[:, 13]
    f3_fit_pval = f3_prof_data[:, 14]
    f3_half_mass_rad = f3_prof_data[:, 15]
    f3_half_light_rad = f3_prof_data[:, 16]
    f3_tot_light = f3_prof_data[:, 17]
    f3_metal, f3_orig_mass = metal_lookup("../sim_log_files/fs035_ms10/logSFC", f3_bes)
    f3_redshift = np.loadtxt("../sim_log_files/fs035_ms10/logSFC")[:, 2]

    with plt.rc_context(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "font.size": 10,
        }
    ):

        cmap = plt.cm.get_cmap("viridis")
        # scale_factor = 50  # scale factor for the sizes
        # # map to differnt sizes for better plotting
        # f7_half_radii = scale_factor * f7_half_mass_rad
        # f3_half_radii = scale_factor * f3_half_mass_rad
        # a cluster should have at least 25 stars
        f7_mask = (f7_mass > 250) & (f7_alpha < 5)  # & & (f7_vir_rad < 10)
        f3_mask = (f3_mass > 250) & (f3_alpha < 5)  # & & (f3_vir_rad < 10)

        x_vars = [
            np.nan,
            ((f7_mass * 0.5)[f7_mask], (f3_mass * 0.5)[f3_mask]),
            (f7_metal[f7_mask], f3_metal[f3_mask]),
            (f7_mass[f7_mask], f3_mass[f3_mask]),
            # (f7_mass[f7_mask], f3_mass[f3_mask]),  #
            (f7_half_mass_rad[f7_mask], f3_half_mass_rad[f3_mask]),
            # (f7_half_mass_rad[f7_mask], f3_half_mass_rad[f3_mask]),  #
            (f7_metal[f7_mask], f3_metal[f3_mask]),
            (f7_core_rad[f7_mask], f3_core_rad[f3_mask]),
            (f7_alpha[f7_mask], f3_alpha[f3_mask]),
        ]
        y_vars = [
            np.nan,
            (f7_half_mass_rad[f7_mask], f3_half_mass_rad[f3_mask]),
            (
                (f7_core_mass / f7_core_rad**3)[f7_mask],
                (f3_core_mass / f3_core_rad**3)[f3_mask],
            ),
            (f7_core_mass[f7_mask], f3_core_mass[f3_mask]),
            # ((f7_core_mass / f7_mass)[f7_mask], (f3_core_mass / f3_mass)[f3_mask]),  #
            ((f7_core_rad)[f7_mask], (f3_core_rad)[f3_mask]),
            # (
            #     (f7_core_rad / f7_half_mass_rad)[f7_mask],
            #     (f3_core_rad / f3_half_mass_rad)[f3_mask],
            # ),  #
            (f7_half_mass_rad[f7_mask], f3_half_mass_rad[f3_mask]),
            (f7_sig_0[f7_mask], f3_sig_0[f3_mask]),
            (f7_core_rad[f7_mask], f3_core_rad[f3_mask]),
        ]
        x_labels = [
            np.nan,
            r"$\log_{10} \: \mathrm{M_{half}} \: \mathrm{(M_{\odot})} $",
            r"$\log_{10} \: \mathrm{Z_{BSC}\:\left(Z_{\odot}\right)}$",
            r"$\log_{10} \: \mathrm{M_{BSC}} \: \mathrm{(M_{\odot})}$",
            # r"$\log_{10} \: \mathrm{M_{BSC}} \: \mathrm{(M_{\odot})}$",  #
            r"$\log_{10} \: \mathrm{ R_{half} \:(pc)}$",
            # r"$\log_{10} \: \mathrm{ R_{half} \:(pc)}$",  #
            r"$\log_{10} \: \mathrm{Z_{BSC}\:\left(Z_{\odot}\right)}$",
            r"$\log_{10} \: R\mathrm{_{core} \: (pc)}$",
            r"$\alpha$",
        ]
        y_labels = [
            np.nan,
            r"$\log_{10} \: \mathrm{R_{half} \: (pc)}$",
            r"$\log_{10} \: \rho_{\mathrm{core }} \: (\mathrm{M_{\odot} \: pc^{-3}})$",
            r"$\log_{10} \: \mathrm{M_{core}}\: \mathrm{(M_{\odot})}$",
            # r"$\log_{10} \: \mathrm{M_{core}}\: / \mathrm{M_{BSC}}}$",  #
            r"$\log_{10} \: R_{\mathrm{core}} \mathrm{(pc)}$",
            # r"$\log_{10} \: R_{\mathrm{core}} / \mathrm{R_{half}}$",  #
            r"$\log_{10} \: \mathrm{R_{half}\:(pc)}$ ",
            r"$\log_{10} \: \mathrm{\Sigma_0\:\left(M_{\odot}\:pc^{-2}\right)}$",
            r"$\log_{10} \: R\mathrm{_{core}}$",
        ]
        # loop through some possible plots.
        fig, ax = plt.subplots(2, 4, figsize=(8.5, 4), dpi=400)
        plt.subplots_adjust(hspace=0.30, wspace=0.35)
        axs = ax.ravel()
        axs[0].set_visible(False)

        subplt_lbl = [
            "",
            "$\mathrm{(a)}$",
            "$\mathrm{(b)}$",
            "$\mathrm{(c)}$",
            "$\mathrm{(d)}$",
            "$\mathrm{(e)}$",
            "$\mathrm{(f)}$",
            "$\mathrm{(g)}$",
            "$\mathrm{(h)}$",
        ]
        for i, (x, y) in enumerate(zip(x_vars, y_vars)):
            if i == 0:
                continue

            if i == 7:
                f7_x = x[0]
                f3_x = x[1]

                f7_y = np.log10(y[0])
                f3_y = np.log10(y[1])

            else:
                f7_x = np.log10(x[0])
                f3_x = np.log10(x[1])

                f7_y = np.log10(y[0])
                f3_y = np.log10(y[1])

            if i == 1:
                # power = 1.315
                # intercept = -4.947
                # rho_of_r_prefactor = 3 * 10**intercept / 4 * np.pi
                # power_of_r = 3 - power
                # convert to linear values to find an average density to us
                f7_x = np.array(10**f7_x)
                f7_y = np.array(10**f7_y)
                max_rho = np.max((f7_x / f7_y**3) * (0.75 / np.pi))
                tenth_max_rho = max_rho / 10
                tenthousandth_max_rho = max_rho / 10000
                ana_mhalf = np.linspace(f7_x.min() * 0.5, f7_x.max() * 1.5, 100)
                ana_rhalf = ((ana_mhalf / max_rho) * (0.75 / np.pi)) ** (1 / 3)

                print(f7_y)
                print(f7_x)
                # print(avg_rho)
                print(max_rho)
                axs[i].plot(
                    np.log10(ana_mhalf),
                    np.log10(ana_rhalf),
                    lw=1,
                    ls="--",
                    color="grey",
                    zorder=0,
                )
                axs[i].plot(
                    np.log10(ana_mhalf),
                    np.log10(((ana_mhalf / tenth_max_rho) * (0.75 / np.pi)) ** (1 / 3)),
                    lw=1,
                    ls="--",
                    color="grey",
                    zorder=0,
                )
                axs[i].plot(
                    np.log10(ana_mhalf),
                    np.log10(
                        ((ana_mhalf / tenthousandth_max_rho) * (0.75 / np.pi))
                        ** (1 / 3)
                    ),
                    lw=1,
                    ls="--",
                    color="grey",
                    zorder=0,
                )
                # convert back to log log values for scatter
                f7_x = np.log10(f7_x)
                f7_y = np.log10(f7_y)
                axs[i].text(
                    0.42,
                    0.34,
                    r"$\mathrm{\rho_{h}^0 \sim5.4\times10^{4} M_{\odot} pc^{-3}}$",
                    rotation=23,
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=axs[i].transAxes,
                    fontsize=6,
                )

                axs[i].text(
                    0.80,
                    0.50,
                    r"$\mathrm{0.1\rho_{h}^0}$",
                    rotation=23,
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=axs[i].transAxes,
                    fontsize=6,
                )

                axs[i].text(
                    0.63,
                    0.98,
                    r"$\mathrm{10^{-4}\rho_{h}^0 }$",
                    rotation=23,
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=axs[i].transAxes,
                    fontsize=6,
                )
                # leg.get_frame().set_edgecolor("w")
                axs[i].set_xlim(f7_x.min() - 0.2, f7_x.max() + 0.2)
                axs[i].set_ylim(f7_y.min() - 0.1, f7_y.max() + 0.1)

            if i == 4 or i == 6:
                axs[i].axvspan(
                    np.log10(0.001),
                    np.log10(0.1),
                    alpha=0.3,
                    color="grey",
                    edgecolor=None,
                    lw=0,
                )
            if i == 1 or i == 5 or i == 7:
                axs[i].axhspan(
                    np.log10(0.001),
                    np.log10(0.1),
                    alpha=0.3,
                    color="grey",
                    edgecolor=None,
                    lw=0,
                )

            f7_scatter = axs[i].scatter(
                f7_x,
                f7_y,
                c=f7_bes[f7_mask],
                # s=f7_half_radii,
                alpha=0.8,
                edgecolors="k",
                marker="o",
                cmap=cmap,
                linewidths=0.25,
                s=15,
            )
            f3_scatter = axs[i].scatter(
                f3_x,
                f3_y,
                c=f3_bes[f3_mask],
                # s=f3_half_radii,
                alpha=0.8,
                edgecolors="k",
                marker="P",
                linewidths=0.25,
                s=15,
                cmap=cmap,
            )
            # plot index of not to fit
            dont_fit = [1, 2]
            # fit them
            if i not in dont_fit:

                f7_params = stats.linregress(x=f7_x, y=f7_y)
                f3_params = stats.linregress(x=f3_x, y=f3_y)

                f7_slope = f7_params[0]
                f7_slope_err = f7_params[4]
                f7_intercept = f7_params[1]
                f7_intercept_err = f7_params.intercept_stderr

                f3_slope = f3_params[0]
                f3_slope_err = f3_params[4]
                f3_intercept = f3_params[1]
                f3_intercept_err = f3_params.intercept_stderr

                # f7_params, f7_pcov = curve_fit(f=lin_model, xdata=f7_x, ydata=f7_y)
                # f3_params, f3_pcov = curve_fit(f=lin_model, xdata=f3_x, ydata=f3_y)

                x_extremas = np.array([f3_x.min(), f3_x.max(), f3_x.max(), f7_x.max()])
                y_extremas = np.array([f3_y.min(), f3_y.max(), f3_y.max(), f7_y.max()])

                theory_x = np.linspace(
                    x_extremas.min() - 0.5, x_extremas.max() + 0.5, 100
                )

                f7_upper_conf, f7_lower_conf, f7_x = confband(
                    f7_x, f7_y, f7_slope, f7_intercept
                )

                f3_upper_conf, f3_lower_conf, f3_x = confband(
                    f3_x, f3_y, f3_slope, f3_intercept
                )

                axs[i].fill_between(
                    f3_x,
                    f3_lower_conf,
                    f3_upper_conf,
                    color="grey",
                    alpha=0.4,
                    lw=0,
                    zorder=0,
                )

                axs[i].fill_between(
                    f7_x,
                    f7_lower_conf,
                    f7_upper_conf,
                    color="grey",
                    alpha=0.4,
                    lw=0,
                    zorder=0,
                )

                axs[i].plot(
                    theory_x,
                    lin_model(theory_x, f7_params[0], f7_params[1]),
                    lw=1,
                    color="grey",
                    label="$\mathrm{{{:.2f} \pm {:.2f}}}$, "
                    # "\n"
                    "$\mathrm{{{:.2f} \pm {:.2f}}}$".format(
                        f7_params[0],
                        f7_params[4],
                        f7_params[1],
                        f7_params.intercept_stderr,
                    ),
                )
                axs[i].plot(
                    theory_x,
                    lin_model(theory_x, f3_params[0], f3_params[1]),
                    lw=1,
                    ls="--",
                    color="grey",
                    label="$\mathrm{{{:.2f} \pm {:.2f}}}$, "
                    # "\n"
                    "$\mathrm{{{:.2f} \pm {:.2f} }}$".format(
                        f3_params[0],
                        f3_params[4],
                        f3_params[1],
                        f3_params.intercept_stderr,
                    ),
                )

                # best fit variations
                # f3_var1 = lin_model(
                #     theory_x,
                #     f3_params[0] - f3_params[4],
                #     f3_params[1] - f3_params.intercept_stderr,
                # )
                # f3_var2 = lin_model(
                #     theory_x,
                #     f3_params[0] + f3_params[4],
                #     f3_params[1] + f3_params.intercept_stderr,
                # )
                # f3_var3 = lin_model(
                #     theory_x,
                #     f3_params[0] + f3_params[4],
                #     f3_params[1] - f3_params.intercept_stderr,
                # )
                # f3_var4 = lin_model(
                #     theory_x,
                #     f3_params[0] - f3_params[4],
                #     f3_params[1] + f3_params.intercept_stderr,
                # )

                # axs[i].fill_between(
                #     theory_x,
                #     f3_var1,
                #     f3_var2,
                #     lw=0,
                #     # ls="-.",
                #     color=fs35_color,
                #     alpha=0.5,
                #     # edgecolor="none",
                # )

                # axs[i].fill_between(
                #     theory_x,
                #     f3_var3,
                #     f3_var4,
                #     lw=0,
                #     # ls="-.",
                #     color=fs35_color,
                #     alpha=0.5,
                #     # edgecolor="none",
                # )

                # axs[i].fill_between(
                #     theory_x,
                #     lin_model(theory_x, *(f3_params - np.sqrt(np.diag(f3_pcov)))),
                #     lin_model(theory_x, *(f3_params + np.sqrt(np.diag(f3_pcov)))),
                #     lw=2,
                #     # ls="-.",
                #     color=fs35_color,
                #     alpha=0.5,
                #     edgecolor="none",
                # )

                # axs[i].fill_between(
                #     theory_x,
                #     lin_model(theory_x, *(f7_params - np.sqrt(np.diag(f7_pcov)))),
                #     lin_model(theory_x, *(f7_params + np.sqrt(np.diag(f7_pcov)))),
                #     lw=2,
                #     # ls="-.",
                #     color=fs70_color,
                #     alpha=0.5,
                #     edgecolor="none",
                # )

                # axs[i].fill_between(
                #     theory_x,
                #     lin_model(theory_x, *(f3_params - np.sqrt(np.diag(f3_pcov)))),
                #     lin_model(theory_x, *(f3_params + np.sqrt(np.diag(f3_pcov)))),
                #     lw=2,
                #     # ls="-.",
                #     color=fs35_color,
                #     alpha=0.5,
                #     edgecolor="none",
                # )

                # tweak some limits

                axs[i].set_xlim(theory_x.min(), theory_x.max())
                axs[i].set_ylim(y_extremas.min() - 1, y_extremas.max() + 0.1)

                # fit parameter legend
                fit = axs[i].legend(
                    # title="$\mathrm{\log_{10}(slope,\:intercept)}$",
                    ncol=1,
                    fontsize=6,
                    title_fontsize=7,
                    loc="lower center",
                )
                fit.get_frame().set_edgecolor("grey")
                fit.get_frame().set_alpha(0.2)

            # if i == 1:
            #     axs[i].set_ylim(-1, 1.1)

            axs[i].set_xlabel(x_labels[i], labelpad=0)
            axs[i].set_ylabel(y_labels[i], labelpad=0)

            # supblot letter label
            axs[i].text(
                0.05,
                0.95,
                subplt_lbl[i],
                horizontalalignment="left",
                verticalalignment="top",
                transform=axs[i].transAxes,
                fontsize=8,
            )
            axs[i].tick_params(axis="both", direction="in", which="both")

            # color bars

        f70 = mlines.Line2D(
            [],
            [],
            color="grey",
            marker="o",
            ls="-",
            label=r"$f_{*} = 0.70$",
            alpha=0.8,
            markeredgewidth=0,
        )
        f35 = mlines.Line2D(
            [],
            [],
            color="grey",
            marker="P",
            ls="--",
            label=r"$f_{*} = 0.35$",
            alpha=0.8,
            markeredgecolor="none",
        )
        sfe_legend = fig.legend(
            loc=(0.063, 0.55),
            title=r"$\mathrm{{t = {:.0f} \: Myr, \: z = {:.2f}}}$".format(
                f3_t_myr, z_from_t_myr(f3_t_myr)
            ),
            # loc="upper left",
            title_fontsize=10,
            fontsize=10,
            handles=[f70, f35],
            # bbox_to_anchor=(0.315, 0.76),
        )
        sfe_legend.get_frame().set_edgecolor("k")
        sfe_legend.get_frame().set_boxstyle("Square")
        # axs[i].add_artist(sfe_legend)
        cbar_ax = fig.add_axes([0.1, 0.75, 0.17, 0.05])
        cbar = fig.colorbar(
            f3_scatter,
            cax=cbar_ax,
            pad=0,
            orientation="horizontal",
        )

        cbar.set_alpha(0.8)
        cbar_ax.set_title(label="$\mathrm{t_{formation} \: (Myr)}$", fontsize=12)

        cbar.ax.tick_params(axis="x", direction="in", which="both", labelsize=7)

plt.savefig(
    os.path.expanduser(
        ("~/g_drive/Research/AstrophysicsSimulation/sci_plots/final/" "gc_scatter.png")
    ),
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.05,
)
