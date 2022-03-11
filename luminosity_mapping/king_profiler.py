import numpy as np
import os
import matplotlib.pyplot as plt
from lum_funcs import look_up_table, get_cluster 
from lum_plotting_lib import star_luminosity_plot
from scipy.optimize import curve_fit
import matplotlib as mpl
from scipy import stats 
import sys


plt.rcParams.update({'figure.max_open_warning': 0})
# mpl.rc('font', family='serif')
# mpl.rc('text', usetex=True) 
plt.style.use('dark_background')

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })
def chi_squared(theory, data, sigma, num_params):
    
    import scipy.stats as st
    
    chi2 = np.sum((theory-data)**2/sigma**2)
    dof = np.size(data) - num_params
    reduced_chi_2 = chi2 / dof
    p_value = st.chi2.sf(chi2, dof)
    
    return p_value, reduced_chi_2

def king_model(r, k, r_c, r_t):
    """
    https://articles.adsabs.harvard.edu/pdf/1962AJ.....67..471K
    """
    f = k *( (1 / np.sqrt(1 + (r / r_c)**2)) - (1 / np.sqrt(1 + (r_t /r_c )**2)) )**2
    return f

def modified_king_model(r, sigma_naught, r_c, alpha, bg):

    sigma = bg + (sigma_naught / (1 + (r/r_c)**alpha) )
    return sigma

def trunc_radius(sigma_0, r_c, alpha, sigma_bg):
    """
    set to 1.5bg =  bg + (peak)/( 1 + (r/r_c)^alpha)
    0.5bg = (peak)/( 1 + (r/r_c)^alpha)
    """
    #trunc_r = (r_c**alpha * ((sigma_0/(.5*sigma_bg)) - 1) )**(1/alpha)
    r_trunc = (r_c**alpha * ((sigma_0/ ((1.5-1)*sigma_bg ) - 1 ) ) )**(1/alpha)
    return r_trunc
    
def get_masses(x_coord, y_coord, masses, r_characteristic):
    """
    get core mass or any mass enclosed a characteristic radius
    """
    all_positions = np.vstack((x_coord, y_coord)).T
    distances = np.sqrt(np.sum(np.square(all_positions), axis=1))  
    mask = distances <= r_characteristic
    core_mass = np.sum(masses [mask])
    return core_mass
    


def unpack_pop_ii_data(
        path:str, 
        lum_scaling=1e-5, 
        lum_link='https://www.stsci.edu/science/starburst99/data/l1500_inst_e.dat'
    ): 
    """
    Parameters
    ----------
    path
        path to file
    lum_scaling
        scaling factor for luminosity, see stsci tables
    lum_link
        link to the lookup table

    Returns
    -------
        (x,y,z) positions of stars
        corresponding stellar luminosities
        masses in M_sun
        current time in Myr
    """
    
    pop_2_data = np.loadtxt(path)
    # birth_epochs = pop_2_data[:,0] *1e6
    ages = pop_2_data[:,1] *1e6             # convert to myr
    ages [ages < 1e6 ] = 1e6                # set minimum age
    t_myr = pop_2_data[0,6]
    masses = pop_2_data[:,5]                #msun
    
    # use look up table
    # current bottle neck 
    stellar_lums = look_up_table(
        stellar_ages=ages,
        table_link=lum_link,
        column_idx=1,
        log=True
        )
    
    scaled_stellar_lums = stellar_lums*lum_scaling 
    star_positions = pop_2_data[:,2:5]      # (x,y,z)

    
    return star_positions, scaled_stellar_lums, masses, ages, t_myr

def projected_surf_densities(
    x_coord, y_coord, lums, masses, radius, num_bins, log_bins=True, dr=None        
    ):
    
    starting_point = .04 #pc might have to tweak this. 
    
    # stack two-1d arrays
    all_positions = np.vstack((x_coord, y_coord)).T

    if log_bins is True:
        r = np.geomspace(starting_point, radius, num=num_bins, endpoint=True)
        #r_inner = np.geomspace(0, radius, num=num_bins, endpoint=False)
    else: 
        r = np.arange(0, radius+dr, dr)
    
    # assume that the cluster fed to this has already been translated to (0,0)
    #ctr_at=np.array([0,0])
    
    distances = np.sqrt(np.sum(np.square(all_positions), axis=1))    

    mass_per_bin, bin_edges = np.histogram(distances, bins=r, weights=masses)
    lum_per_bin, _ = np.histogram(distances, bins=r, weights=lums)
    count_per_bin, _ = np.histogram(distances, bins=r)
    #print(mass_per_bin.size, lum_per_bin.size, count_per_bin.size)
    # mask zero count bins
    mask = count_per_bin > 0
    count_per_bin = count_per_bin[mask]
    mass_per_bin = mass_per_bin[mask]
    lum_per_bin = lum_per_bin[mask]
    
    # getting bin properties
    right_edges = bin_edges[1:] 
    left_edges = bin_edges[:-1]
    bin_ctrs = 0.5*( left_edges + right_edges)[mask]
    ring_areas = np.pi * (right_edges**2 - left_edges**2)[mask]
    
    
    # calculate densities
    surf_mass_density = mass_per_bin / ring_areas
    surf_lum_density = lum_per_bin / ring_areas
    surf_number_density =  count_per_bin / ring_areas
     
    # characterize what the typical mass is for a bin 
    avg_star_masses = mass_per_bin / count_per_bin
    # piosson error in the surface density
    err_surf_mass_density = np.sqrt(count_per_bin)*(avg_star_masses/ring_areas)
    # sum the bins to get total mass out to the specified cluster radii
    total_clust_m = np.sum(mass_per_bin)
    
    return bin_ctrs, surf_mass_density, err_surf_mass_density, total_clust_m
    
        
def king_profiler(
        star_pos, 
        lums, 
        masses, 
        ages, 
        gc_ctr, 
        gc_rad, 
        gc_label, 
        bins=30
        ):
    """
    depends on projected_surf_densities
    """
    # get a cluster given a center --> (0,0,0) and spherically mask around it
    # returns positons (recentered), masses, luminosites, and ages of each star
    if gc_ctr.size == 2:
        clust_x, clust_y, clust_lums, clust_masses, clust_ages = get_cluster(
            xpos=star_pos[:,0],
            ypos=star_pos[:,1],
            zpos=None,
            ctr_at=gc_ctr,
            masses=masses,
            ages=ages,
            cluster_radius=gc_rad,
            lums=lums,
            trns_coord = True
            )
    else:
        # 3d case
        clust_x, clust_y, clust_z, clust_lums, clust_masses, clust_ages = get_cluster(
            xpos=star_pos[:,0],
            ypos=star_pos[:,1],
            zpos=star_pos[:,2],
            ctr_at=gc_ctr,
            masses=masses,
            ages=ages,
            cluster_radius=gc_rad,
            lums=lums,
            trns_coord = True
            )

    # given an isolated cluster, find projected density quantities 
    # as a function of radius with log bins or linear bins
    r, rho, err, tot_m = projected_surf_densities(
        x_coord=clust_x, 
        y_coord=clust_y, 
        lums=clust_lums, 
        masses=clust_masses, 
        radius=gc_rad, 
        num_bins=bins, 
        log_bins=True, 
        dr=None        
        )
    
    try:
        # try to do the fit
        fit_params, cov_matrix = curve_fit(
            f=modified_king_model,
            xdata=r,
            ydata=rho,
            sigma=err,
            absolute_sigma=True,
            p0 = [1e4, 0.2, 2, 10],
            bounds=([0,0,0,0], [np.inf,np.inf,100,np.inf]),
            )
        #r, sigma_naught, r_c, alpha, bg
        fit_sigma = np.sqrt(np.diag(cov_matrix))
        # print(fit_params)
        # print(fit_sigma)
    
        # calculate theoretical best fit
        theory_rho = modified_king_model(r,*fit_params)
        
# ===========================calc derived quantities===========================
        fit_sigma_naught = fit_params[0]
        fit_r_c = fit_params[1]
        fit_alpha = fit_params[2]
        fit_sigma_bg = fit_params[3]
        truncation_radius = trunc_radius(
            r_c=fit_r_c, 
            alpha=fit_alpha, 
            sigma_0=fit_sigma_naught,
            sigma_bg=fit_sigma_bg
            )
        core_mass = get_masses(clust_x, clust_y, clust_masses, fit_r_c)
        gc_char_age = float(stats.mode(clust_ages)[0]/1e6) 
# =============================================================================

        # quantify goodness of fit
        p_value, reduced_chi_2 = chi_squared(theory=theory_rho, 
                    data=rho, 
                    sigma=err,
                    num_params=4
                    )
        if truncation_radius <= 20:
            plot_label = (
                r'$R_{{trunc}} = {:.2f} \: pc$'
                '\n'
                r'$R_{{core}} = {:.2f} \: pc$'
                '\n'
                r'$\alpha = {:.2f} $'
                '\n'
                r'$\Sigma_0 = {:.2e} $'
                '\n'
                r'$\Sigma_{{bg}} = {:.2f} $'
                '\n'
                r'$M_{{r_c}} = {:.2e} \: M_{{\odot}}$'
                ).format(
                    truncation_radius,
                    fit_r_c, 
                    fit_alpha, 
                    fit_sigma_naught,
                    fit_sigma_bg,
                    core_mass
                        )
        else: 
            plot_label = (
                r'$R_{{trunc}} > 20 $ pc'
                '\n'
                r'$R_{{core}} = {:.2f}$ pc'
                '\n'
                r'$\alpha = {:.2f} $'
                '\n'
                r'$\Sigma_0 = {:.2e} $'
                '\n'
                r'$\Sigma_{{bg}} = {:.2f} $'
                '\n'
                r'$M_{{r_c}} = {:.2e} \: M_{{\odot}}$'
                ).format(
                    fit_r_c, 
                    fit_alpha, 
                    fit_sigma_naught,
                    fit_sigma_bg,
                    core_mass
                        )
            
        # plot the fit if it is good
        if fit_alpha < 5 and core_mass > 1:
            plt.figure(figsize = (8,8), dpi=200)
            plt.errorbar(
                r,
                rho,
                yerr=err,
                fmt='o',
                capsize=5,
                capthick=3,
                elinewidth=3,
                label= (
                    r'$M_{{total}}= {:.2e} \: M_{{\odot}}$'
                    '\n'
                    r'$t_{{age}}= {:.2f}$ Myr'
                    ).format(tot_m, gc_char_age) 
                )
            plt.plot(r,theory_rho,linewidth=4,label=plot_label)
            #plt.axvline(fit_r_c)
            #plt.text(fit_r_c, 0,r'$R_core$',rotation=90)
            plt.title(r"GC # {:.0f}".format(gc_label), fontsize=16)
            plt.ylabel(r'Surface Mass Density ($M_{\odot} \; pc^{-2}$)', fontsize=16)
            plt.xlabel(r'Radius ($pc$)', fontsize=16)
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(visible=True, which='both', axis='y', ls='--')
            plt.legend(fontsize=16)
            return r, rho, err, tot_m, fit_r_c, core_mass, truncation_radius, gc_char_age 
        else: 
            print(r"> bad alpha for GC #{:.0f}".format(gc_label))
            return -1, -1, -1, -1, -1, -1, -1, -1
    

    except:
        # if it can't fit it
        print(r"> can't fit GC #{:.0f}".format(gc_label))
        return -1, -1, -1, -1, -1, -1, -1, -1
    
#%%

def run_profiler (file_name, proj_width, gc_radii, lum_map_bins): 
    print("# read in:", file_name)
    
    time_str = file_name[23:29].replace('_','.') #in myr
    time = float(time_str)
    snapshot_num = int(file_name[17:22])

    save_name = './gc_profiles/snapshot_{}_t_{}/'.format(
        str(snapshot_num).zfill(4),
        str(time).ljust(6, '0').replace('.','_'),
        )
    
    # put all verbose output into a text file
    sys.stdout = open(save_name + 'log.txt','wt') 
    print("> snapshot time", time, "Myr")
    print("> snapshot number", snapshot_num)
    print("> uniform radius of", gc_radii, "pc")
    
    star_positions, scaled_stellar_lums, masses, ages, t_myr= unpack_pop_ii_data(
        file_name
        )
    # get center x and y coordinates
    peak_x, peak_y, gc_labels = star_luminosity_plot(
        proj_width=proj_width,
        star_positions=star_positions,
        scaled_stellar_lums=scaled_stellar_lums,
        time=t_myr,
        snapshot_num=snapshot_num,
        pi_multiple=0,
        plt_bins=lum_map_bins,
        lum_scale=('dynamic',0,0),
        get_ctr=(True, 'potential', 0.04, True),
        masses=masses,
        ) 

    if not os.path.exists(save_name):
        print("# Creating new sequence directory",save_name )
        os.makedirs(save_name )
    
    plt.savefig(
        save_name+'annotated_gcs.png',
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.05
        )
    # loop over the centers, make profiles, and get data on a cluster basis.
    gc_tot_masses = []
    gc_r_core = []
    gc_m_core = []
    gc_r_trunc = []
    gc_char_age = []
    
    test_ctrs = np.array([peak_x, peak_y]).T
    # iterate over x,y maximas and plot
    for ctr,label in zip(test_ctrs,gc_labels):
        # print(ctr)
        #print(label)
        _, _, _, m_tot, r_c, m_r_c, r_trunc, char_age = king_profiler(
                    star_pos=star_positions,
                    lums=scaled_stellar_lums, 
                    masses=masses, 
                    ages=ages,
                    gc_ctr=ctr, 
                    gc_rad=gc_radii,
                    gc_label=label,
                    bins=25
                    )
        #print(char_age )
        gc_tot_masses.append(m_tot)
        gc_r_core.append(r_c)
        gc_m_core.append(m_r_c)
        gc_r_trunc.append(r_trunc)
        gc_char_age.append(char_age)
        if m_tot > 0:
            plt.savefig(
                save_name+'gc_{}'.format(label),
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.05
                )
        
    # turn into arrays so we can index them and then clean up
    gc_tot_masses = np.array(gc_tot_masses) 
    gc_r_core = np.array(gc_r_core)
    gc_m_core = np.array(gc_m_core)
    gc_r_trunc = np.array(gc_r_trunc)
    gc_char_age = np.array(gc_char_age)
    
    mask = gc_tot_masses > 0
    gc_tot_masses = gc_tot_masses[mask] 
    gc_r_core = gc_r_core[mask]
    gc_m_core = gc_m_core[mask]
    gc_r_trunc = gc_r_trunc[mask]
    gc_char_age = gc_char_age[mask]

    print("> found",gc_char_age.size,"good profiles" )
    
    sys.stdout.close()
    
    return gc_tot_masses, gc_r_core, gc_m_core, gc_r_trunc, gc_char_age
    
masses, core_radii, core_masses, r_trunc, ages  = run_profiler(
    "./pop_2_data/pos_00694_523_92_myr.txt", 400, 10, 4000,
    
    )
#%%

plt.figure(figsize = (8,8), dpi=200)
plt.hist(core_masses, bins=np.geomspace(core_masses.min(), core_masses.max(),10), histtype='step',  fill=False) 
plt.xscale('log')

#%%https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html

colors =  np.random.uniform(size=masses.size)
biggest_gc = np.max(core_radii)
# map to differnt sizes for better plotting
core_radii_per_size = (500*core_radii) / biggest_gc

fig, ax = plt.subplots(figsize = (8,8), dpi=200) 

scatter = ax.scatter(ages, 
                     masses, 
                     c=colors, 
                     s=core_radii_per_size,
                     cmap='Set3', 
                     alpha=0.6,
                     linewidths=2
                     )

# remap to actual sizes for legend
legend_properties = dict(prop='sizes', num=4, color='white', fmt=' {x:.2f}',
          func=lambda r: (r*biggest_gc)/500 )

legend = ax.legend(
    *scatter.legend_elements(**legend_properties),
    loc='upper right', 
    title='$R_{core}$ (pc)',
    title_fontsize=16, 
    fontsize=14,
    
    )
ax.set_yscale('log')
plt.ylabel(r'Total GC Mass ($M_{\odot}$)', fontsize=16)
plt.xlabel(r'Age (Myr)', fontsize=16) 
plt.show()