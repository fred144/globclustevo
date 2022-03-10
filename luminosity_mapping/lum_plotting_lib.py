import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from pytreegrav import Accel, Potential
from skimage.feature import peak_local_max


def star_luminosity_plot(
        proj_width,
        star_positions,
        scaled_stellar_lums,
        time,
        snapshot_num,
        pi_multiple,
        plt_bins=2000,
        lum_scale=('static', 3e+32, 3e+36), 
        get_ctr=(True, 'potential', 0.01, False),
        num_ctr=100, 
        ctr_dist_thresh=2, 
        ctr_rel_thresh=.1,
        masses=None,
        sfc_positions=None,
        psc_positions=None,
        ):
    """
    Main single panel movie for luminosities of popII stars. 
    Currently makes projection plot along projected on an axis. 
    Can return and/or annotate local maxima corresponding to centers of 
    clusters.
    
    Parameters
    ----------
    proj_width
        width of path in pc
    star_positions
       postions of stars, (x,y)
    scaled_stellar_lums
        scaled star luminosities
    time
        current time of snapshot in myr
    snapshot_num
        snapshot number for simulation
    pi_multiple
        pi multiple of the rotation matrix, can be 0
    plt_bins
        number of bins along the axis of the luminosity map
    lum_scale: tuple (log_scale_type, min, max
        log_scale_type - static or dynamic
        
        min - minimum value color scale
        
        max- maximum value color scale
        
    get_ctr : tuple  (get_ctr, method, resolution, overplot)
        get_ctr - gets centers of GCs 
        
        method - either using raw "counts" of star in a bin or grav "potential"
        
        resolution - produces bins for centering using width/resolution
        
        overplot - True or false, overplots found centers
        
    Returns
    -------
    x_peak
        2d coordinates of centers
    y_peak
        2d coordinates of centers
    labels
        number labels for each GC

    """
   
    # 2d histogram using luminosities
    lums, xedges, yedges = np.histogram2d(
        star_positions[:,0],
        star_positions[:,1],
        bins=plt_bins,
        weights=scaled_stellar_lums,
        normed=False,
        range= [
            [-proj_width/2, proj_width/2], 
            [-proj_width/2, proj_width/2]
            ]
        )
    # need to transpose for some reason
    lums = lums.T
# =============================================================================
#              get GC centers based on potential or density counts
# =============================================================================
    if get_ctr is not None:
        print("> calculating centers")
        if  get_ctr[1] == 'counts':
            print('> finding peaks using star density counts ')
            pc_accuracy = get_ctr[2]
            centring_bins = int(proj_width/pc_accuracy)
            
            center_threshold_pixels = ctr_dist_thresh/pc_accuracy 
            
            # find peaks, returns indeces in the matrix
            
            # 2d histogram using raw counts
            counts, _, _ = np.histogram2d(
                star_positions[:,0],
                star_positions[:,1],
                bins=centring_bins,
                normed=False,
                range= [
                    [-proj_width/2, proj_width/2], 
                    [-proj_width/2, proj_width/2]
                    ]
                )
            counts = counts.T 
            # get bin centers 
            x_ctr = 0.5*(xedges[1:] + xedges[:-1])
            y_ctr = 0.5*(yedges[1:] + yedges[:-1])
            
            peaks = peak_local_max(
                counts, 
                threshold_rel=ctr_rel_thresh, 
                min_distance=center_threshold_pixels
                ) 
            
            col_idx = peaks[:,1]
            row_idx = peaks[:,0]
            # translate indeces to coordinates
            x_peak = x_ctr[col_idx] 
            y_peak = y_ctr[row_idx]
            # optionally annotate with found centers

        
        elif get_ctr[1] == 'potential':
            #print('> finding peaks using grav potentials')
            pc_accuracy = get_ctr[2]                        # pc/pixel
            centring_bins = int(proj_width/pc_accuracy)     # no. pixels ctr
            pc_per_pixel = proj_width/centring_bins         # might be rounded
            # calculate the minimum distance between centers in pc
            center_threshold_pixels = int(ctr_dist_thresh/pc_accuracy)
            print( 
                '> finding peaks using grav potentials with precision',
                pc_per_pixel, 
                'pc/pixel'
                )
            print('> pixels along each dimension', centring_bins)
            print( 
                '> center distance threshold', 
                center_threshold_pixels,
                'pixels'
                )

            phi = Potential(pos=star_positions, m=masses, method='bruteforce')
            
            grav, xedges, yedges = np.histogram2d(
                star_positions[:,0],
                star_positions[:,1],
                bins=centring_bins,
                weights=np.abs(phi),
                normed=False,
                range= [
                    [-proj_width/2, proj_width/2], 
                    [-proj_width/2, proj_width/2]
                    ]
            )
            grav = grav.T
            x_ctr = 0.5*(xedges[1:] + xedges[:-1])
            y_ctr = 0.5*(yedges[1:] + yedges[:-1])
            
            peaks = peak_local_max(
                grav, 
                num_peaks=num_ctr,
                min_distance=center_threshold_pixels, 
                threshold_rel=ctr_rel_thresh
                ) 
            
            col_idx =  peaks[:,1]
            row_idx = peaks[:,0] 
            
            x_peak = x_ctr[col_idx] 
            y_peak = y_ctr[row_idx]
            
            gc_labels = np.arange(1, x_peak.size+1, 1)
            
            print("> found Centers for", x_peak.size)
            
        else:
            print("!centering method not supported!")
            exit
             
    else:
        pass
    

    fig = plt.figure(figsize=(14,12),dpi=300)
    ax = fig.add_subplot(111, facecolor=cm.inferno(0))
    
    # color maps
    if lum_scale[0] == 'static':
        rectbin = plt.imshow(
                   lums,
                   cmap='inferno',
                   interpolation='gaussian',
                   origin='lower',
                   extent=[-proj_width/2,
                           proj_width/2,
                           -proj_width/2,
                           proj_width/2],
                   norm=LogNorm(vmin=lum_scale[1], vmax=lum_scale[2])
                   )
    elif lum_scale[0] == 'dynamic':
        rectbin = plt.imshow(
                   lums,
                   cmap='inferno',
                   interpolation='gaussian',
                   origin='lower',
                   extent=[-proj_width/2,
                           proj_width/2,
                           -proj_width/2,
                           proj_width/2],
                   norm=LogNorm()
                   )
    else:
        print("!not a valid color scale!")
        exit()
    
    #print('lum plot')

# =============================================================================
#                            Optionally plot test particles
# =============================================================================
    if sfc_positions is not None:
        sfc_x = sfc_positions[:,0]
        sfc_y = sfc_positions[:,1]
        sfc_tags = sfc_positions[:,3]

        plt.scatter(
            sfc_x,
            sfc_y,
            marker='.',
            s=5
            )

        for i, txt in enumerate(sfc_tags):
            plt.annotate(int(txt), (sfc_x[i], sfc_y [i]), fontsize=7, ha='center')
        plt.xlim(-proj_width/2, proj_width/2)
        plt.ylim(-proj_width/2, proj_width/2)
    if psc_positions is not None:
        psc_x = psc_positions[:,0]
        psc_y = psc_positions[:,1]
        psc_tags = psc_positions[:,3]

        plt.scatter(
            psc_positions[:,0],
            psc_positions[:,1],
            marker='.',
            s=5,
            c='lime'
            )
        for i, txt in enumerate(psc_tags):
            plt.annotate(int(txt), (psc_x[i], psc_y [i]), fontsize=8, ha='center')

        plt.xlim(-proj_width/2, proj_width/2)
        plt.ylim(-proj_width/2, proj_width/2)
# =============================================================================
#                              plot centers of max values if true
# =============================================================================
    if get_ctr is not None: 
        # can't put in the same line since it will check the tuple regardless
        if get_ctr[3] is True:
            plt.scatter(x_peak, y_peak, color='green',marker='.',s=.5)
            plt.xlim(-proj_width/2, proj_width/2)
            plt.ylim(-proj_width/2, proj_width/2)
            
            # iterate over labels and label each scatter point
            for i, label in enumerate(gc_labels):
                plt.annotate(label, (x_peak[i], y_peak[i]), fontsize=4, ha='center')
# =============================================================================
#                            plot aesthetics
# =============================================================================
    
    plt_label = (r"$\lambda = 1500\;\AA$ Projected Monochromatic Luminosity" 
    r" $\left(erg\;s^{-1}\AA^{-1} \right)$")
    
    # add color bar to the bottom
    # fig.subplots_adjust(wspace=0, hspace=0, bottom=.1)
    # cbar_ax = fig.add_axes([.178, .090, 0.67, 0.010])
    # cbar = fig.colorbar(
    #              rectbin,
    #              cax=cbar_ax,
    #              orientation='horizontal',
    #              pad=0
    #             )
    # cbar.set_label(
    #     label=plt_label,
    #     size=12
    #     ) 
    
    # add color bar inside 
    fig.subplots_adjust(wspace=0, hspace=0, bottom=.1)
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.319, 0.835, 0.39, 0.008])
    cbar = fig.colorbar(
                  rectbin,
                  cax=cbar_ax,
                  orientation='horizontal',
                  pad=0
                )
    cbar_ax.xaxis.set_label_position('top')
    cbar.set_label(
        label=plt_label,
        labelpad=8,
        size=12
        )

    # annotate with time
    ax.text(
        -proj_width*0.375,
        -proj_width*0.45,
        't = %.2f Myr'%(time),
        size=12,
        ha='center',
        va='center',
        color='white')
    
    # add scale bar
    rect = patches.Rectangle(
            xy=(-proj_width*0.125, -proj_width*0.45),
            width=proj_width*0.25,
            height=proj_width*0.005,
            linewidth=0,
            edgecolor='white',
            facecolor='white')
    ax.add_patch(rect)
    ax.text(
        0,
        -proj_width*0.475,
        '{} pc'.format(int(proj_width/4)),
        size=12,
        ha='center',
        va='center',
        color='white'
        )
    #ax.set_axis_off()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.add_artist(ax.patch)
    ax.patch.set_zorder(-1)

    if get_ctr is not None:
        return x_peak, y_peak, gc_labels
    else:
        pass
    
