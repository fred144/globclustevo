import warnings
import os
import yt
import numpy as np
#import scipy as sc
from yt.funcs import mylog
from clump_filters import clump_filters

mylog.setLevel(40)
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

def succ_distance (current, previous):
    dist = np.linalg.norm(np.array(current) - np.array(previous)) 
    return dist 
def code_age_to_yr (all_star_ages, hubble_const, unique = True): 
    r"""
    Returns an array with unique birth epochs in Myr given  
    raw_birth_epochs = ad['star', 'particle_birth_epoch'] 
    AND 
    hubble = ds.hubble_constant  
    Youngest is 0 Myr, all others are relative to the youngest.
    """
    cgs_yr = 3.1556926e7          # 1yr (in s)
    cgs_pc = 3.08568e18           # pc (in cm)
    h_0 =  hubble_const*100 #hubble parameter (km/s/Mpc)
    h_0_invsec = h_0*1e5/(1e6*cgs_pc) #hubble constant h [km/s Mpc-1] -> [1/sec]
    h_0inv_yr = 1/h_0_invsec/cgs_yr   #1/h_0 [yr]
    
    if unique == True:
        #process to unique birth epochs only
        be_star_processed = np.array(sorted( list( set(all_star_ages.to_ndarray()) ) ) ) 
        star_age_myr = be_star_processed * h_0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
        #print(relative_ages)
    else: 
        all_stars  =  np.array(list( all_star_ages.to_ndarray()) )
        star_age_myr = all_stars * h_0inv_yr/1e6
        relative_ages = star_age_myr - star_age_myr.min()
    #print(relative_ages)
    return relative_ages

def clump_tracker(ds, birth_epochs, plot_object, width): 
    print (birth_epochs)
    for clump_num, be in enumerate(list(birth_epochs)):
        
        #clump_name = 'clump_' + str(clump_num)
        
        @yt.particle_filter(requires=['particle_birth_epoch'], filtered_type='star')
        def clump(pfilter, data):
            h = ds.hubble_constant
            code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
            relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                           hubble_const=h,
                                           unique=False)
            #print(relative_ages)
            print(clump_num)
            print(be)
            filter = relative_ages == be
            #filter = 
            return filter
        # yt.add_particle_filter( 
        #     "clump1", function = clump1, filtered_type="star", requires=["particle_birth_epoch"]
        #     )
        ds.add_particle_filter('clump') 
        # 
        plot_object.annotate_particles( 
            width=width, ptype='clump', p_size=10.0, marker='.', col=np.random.rand(3,1).T  
            ) 
        del clump
        
        
        
#some colors https://www.rapidtables.com/web/color/RGB_Color.html
maroon = np.array([128,0,0])/255.; maroon = maroon.reshape(1,-1)
tomato = np.array([255,99,71])/255.; tomato = tomato.reshape(1,-1)
orange_red = np.array([255,69,0])/255.; orange_red = orange_red.reshape(1,-1)
dark_orange = np.array([255,140,0])/255.; dark_orange = dark_orange.reshape(1,-1)
dark_golden_rod =  np.array([184,134,11])/255.; dark_golden_rod = dark_golden_rod.reshape(1,-1)
olive =  np.array([128,128,0])/255.; olive = olive.reshape(1,-1)
lime  =  np.array([0,255,0])/255.; lime  = lime .reshape(1,-1)     
teal =  np.array([0,128,128])/255.; teal  = teal .reshape(1,-1)              
indigo = np.array([75,0,130])/255.; indigo  = indigo .reshape(1,-1)  
thistle = np.array([216,191,216])/255.; thistle  = thistle .reshape(1,-1)  
plum = np.array([221,160,221])/255.; plum  = plum .reshape(1,-1)  
violet = np.array([238,130,238])/255.; violet  = violet .reshape(1,-1)  
deep_pink = np.array([255,20,147])/255.; deep_pink  = deep_pink .reshape(1,-1) 
light_pink = np.array([255,182,193])/255.; light_pink   = light_pink  .reshape(1,-1) 
chocolate = np.array([210,105,30])/255.; chocolate    = chocolate   .reshape(1,-1) 
beige = np.array([245,245,220])/255.; beige= beige.reshape(1,-1) 
wheat = np.array([245,222,179])/255.; wheat= wheat.reshape(1,-1) 
saddle_brown = np.array([139,69,19])/255.; saddle_brown= saddle_brown.reshape(1,-1)
honeydew = np.array([240,255,240])/255.; honeydew = honeydew .reshape(1,-1)
ivory = np.array([255,255,240])/255.; ivory = ivory .reshape(1,-1)
azure = np.array([240,255,255])/255.; azure = azure .reshape(1,-1)
mintcream = np.array([245,255,250])/255.; mintcream = mintcream .reshape(1,-1)
      
#---------------------------------data directory/info file---------------------
#datadir = os.path.expanduser(
#    'G:/My Drive/Research/AstrophysicsSimulation/DesktopEnvironment/data_globular_cluster/refine') 

#datadir = os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_rerun') 
datadir = os.path.expanduser('/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/fs07_refine') 

# local save path 
#parent_folder = 'G:/My Drive/Research/AstrophysicsSimulation/' + \
#                'ClusterEnvironment/AstroSimulationResearch/cluster_evolution_fs07'
#sequence_folder = 'test_frames'
#---------------------------------save path---------------------
##### cluster save path ######
parent_folder = '/homes/fgarcia4/analysis/cluster_evolution_fs07/sequences/new_refine'
sequence_folder = 'z_proj_den_old_clump_tracking'
# make new folder
newpath = parent_folder + '/' + sequence_folder
if not os.path.exists(newpath):
    os.makedirs(newpath)
#plot params
sequence_title = 'Z Projected Density'
width = (690,'pc')
axis = 'z'
start_step = 163
end_step = 254
ctr_shift_thresh =  0.00060 #code length

max_density_coords = []

#generate colors test
#colors = np.zeros([13,3])
#for i in range (0,10):
#    #color = np.random.rand(3,1).T 
#    
#    color =  np.array([np.random.rand(),np.random.rand(),np.random.rand()])/255.
#    color= color.reshape(1,-1)
#    colors[i,:]= color
    
#---------------------------------MAIN LOOP-----------------------------------
for loop_num, output_num in enumerate(range(start_step, end_step + 1)) :
    print ("-")
    infofile = os.path.abspath (datadir + "/output_%05d/info_%05d.txt" % (output_num,output_num))
    print ("#reading in info file: %s" %infofile)  

    #cell fields
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
    
    print('reading fields...')
    
    ds = yt.load(infofile, fields=FIELDS, extra_particle_fields=EPF)
    clump_filters(ds)###################
    redshft = ds.current_redshift
    current_hubble = ds.hubble_constant  
    
    # extract all data fields
    ad = ds.all_data()
    
    # get SFC/PSC positions and other important fields
    pos_SFCs = ad['SFC', 'particle_position']
    pos_PSCs = ad['PSC', 'particle_position']
    be_star = ad['star', 'particle_birth_epoch']
    raw_birth_epochs = ad['star', 'particle_birth_epoch'] 
    
    # centering using max density 
    max_den = ad.argmax(('gas', 'density'))
    max_density_coord = yt.YTArray(max_den).in_units('code_length') 
    max_density_coord = np.array(max_density_coord)

    # particle clumps by age 
    unique_birth_epochs = code_age_to_yr(raw_birth_epochs, current_hubble)
    print('> clumps:', len(unique_birth_epochs))
    #keep center of plots relatively stable
    if loop_num == 0:
        max_density_coords.append(max_density_coord)
    
    distance = succ_distance(max_density_coord, max_density_coords[-1])
    
    print('> distance b/w current and previously used max density:', distance)
    
    if distance < ctr_shift_thresh: 
        p = yt.ProjectionPlot(ds, axis, "density", width = width, center = max_density_coord)
        
        # if the plot center migrates, annotate previous center
        p.annotate_marker(
            max_density_coords[-1],
            marker="x",
            coord_system="data",
            plot_args={"color": "lime", "s": 30},)

        # appened new center to list
        max_density_coords.append(max_density_coord)

        print('> plot centered at {}'. format(max_density_coord)) 
    else: 
        center = max_density_coords[-1]
        p = yt.ProjectionPlot(ds, axis, "density", width = width, center = max_density_coords[-1])
        print('> using old center at {}'. format(center)) 
        
        
    print('annotating', np.array(be_star).size, 'star particles')
    if pos_SFCs.size > 0:
#         first_sfc_center = pos_SFCs[0]  
#         p = yt.SlicePlot(ds, 'z', "density", width = width, center = first_sfc_center) #turn this on to track center of sfc
        p.annotate_particles(width = width,
                             ptype='SFC', 
                             p_size=100.0,
                             marker='x',col='b') 
    if pos_PSCs.size > 0: 
        p.annotate_particles(width = width,
                             ptype='PSC', 
                             p_size=100.0,
                             marker='x',col='k')
#         if pos_SFCs.size == 0: 
#             first_psc_center = pos_PSCs[0] 
#             p = yt.SlicePlot(ds, 'z', "density", width = width, center = first_psc_center) #turn this on to track center of psc
        
            
    #p.annotate_particles(width=width, ptype='star', p_size=10.0,marker='.',col='r')    
    #for clumpnum, color in enumerate(colors, start = 1):
    #    clumpname = 'clump'+str(clumpnum) 
     #   print(color)
     #   p.annotate_particles(width=width, ptype=clumpname, p_size=20.0, marker='.',col=color) 
    p.annotate_particles(width=width, ptype="clump1", p_size=10.0, marker='.',col='b') 
    p.annotate_particles(width=width, ptype="clump2", p_size=10.0, marker='.',col='g') 
    p.annotate_particles(width=width, ptype="clump3", p_size=10.0, marker='.',col='r') 
    p.annotate_particles(width=width, ptype="clump4", p_size=10.0, marker='.',col='c') 
    p.annotate_particles(width=width, ptype="clump5", p_size=10.0, marker='.',col='m') 
    p.annotate_particles(width=width, ptype="clump6", p_size=10.0, marker='.',col='y') 
    p.annotate_particles(width=width, ptype="clump7", p_size=10.0, marker='.',col='k') 
    p.annotate_particles(width=width, ptype="clump8", p_size=10.0, marker='.',col=maroon) 
    p.annotate_particles(width=width, ptype="clump9", p_size=10.0, marker='.',col=tomato) 
    p.annotate_particles(width=width, ptype="clump10", p_size=10.0, marker='.',col=orange_red) 
    p.annotate_particles(width=width, ptype="clump11", p_size=10.0, marker='.',col=dark_orange) 
    p.annotate_particles(width=width, ptype="clump12", p_size=10.0, marker='.',col=dark_golden_rod) 
    p.annotate_particles(width=width, ptype="clump13", p_size=10.0, marker='.',col=olive) 
#    clump_tracker(ds=ds, birth_epochs=unique_birth_epochs , plot_object=p, width=width)
    
    p.annotate_timestamp(corner='lower_left', 
                         time_format='t = {time:.2f} {units}', 
                         time_unit= 'Myr', 
                         redshift=True) 
                         #draw_inset_box=True)
    p.annotate_scale(corner='lower_right') #, draw_inset_box= True)
    p.set_cmap('density', 'magma')
    p.set_zlim('density', 0.01, .05)

    # plot_title = str( 
    #     "{} {}| Red = Pop II, Blue = SFC, Black = PSC ".format(sequence_title, output_num) ) 
    #p.annotate_title(plot_title)
    
    save_path = str ("{}/{}/output-{}-z-{}-{}.png".format(parent_folder,
                                                sequence_folder,
                                                str(output_num).zfill(5),
                                                str(np.around(redshft, 2)).replace('.', '_'),
                                                sequence_title.replace(' ','-'))
                      )
    p.save(save_path, mpl_kwargs=dict(dpi=200))
    print('#saved:', save_path)
##
##%%
#colors = np.zeros([500,3])
#
#for i in range (0,10):
#    color = np.random.rand(3,1).T 
#    colors[i,:]= color
#
##%%
#for i in range(0, np.size(colors)[0]):
#    print (colors[i,:])
#    
