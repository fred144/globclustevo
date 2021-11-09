import yt
import warnings
import numpy as np
from yt.funcs import mylog 
from macros import code_age_to_yr
#reduces some of the outputs when reading in yt data
mylog.setLevel(40) 
warnings.simplefilter(action = "ignore", category = RuntimeWarning)


        
def clump_filters(ds): 
    
    def clump1(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages, 0), relative_ages >= 0
            )
        return filter
    yt.add_particle_filter("clump1",function=clump1,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump1") 
    
    def clump2(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages, 18.52773269 ), relative_ages >= 0
            )

        return filter
    yt.add_particle_filter("clump2",function=clump2,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump2") 
    
    def clump3(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages, 35.60895952), relative_ages >= 0
            )
        
        return filter
    yt.add_particle_filter("clump3",function=clump3,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump3") 

    def clump4(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages, 68.6774688), relative_ages >= 0
            )        
        return filter
    yt.add_particle_filter("clump4",function=clump4,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump4") 

    def clump5(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages, 68.96389576), relative_ages >= 0
            )  
        return filter
    yt.add_particle_filter("clump5",function=clump5,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump5") 

    def clump6(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)                                                            
        filter = np.logical_and(
            np.isclose(relative_ages, 69.09889334 ), relative_ages >= 0
            )         
        return filter
    yt.add_particle_filter("clump6",function=clump6,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump6") 

    def clump7(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages, 69.34419314), relative_ages >= 0
            )    
        return filter
    yt.add_particle_filter("clump7",function=clump7,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump7") 

    def clump8(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages, 70.17392563), relative_ages >= 0
            ) 
        return filter
    yt.add_particle_filter("clump8",function=clump8,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump8") 

    def clump9(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages, 70.19719182), relative_ages >= 0
            ) 
        return filter
    yt.add_particle_filter("clump9",function=clump9,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump9") 

    def clump10(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages, 70.2027757), relative_ages >= 0
            )         
        return filter
    yt.add_particle_filter("clump10",function=clump10,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump10") 
    
    def clump11(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages, 70.30222356), relative_ages >= 0
            )    
        return filter
    yt.add_particle_filter("clump11",function=clump11,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump11") 
    
    def clump12(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages, 70.41787201), relative_ages >= 0
            )                                                                         
        return filter
    yt.add_particle_filter("clump12",function=clump12,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump12") 
    
    def clump13(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages, 70.79423501), relative_ages >= 0
            )  
        return filter
    yt.add_particle_filter("clump13",function=clump13,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump13") 
        
    def clump14(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages, 88.28128775), relative_ages >= 0
            )  
        return filter
    yt.add_particle_filter("clump14",function=clump14,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump14") 
    
    def clump15(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages, 89.38194823), relative_ages >= 0
            )  
        return filter
    yt.add_particle_filter("clump15",function=clump15,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump15") 
        
    def clump16(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages,  90.51360277), relative_ages >= 0
            )  
        return filter
    yt.add_particle_filter("clump16",function=clump16,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump16") 

    def clump17(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages,  95.28010249), relative_ages >= 0
            )  
        return filter
    yt.add_particle_filter(
        "clump17",function=clump17,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump17") 
    

    def clump18(pfilter, data):
        h = ds.hubble_constant
        code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
        relative_ages = code_age_to_yr(all_star_ages=code_ages, 
                                        hubble_const=h,
                                        unique=False)
        filter = np.logical_and(
            np.isclose(relative_ages,  98.17317232), relative_ages >= 0
            )  
        return filter
    yt.add_particle_filter(
        "clump18",function=clump17,filtered_type="star",requires=["particle_birth_epoch"])
    ds.add_particle_filter("clump18")
 
     

# def clump_tracker(ds, birth_epochs, width): 
#     print (birth_epochs)
    
#     for clump_num, be in enumerate(list(birth_epochs)):
        
#         clump_name = 'clump_' + str(clump_num)
        
#         @yt.particle_filter(requires=['particle_birth_epoch'], filtered_type='star')
#         def clump(pfilter, data):
#             h = ds.hubble_constant
#             code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
#             relative_ages = code_age_to_yr(all_star_ages=code_ages, 
#                                             hubble_const=h,
#                                             unique=False)
#             #print(relative_ages)
#             print(clump_num)
#             print(be)
#             filter = relative_ages == be
#             #filter = 
#             return filter
#         # yt.add_particle_filter( 
#         #     "clump1", function = clump1, filtered_type="star", requires=["particle_birth_epoch"]
#         #     )
#         ds.add_particle_filter('clump') 
#         # 
#         # plot_object.annotate_particles( 
#         #     width=width, ptype='clump', p_size=10.0, marker='.', col=np.random.rand(3,1).T  
#         #     ) 
#         del clump
        
# def clump_tracker(ds, birth_epochs, width): 
#     birth_epochs = np.array(birth_epochs)
    
#     for clump_num, be in enumerate(birth_epochs, start=1):
        
#         clump_name = 'clump_' + str(clump_num)
#         #print(be)
#        # @yt.particle_filter(requires=['particle_birth_epoch'], filtered_type='star')
       
#         def clump(pfilter, data):
#             h = ds.hubble_constant
            
#             code_ages = data[pfilter.filtered_type, 'particle_birth_epoch'] 
            
#             relative_ages = code_age_to_yr(all_star_ages=code_ages, 
#                                            hubble_const=h,
#                                            unique=False)
#             test = birth_epochs[clump_num-1]
#             print(test)
#             filter = relative_ages == test 
#             return filter
        
        
        
#         namespace['clump_%s'%str(clump_num)] = functools.partial(clump)
        
#         yt.add_particle_filter(clump_name, 
#                                 function=namespace['clump_%s'%str(clump_num)], 
#                                 filtered_type='star', 
#                                 requires=['particle_birth_epoch'],
#                                 )
#         ds.add_particle_filter(clump_name)
   
if __name__ == '__main__': 
    clump_filters()