import os 
import sys
import shutil

lustre = "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution" 
#lustre = "/home/fabg/cosm_test_data"
for path, subdirs, files in os.walk(lustre):
    for name in files:
       
        #print(name)
        #print(os.path.join(path, name))
        if "log" in name and "old" not in path: 
            full_path_to_files = os.path.join(path, name)
            test = os.path.basename(full_path_to_files)
            sim_name = os.path.basename(os.path.dirname(full_path_to_files))
            save_folder_name = "./" + sim_name
            if not os.path.exists(save_folder_name):
                print("# Creating new directory: ",  save_folder_name)
                os.makedirs( save_folder_name)
                
            shutil.copy(src=full_path_to_files, dst=save_folder_name)
            
        
           # print(test)
            print(full_path_to_files )
        
