#%%
import os, sys, shutil
sys.path += ["/home/user/lab-notebook/astrid/physion/src"]

from physion.analysis.read_NWB import scan_folder_for_NWBfiles

#%%
datafolder = '/home/user/DATA/Astrid/Taddy_data/photodiode_aligned/NWBs'

DATASET = scan_folder_for_NWBfiles(datafolder)

subjects_types = [txt.split('_')[1] for txt in DATASET['subjects']]

os.mkdir('/home/user/DATA/Astrid/Taddy_data/photodiode_aligned/SST_WT')
os.mkdir('/home/user/DATA/Astrid/Taddy_data/photodiode_aligned/SST_GluN1')
os.mkdir('/home/user/DATA/Astrid/Taddy_data/photodiode_aligned/SST_GluN3')
os.mkdir('/home/user/DATA/Astrid/Taddy_data/photodiode_aligned/SST_WT/NWBs')
os.mkdir('/home/user/DATA/Astrid/Taddy_data/photodiode_aligned/SST_GluN1/NWBs')
os.mkdir('/home/user/DATA/Astrid/Taddy_data/photodiode_aligned/SST_GluN3/NWBs')

for i, subject in enumerate(subjects_types) : 
    print(i)
    nwb_name = DATASET['files'][i].split('/')[-1]

    if subject in  ['SSTCre', 'SSTcre', 'SSTcreXtdTomato'] :
        dir = '/home/user/DATA/Astrid/Taddy_data/photodiode_aligned/SST_WT/NWBs/'

    elif subject in ['SSTCreNR1', 'SSTcreNR1'] : 
        dir = '/home/user/DATA/Astrid/Taddy_data/photodiode_aligned/SST_GluN1/NWBs/'

    elif subject in ['SSTCreNR3'] :
        dir = '/home/user/DATA/Astrid/Taddy_data/photodiode_aligned/SST_GluN3/NWBs/' 

    shutil.move( DATASET['files'][i], dir + nwb_name)

# %%
