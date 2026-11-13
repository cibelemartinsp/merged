#%%
import numpy as np
import os, sys 
sys.path += ['/home/user/lab-notebook/astrid/suite2p']
sys.path += ['/home/user/lab-notebook/astrid/physion/src']

import suite2p
from suite2p.run_s2p import run_s2p
from SyntheticData import compute_F_Fneu_traces, get_neuropil_params_filename, get_lab_default_db_and_settings, get_neuropil_params_dicts

#%% run_s2p in order to do registration and detection and to create data.bin


folderpath = "/home/user/Desktop/h5-11142025-001"

set_of_neuropil_params =  {
    'inner_neuropil_radius': [1,6],
    'min_neuropil_pixels': [350],
    'lam_percentile': [50.0],
    'allow_overlap': [False, True]
}

#%%
if __name__=='__main__':

    db, settings = get_lab_default_db_and_settings(folderpath)

    run_s2p(db=db, settings=settings)

    neuropil_params_dicts = get_neuropil_params_dicts(set_of_neuropil_params)

    for neuropil_params in neuropil_params_dicts : 
        compute_F_Fneu_traces(folderpath, neuropil_params, mode = 'save')


