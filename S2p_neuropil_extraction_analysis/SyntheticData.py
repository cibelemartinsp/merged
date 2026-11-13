import numpy as np
import os, sys 
sys.path += ['/home/user/lab-notebook/astrid/suite2p']
sys.path += ['/home/user/lab-notebook/astrid/physion/src']

import suite2p
from suite2p.extraction.masks import create_masks
from suite2p.extraction.extract import extract_traces
from physion.analysis.read_NWB import Data
from itertools import product
import torch


#data = Data(nwb_filename)
# has attributes: data.dFoF, data.rawFluo, data.neuropil, ndata.nROIs, data.t_rawFluo, ...

# by default:
# F.npy
# Fneu.npy

# create a bunch
# Fneu-neuropil_min_radius-

# %%
neuropil_params = {
    'inner_neuropil_radius':2,
    'min_neuropil_pixels':350,
    'lam_percentile':50.0,
    'allow_overlap':False,
}

# %%
#syndata = SyntheticData(data, raw_data_folder)
#syndata.set_neuropil(neuropil_params)

class SyntheticData:

    """
    minimalist version of Data object, from which the neuropil trace of choice can be set.
    Can notably be used to create EpisodeData as Data would be.
    """

    def __init__(self,
                 original_data,
                 raw_data_folder):


        self.rawFluo = original_data.rawFluo[:,:]
        self.nROIs = original_data.nROIs
        self.t_rawFluo = original_data.t_rawFluo

        self.raw_data_folder = raw_data_folder


    def set_neuropil(self,
                    neuropil_params):

        # neuropil caclulation

        self.neuropil = get_Fneu(self.raw_data_folder,
                             neuropil_params)

        self.t_neuropil = original_data.t_neuropil


    def get_Fneu(folderpath, neuropil_params):

        Fneu_filename = 'Fneu' + get_neuropil_params_filename(neuropil_params)

        try: 
            Fneu = np.load(folderpath + Fneu_filename, allow_pickle = True)
            return Fneu
        
        except FileNotFoundError: 
            print("[!!] %s does not exist in %s [!!] " % (Fneu_filename, folderpath))




#%%

    
def get_neuropil_params_filename(neuropil_params) :

    filename = ''
    for key in neuropil_params:
        if type(neuropil_params[key]) in [int]:
            filename += '-%s-%i' % (key, neuropil_params[key])
        elif type(neuropil_params[key]) in [bool]:
            filename += '-%s-%s' % (key, neuropil_params[key])
        else:
            filename += '-%s-%.1f' % (key, neuropil_params[key])
    filename += '.npy'

    return filename

def get_registration_and_detection_relevant_outputs(folderpath):

    plane0_folderpath = folderpath + '/suite2p/plane0/'
    stat = np.load(plane0_folderpath + 'stat.npy', allow_pickle=True)
    ops = np.load(plane0_folderpath + 'ops.npy', allow_pickle=True).item()
    Ly = ops['Ly']
    Lx = ops['Lx']

    f_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename= plane0_folderpath + 'data.bin')

    return plane0_folderpath, stat, Ly, Lx, f_reg


def compute_F_Fneu_traces(folderpath, neuropil_params, mode = 'return'):

    plane0_folderpath, stat, Ly, Lx, f_reg = get_registration_and_detection_relevant_outputs(folderpath)

    cell_masks, neuropil_masks = create_masks(stats = stat, Ly = Ly, Lx = Lx, 
                                            lam_percentile = neuropil_params['lam_percentile'],
                                            allow_overlap = neuropil_params['allow_overlap'],
                                            inner_neuropil_radius = neuropil_params['inner_neuropil_radius'],
                                            min_neuropil_pixels = neuropil_params['min_neuropil_pixels'])

    F, Fneu = extract_traces(f_reg, cell_masks, neuropil_masks, batch_size=500, device = torch.device('cpu'))

    if mode == 'save' : 
            if not os.path.isdir(plane0_folderpath + '/traces/') : 
                os.mkdir(plane0_folderpath + '/traces/')

            filename = get_neuropil_params_filename(neuropil_params) 
            np.save(plane0_folderpath + '/traces/' + 'Fneu' + filename, Fneu)
            if neuropil_params['allow_overlap'] == True : 
                np.save(plane0_folderpath + '/traces/' + 'F' + filename, F)

    elif mode == 'return' : 
            return F, Fneu


def get_lab_default_db_and_settings(folderpath): 

    db = suite2p.default_db()
    db['data_path'] = [folderpath] #folder of the h5 file
    db['input_format'] = 'h5'
    db['save_path0'] = folderpath 

    settings = suite2p.default_settings()
    settings['torch_device'] = 'cpu'
    #settings['fs'] = 
    settings['run']['do_deconvolution'] = False

    return db, settings

def get_neuropil_params_dicts(set_of_neuropil_params) : #may be written more efficiently

    params_list = list(product(set_of_neuropil_params['inner_neuropil_radius'],
                                set_of_neuropil_params['min_neuropil_pixels'],
                                set_of_neuropil_params['lam_percentile'],
                                set_of_neuropil_params['allow_overlap']))

    params_dicts = []

    for params_list in params_list : 

        instance_params = {
        'inner_neuropil_radius': params_list[0],
        'min_neuropil_pixels': params_list[1],
        'lam_percentile': params_list[2],
        'allow_overlap': params_list[3],
        }

        params_dicts.append(instance_params)

    return params_dicts
#%%

if False:

    from physion.analysis.read_NWB import Data

    original_data = Data(nwb_filename)
    # has attributes: data.dFoF, data.rawFluo, data.neuropil, ndata.nROIs, data.t_rawFluo, ...

    new_neuropil = ...

    newData = SyntheticData(original_data,
                            new_neuropil)


# %%