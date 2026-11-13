#%%
import os, sys 
# physion and the merged modules are taken from merged/ (parent of general_tools/)
try:
    merged_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError: # interactive window
    merged_root = os.path.join(os.path.expanduser('~'), 'code', 'merged')
for path in [merged_root, os.path.join(merged_root, 'physion', 'src')]:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
import numpy as np
from scipy.stats import sem
import itertools
from create_summaries.Preprocessing_Settings import get_dFoF_params
from create_summaries.propagated_uncertainty import session_sem_with_indepedance_hypothesis_universal

def get_uncertainty(Responses, uncertainty_sy, summaries = None): 
    
    if uncertainty_sy == 'std' : 
        uncertainty_sy = np.nanstd(Responses, axis = 0)
    elif uncertainty_sy == 'sem' : 
        uncertainty_sy = sem(Responses, axis = 0, nan_policy = 'omit')
    elif uncertainty_sy == 'propagated' : 
        print('WARNING :  not implemented yet for tunings')
        uncertainty_sy = session_sem_with_indepedance_hypothesis_universal(summaries)

    return uncertainty_sy

def get_protocol_keys(summary_protocol = 'Tunings') : 

    if summary_protocol == 'Tunings' : 
        keys = ['_contrast-1.0', 
            '_contrast-0.5' ]

    elif summary_protocol == 'Sensitivities' :
        keys =  ['_angle-90.0' , 
            '_angle-0.0' ] 

    return keys

         
def get_nwb_filenames_from_summary(folder, summary_path, summary_protocol):

    key = get_protocol_keys(summary_protocol = summary_protocol)[0]

    summary= np.load(summary_path + '/' + summary_protocol + '_' + folder + key + '.npy', allow_pickle=True)  #only one control cond is sufficient
    nwb_filenames = [local_nwb_path(s['datafile']) for s in summary]

    return nwb_filenames


def local_nwb_path(datafile):
    """ summaries built on another machine store e.g. /home/user/CURATED/...
        -> same file in the local ~/CURATED/ when the stored path does not exist """
    if (not os.path.exists(datafile)) and ('/CURATED/' in datafile):
        return os.path.join(os.path.expanduser('~'), 'CURATED', datafile.split('/CURATED/', 1)[1])
    return datafile


def get_summary_names(folders, summary_paths, summary_protocols, summary_prefixe_names = None, summary_sufixe_names = None, template = [], add_keys = False): 

    def input_form_check(input, ref = [1]): 
        if type(input) == str:
            input = [input]*len(ref)
        if type(input) == list and len(input) != len(ref):
            input = input *len(ref)
        return input
    
    folders = input_form_check(folders)
    summary_protocols = input_form_check(summary_protocols, folders)
    summary_paths = input_form_check(summary_paths, folders)
    summary_prefixe_names = input_form_check(summary_prefixe_names)
    summary_sufixe_names = input_form_check(summary_sufixe_names)
    
    if summary_prefixe_names == None : 
        summary_prefixe_names = []
    if summary_sufixe_names == None : 
        summary_sufixe_names = []

    summary_filenames = []
    for i,folder in enumerate(folders) :

        for item in template : 

            if item == 'multiple_corfact': 
                for neuropil_correction_factor in [0,0.15,0.3,0.55,0.7,0.85,1] : 
                    inclufact = get_dFoF_params(folder, None)['roi_to_neuropil_fluo_inclusion_factor']
                    summary_prefixe_names.append('roi_to_neuropil_fluo_inclusion_factor_' + str(inclufact) + '_neuropil_correction_factor_' + str(neuropil_correction_factor) + '_') 
            
            if item == 'inclufact_2_3' : 
                #default_inclufact = get_dFoF_params(folder, None)['roi_to_neuropil_fluo_inclusion_factor']
                summary_prefixe_names.append('')
                for inclufact in [2, 3]:
                    summary_prefixe_names.append('roi_to_neuropil_fluo_inclusion_factor_' + str(inclufact) + '_neuropil_correction_factor_0.7_') 

            if item == 'young' : 
                for interval in (15,19), (20,23), (24,27) : 
                        summary_sufixe_names.append('_P%i-P%i' % interval)

        if not summary_prefixe_names : 
            summary_prefixe_names = ['']
        if not summary_sufixe_names : 
            summary_sufixe_names = ['']

        for summary_prefixe_name, summary_sufixe_name in list(itertools.product(summary_prefixe_names,summary_sufixe_names)) :

            if add_keys : 
                keys = [key + '.npy' for key in get_protocol_keys(summary_protocols[i])]
            else : 
                keys = ['']
                
            for key in keys : 
                    summary_name = summary_paths[i] + '/' + summary_prefixe_name + summary_protocols[i] + '_' + folder + summary_sufixe_name + key 
                    summary_filenames.append(summary_name)

    return summary_filenames

