#%%
import os, sys

try:
    summaries_dir = os.path.dirname(os.path.abspath(__file__))
except NameError: # interactive window
    summaries_dir = os.path.join(os.path.expanduser('~'), 'code', 'merged', 'create_summaries')

neuropil_inclusion_factors = [2] #[1.15,2,3,4] 
neuropil_correction_factors = [0.85, 1] #[0,0.15,0.3,0.55,0.7, 0.85,1] 

for neuropil_inclusion_factor in neuropil_inclusion_factors : 

    for neuropil_correction_factor in neuropil_correction_factors : 
        print('COMPUTING : inclusion_factor = ' + str(neuropil_inclusion_factor) + ' and correction_factor = ' + str(neuropil_correction_factor))

        sys.argv = {'roi_to_neuropil_fluo_inclusion_factor' : neuropil_inclusion_factor,
                    'neuropil_correction_factor' : neuropil_correction_factor}
        
        #exec(open(os.path.join(summaries_dir, 'Tuning-Dataset.py')).read())
        exec(open(os.path.join(summaries_dir, 'Contrast-Dataset.py')).read())
