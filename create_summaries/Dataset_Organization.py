import os
import sys 

folders = [
    #"SST-cells_cond-GluN1-KO_Adult_V1",
    #"PV-cells_WT_Adult_V1", 
    #"PV-cells_WT_Young_V1", 
    #"PV-cells_cond-GluN1-KO_Adult_V1", 
    #"PYR-PV-SynGCaMP_WT_Young_V1",
    #"PYR-SynGCaMP_WT_Adult_V1",
    #"SST-cells_cond-GluN1-KO_Young_V1",
    #"SST-cells_WT_Adult_V1",
    #"SST-cells_WT_Young_V1",
    #"SST-cells_cond-GluN1-KO_Adult_V1_Taddy",
    #"SST-cells_WT_Adult_V1_Taddy"
    "GCaMP8m-Test_V1",
    "GCaMP8s-Test_V1",
]

base_path = os.path.expanduser('~/CURATED/Cibele/')

# folders that are NOT directly in base_path: name (used in folders above and in the
# summary file names, no "/") -> folder that holds the NWBs/ subfolder
other_folder_paths = {
    "GCaMP8m-Test_V1" : os.path.join(base_path, 'GCaMP-Test_V1', 'GCAmP8m'),
    "GCaMP8s-Test_V1" : os.path.join(base_path, 'GCaMP-Test_V1', 'GCAmP8s'),
}

def get_datafolder(c):
    """ NWBs folder of the dataset c """
    return os.path.join(other_folder_paths.get(c, os.path.join(base_path, c)), 'NWBs')

summary_folder = base_path + '/summary'

quantity = 'dFoF' 
#quantity = 'Deconvolved_dFoF' 
#quantity = 'Deconvolved_correctedFluo', 
#quantity = 'Zscore_correctedFluo'
filtering_cond_name = None #'Rest' 

#sys.argv = {#'roi_to_neuropil_fluo_inclusion_factor' : neuropil_inclusion_factor,
            #'neuropil_correction_factor' : neuropil_correction_factor,
            #'computed_neuropil_fact_method' : 'slope'}

# age intervals in Young
AGE_INTERVALS = [\
    (15,19), (20,23), (24,27) #, (16,21), (22,27)
    ]

# to be a valid dataset:
nMIN_DATAFILES = 2

parallelized = False
debug = False

def datasets_func(key='contrast',
        values=[0.5,1.0]):

    datasets = {}
    for c in folders:

        for val in values:

            # we split young animals into age groups
            if 'Young' in c:
                for interval in AGE_INTERVALS:
                    datasets[c.replace('Young', 'P%i-P%i' % interval)+'_%s-%.1f' % (key,val)] =\
                        {'datafolder':get_datafolder(c), 
                            'age_interval':interval}
            else : 
                datasets[c+'_%s-%.1f' % (key, val)] =\
                    {'datafolder':get_datafolder(c), 
                        'age_interval':None}
                
    return datasets


if __name__=='__main__':
    from pprint import pprint
    pprint(datasets)

