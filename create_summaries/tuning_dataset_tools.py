#%% Tools necessary to create Tuning summaries with arousal cond, neuropil cond etc.

import os
import numpy as np
import pandas as pd 

def get_prefered_angles(prefered_angles_dataset, filename):

    short_filename = os.sep.join(filename.split('/')[-3:])
    prefered_angles = prefered_angles_dataset['prefered_angles'][(prefered_angles_dataset['short_filenames'] == short_filename)]

    if not prefered_angles.empty : 
        return prefered_angles.values[0]
    else : 
        raise ValueError('file %s not present in the prefered_angles_dataset' % short_filename)


def get_prefered_angles_dataset(data_folder, summary_folder, contrast) :

    folder = data_folder.split('/')[-2]
    prefered_angles_dataset = pd.DataFrame({})
    try : 
        Tunings = np.load(summary_folder + '/Tunings_%s_contrast-%s.npy' % (folder, contrast), allow_pickle=True)
        prefered_angles_dataset['short_filenames'] = [os.sep.join(Tuning['datafile'].split('/')[-3:]) for Tuning in Tunings] #ignore basepath
        prefered_angles_dataset['contrasts'] = [contrast]*len(prefered_angles_dataset['short_filenames'])
        prefered_angles_dataset['prefered_angles'] = [Tuning['prefered_angles'] for Tuning in Tunings]
        
    except: 
        print('No Tunings file found for folder %s and contrast %s' % (folder, contrast))
        print('You need to build the classical tunings summary first, before building one with a filtering condition')


    return prefered_angles_dataset

