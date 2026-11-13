#%%
import numpy as np

def fill_missing_contrast_values_with_nans(Sensitivity, Episodes): 
    corrected_sensitivity = {}
    contrast_values = np.unique(Episodes.contrast)
    nROIs = Episodes.data.nROIs

    for key in Sensitivity.keys() : 
        corrected_sensitivity[key] = []

    for contrast in contrast_values : 
        if contrast in Sensitivity['contrast'] : 
            ind_contrast = np.argwhere(Sensitivity['contrast'] == contrast)[0][0]
            for key in Sensitivity.keys() :
                if key != 'contrast' and key != 'ntrials': 
                    corrected_sensitivity[key].append(Sensitivity[key][:,ind_contrast])
                if key == 'ntrials' : 
                    corrected_sensitivity[key].append(Sensitivity[key][ind_contrast])
        else : 
            for key in Sensitivity.keys() : 
                if key != 'contrast': 
                    if 'significant' in key : 
                        corrected_sensitivity[key].append([False]*nROIs)
                    elif key == 'ntrials' : 
                        corrected_sensitivity[key].append(np.nan)
                    else : 
                        corrected_sensitivity[key].append([np.nan]*nROIs)

    #reshape correctly 
    for key in Sensitivity.keys() : 
        if key != 'contrast' : 
            corrected_sensitivity[key]  = np.array(corrected_sensitivity[key]).T
    corrected_sensitivity['contrast'] = contrast_values

    return corrected_sensitivity
