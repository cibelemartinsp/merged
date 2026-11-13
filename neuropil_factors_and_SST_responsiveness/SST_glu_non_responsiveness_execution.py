# -*- coding: utf-8 -*-
#%%
import numpy as np
import os, sys , shutil 
os.chdir('/home/user/lab-notebook/astrid/physion/src')
sys.path += ['/home/user/lab-notebook/astrid/physion/src']
import physion.utils.plot_tools as pt

sys.path += ['/home/user/lab-notebook/astrid/neuropil_factors_and_SST_responsiveness']
from STT_glu_non_responsiveness_functions import *


#%% observations

Tunings = np.load('/home/user/DATA/Astrid/Tunings_SST-cells_cond-GluN1-KO_Adult_V1_contrast-0.5.npy', allow_pickle=True)   

Tunings[0]['datafile'] #clear response to behavior
Tunings[1]['datafile'] #clear response of 1 ROI to visual stim but not the others
Tunings[2]['datafile'] #mixed of responsive to visual and just noise trace
Tunings[4]['datafile'] #almost no responsive to visual stim
Tunings[5]['datafile'] #que du bruit, quelques réponses au behavior
Tunings[-1]['datafile'] #clear response to behavior but not to visual stim

#%%
folder = "SST-cells_cond-GluN1-KO_Adult_V1"

summary_folder = '/home/user/DATA/Astrid/Cibele_data/summary'
base_path = os.path.expanduser('~/DATA/Astrid/Cibele_data')
colors = [pt.tab10(2), 'lightgrey']

plot_percentage_of_responsiveness(folder, summary_folder, base_path, colors)

#%%#-----------------------------------------Cells responsivness profiles (to visual and crosscorelation to behavior)------------------------------------------------------#
summary_protocol = 'Tunings'
corr_coeff_funct = pearsonr
resp_profiles, corr_dict = get_population_responsiveness_profiles(folder, corr_coeff_funct = pearsonr, absolute_values = True)

print_population_responsiveness_relevant_percentages(resp_profiles, folder, summary_protocol = summary_protocol) 

plot_correlation_functof_responsiveness(corr_dict, folder, colors = ['lightgrey', pt.tab10(2)])

plot_corr_coeffs_distribution_between_sign_non_sign(corr_dict, visual_control = 'contrast', visual_control_values = [0.5,1.0], plot_by_roi = False) 


#%%#-----------------------------------------Responses's pval to tuning distribution ------------------------------------------------------#
PVAL_pref_angle, PVAL_min, iANGLE_of_pval_min = get_pval_of_tuning_response(folder, summary_folder,  plot_distribution = True, color = ['lightgrey', pt.tab10(2)])


#%% #-----------------------------------Effect of interval pre/post on nb significatif ROIs-----------------------------------------#
plot_responsiveness_at_different_post_stimuli_interval(folder, summary_folder,plot = True)



#%%
