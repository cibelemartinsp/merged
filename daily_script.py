# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:31:34 2026

@author: astrid.nicolini
"""
# %% 
import numpy as np
import os, sys 
sys.path.insert(0,'/home/user/lab-notebook/astrid/physion/src')

#sys.path = [sys.path[i] for i in range(-1,len(sys.path)-1)]
from physion.analysis.read_NWB import scan_folder_for_NWBfiles, Data
from physion.analysis.episodes.build import *
import matplotlib.pyplot as plt
import physion

from general_tools.cell_type_color_codes import *
from general_tools.plot_general_tools import *
#%% TEMPLATES

filename = '/home/user/DATA/Astrid/Cibele_data/PV-cells_WT_Adult_V1/NWBs/2024_03_22-14-35-26.nwb' #contrasts
filename = '/home/user/DATA/Astrid/Cibele_data/SST-cells_cond-GluN1-KO_Adult_V1/NWBs/2026_03_06-15-14-09.nwb' #orientations

data = physion.analysis.read_NWB.Data(filename, verbose=False) 
#data = Data(filename, verbose=False)
#protocol_name=[p for p in data.protocols if '8orientation' in p][0]
dFoF_parameters = dict(\
        roi_to_neuropil_fluo_inclusion_factor=1.15,
        neuropil_correction_factor = 0.7,
        method_for_F0 = 'sliding_percentile',
        percentile=5., # percent
        sliding_window = 5*60, # seconds
        with_computed_neuropil_fact=True,
        computed_neuropil_fact_method='slope',

)
print(data.protocols)
data.build_dFoF(**dFoF_parameters, verbose=False)
#data.build_Deconvolved(quantity = 'dFoF')
data.build_running()

Episodes = EpisodeData(data, 
                        quantities=['dFoF', 'running'], 
                        protocol_name = data.protocols[0],
                        verbose=False)

filename = '/home/user/DATA/Astrid/Cibele_data/PV-cells_WT_Young_V1/NWBs/2025_07_11-16-12-30.nwb'
#%%

Tunings = np.load('/home/user/DATA/Astrid/Cibele_data/summary/Tunings_PV-cells_WT_Adult_V1_contrast-1.0.npy', allow_pickle = True)

test = 'slope'

if test : 
    print('1')
else : 
    print('2')

#%% allen


filename = Tunings_ko[11]['datafile']
allen_data = physion.analysis.read_NWB.Data(filename, verbose=False) 
#data = Data(filename, verbose=False)
#protocol_name=[p for p in data.protocols if '8orientation' in p][0]
dFoF_parameters = dict(\
        roi_to_neuropil_fluo_inclusion_factor=1.15,
        neuropil_correction_factor = 0.7,
        roi_specific_neuropil_correction_factor = True,
        method_for_F0 = 'sliding_percentile',
        percentile=5., # percent
        sliding_window = 5*60, # seconds
)

allen_data.build_dFoF(**dFoF_parameters, verbose=False)

Episodes = EpisodeData(allen_data, 
                        quantities=['dFoF'], 
                        protocol_name = allen_data.protocols[0],
                        verbose=False)




#%%


#%%
Tunings = np.load('/home/user/DATA/Astrid/Cibele_data/summary/Tunings_PV-cells_WT_P20-P23_V1_contrast-1.0.npy', allow_pickle = True)
for T in Tunings : 
    print(T['significant_ROIs'])


#%%

stat_test_props = dict(interval_pre=[-1.,0],                                   
                       interval_post=[1.,2.],                                   
                       test='ttest',                                            
                       sign='positive')

response_significance_threshold = 0.001 # very very conservative

# possibility to **FILTER THE EPISODES**:
# stim_cond = (Episodes.t>0) & (Episodes.t<Episodes.time_duration[0])
# filtering_cond = Episodes.running[:,stim_cond].mean(axis=1)<0.1

Tuning = compute_tuning_response_per_cells(data, Episodes,
                                        #    filtering_cond=filtering_cond,
                                           quantity='dFoF',
                                           stat_test_props=stat_test_props,
                                           response_significance_threshold = response_significance_threshold,
                                           contrast=1,
                                           verbose=True)

#%%

summary_filenames = get_summary_names(folders, summary_paths, summary_protocols = 'Tunings')

#%%

#%%
summary_paths, folders, colors= get_working_data_defaults(experimenters = ['Cibele'],
                                                        folders_surnames = ['SST_WT', 'PV_WT'], 
                                                        color_with_grey=True)


#summary_paths = ['/home/user/DATA/Astrid/other_old_summaries/run_rest_summary']
colors = reshape_colors_with(colors, second_cond_grey = True)
neuropil_correction_factors = [0,0.15,0.3,0.55,0.7,0.85,1]

fig = plt.figure(figsize = (7*len(folders),10))
for i, folder in enumerate(folders) : 
    fig.text(x= 0.37+0.23*i, y= 1, s = '%s' % folder, color = colors[i][0], fontsize = 15)

axes = get_multiple_corrfact_axes_and_implement_mosaic(folders, fig)


i = 0
for folder, summary_path, folder_colors in zip(folders, summary_paths, colors) : 

    synthetic_folders, synthetic_colors, synthetic_summary_paths, alphas, special_dict  = get_input_multiple_corrfact_plot(folder, folder_colors, summary_path)
    special_dict['ylims'] = [-0.2,0.7]
    plot_responses(axes[i][0],
                synthetic_folders, 
                synthetic_colors,  
                alphas = alphas,
                summary_protocol = 'Tunings',
                clip_negative = False,
                special_dict = special_dict, 
                summary_paths = synthetic_summary_paths,
                sensitivitie_uncertainty = None) 

    axes[i][1].set_title('neuropil correction factors', fontsize = 13)
    plot_corrfact_legend_color_maps(colors = colors[i], fig = fig,  axes = [axes[i][1], axes[i][2]], neuropil_correction_factors = neuropil_correction_factors) 
    i+=1

#%%%
summary_paths, folders, colors= get_working_data_defaults(experimenters = ['Cibele'],
                                                        folders_surnames = ['PV_WT_P20', 'PV_WT_P24','PV_WT'],
                                                        color_with_grey=True)

summary_paths, folders, colors= get_working_data_defaults(experimenters = ['Cibele'],
                                                        folders_surnames = ['SST_WT_P15', 'SST_WT_P20', 'SST_WT_P24','SST_WT'],
                                                        color_with_grey=True)

summary_filenames = get_summary_names(folders, summary_paths, summary_protocols = 'Tunings')

mosaic = "A" 
fig= plt.figure(layout="constrained", figsize = (2*len(folders),6))
fig.get_layout_engine().set(wspace = 0.15)
ax_dict = fig.subplot_mosaic(mosaic)
colors = reshape_colors_with(colors)
plot_mean_F_val(ax_dict['A'], folders, colors, 'correctedFluo', summary_paths)

ymin = 0
ymax = 500
ax_dict['A'].set_ylim([ymin, ymax])
