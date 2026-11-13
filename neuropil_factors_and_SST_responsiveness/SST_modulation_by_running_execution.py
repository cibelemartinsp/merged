# -*- coding: utf-8 -*-
#%%
import numpy as np
import os, sys 
os.chdir('/home/user/lab-notebook/astrid/physion/src')

sys.path += ['/home/user/lab-notebook/astrid']
from SST_modulation_by_running_functions import *
#%%


folders = [#"PV-cells_WT_Adult_V1", 
    "SST-cells_WT_Adult_V1",
    "SST-cells_cond-GluN1-KO_Adult_V1"
    ]

plot_corrcoeff_hist_and_pie(folders, summary_protocol = "Tunings") 

run_events_ind_bounds, stationary_events_ind_bounds = get_run_stationary_events_inds(data, return_bounds = True)
ind_run, ind_stationary = get_run_stationary_events_inds(data, return_bounds = False)

folder = folders[0]
summary = np.load('/home/user/DATA/Astrid/run_rest_summary/Tunings_SST-cells_cond-GluN1-KO_Adult_V1_contrast-1.0.npy', allow_pickle=True)
RUN, TRACE, ses_included = get_run_triggered_dfof_and_run(folder, summary) 

#%%
summary_protocol = 'Sensitivities'
protocol_control_cond = 'angle-0.0'

summary_protocol = 'Tunings'
protocol_control_cond = 'contrast-1.0'

time_from_other_events_cond = None
run_longer_than_cond = 2

plot_bar_mean_deltaF_run_statio(folders, summary_protocol, protocol_control_cond, time_from_other_events_cond = time_from_other_events_cond, run_longer_than_cond = run_longer_than_cond)

summary = np.load('/home/user/DATA/Astrid/run_rest_summary/Tunings_SST-cells_cond-GluN1-KO_Adult_V1_contrast-1.0.npy', allow_pickle=True)
color = pt.tab10(2)
plot_average_run_triggered_dfof_and_run(summary, color, plot_diff_pre_post = False, time_from_other_events_cond = time_from_other_events_cond, run_longer_than_cond = run_longer_than_cond, return_only_sign = True)

summary = np.load('/home/user/DATA/Astrid/run_rest_summary/Tunings_SST-cells_WT_Adult_V1_contrast-1.0.npy', allow_pickle=True)
color = pt.tab10(1)
plot_average_run_triggered_dfof_and_run(summary, color, plot_diff_pre_post = False, time_from_other_events_cond = time_from_other_events_cond, run_longer_than_cond = run_longer_than_cond, return_only_sign = None)