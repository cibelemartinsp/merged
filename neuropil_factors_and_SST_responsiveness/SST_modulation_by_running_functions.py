# -*- coding: utf-8 -*-
#%%
import random
import numpy as np

import os, sys 
os.chdir('/home/user/lab-notebook/astrid')
sys.path += ['./physion/src']
from physion.analysis.read_NWB\
                         import scan_folder_for_NWBfiles, Data
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.protocols.orientation_tuning import *
import physion 

import matplotlib.pyplot as plt 
from matplotlib.pyplot import hist
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from physion.utils import plot_tools as pt

from scipy.signal import correlation_lags, correlate 
from scipy.stats import pearsonr, spearmanr
from scipy.stats import sem


#%%
#--------------------------------------correlation coefficients with running analysis -------------------------------------#
def get_session_corr_coeffs_and_ci(summary, ses_number, corr_coeff_funct = pearsonr, verbose = False):

    #build the session
    filename = summary[ses_number]['datafile']

    dFoF_parameters = dict(\
            roi_to_neuropil_fluo_inclusion_factor=1.15,
            neuropil_correction_factor = 0.7,
            method_for_F0 = 'sliding_percentile',
            percentile=5., # percent
            sliding_window = 5*60, # seconds
    )

    data = physion.analysis.read_NWB.Data(filename, verbose=False) 
    data.build_dFoF(**dFoF_parameters, verbose=False)
    data.running = data.build_running(specific_time_sampling = data.t_dFoF)

    behavior = data.running

    rois_corr_coeffs = []
    rois_corr_coeffs_significance = []
    #for each ROI : 
    for i in range(len(data.dFoF)) : 

            # compute pearson corr coeff with best lag 
            #lag = correlation_lags(len(behavior), len(data.dFoF[i]))[np.argmax(correlate(behavior,data.dFoF[i]))]
            #df = np.roll(data.dFoF[i], shift = lag)
            #rois_corr_coeffs.append(corr_coeff_funct(behavior, df)[0])
            rois_corr_coeffs.append(corr_coeff_funct(behavior, data.dFoF[i])[0])

            #compute a 1000 shuffles's corr coeffs 
            shifted_beh_corr_coeffs = []
            for j in range(1000) :
                    shifted_behavior = np.roll(behavior, shift = random.randint(1,len(behavior)))
                    shifted_beh_corr_coeffs.append(corr_coeff_funct(shifted_behavior, data.dFoF[i]))
            
            if verbose : 
                    print(str((np.sum(rois_corr_coeffs_significance) * 100)/len(rois_corr_coeffs_significance)) + ' %')

            #compute confidence interval and significance
            ci = np.quantile(np.array(shifted_beh_corr_coeffs)[:,0], q = [0.05,0.95])
            rois_corr_coeffs_significance.append(rois_corr_coeffs[-1] < ci[0] or rois_corr_coeffs[-1] > ci[1])


    return rois_corr_coeffs, rois_corr_coeffs_significance 


def plot_corrcoeff_hist_and_pie(folders, summary_protocol = "Tunings") : 

    for i, folder in enumerate(folders) : 
        print('Computing ' + folder)
        if summary_protocol == 'Tunings' : #no need to loop on control condition
                summary = np.load('/home/user/DATA/Astrid/run_rest_summary/' + summary_protocol + '_' + folder + '_' + 'contrast-1.0.npy', allow_pickle=True)  


        ROIS_CORR_COEFFS = []
        ROIS_CORR_SIGNIFICANCE = []

        for ses_number in range(len(summary)) : 
                print(ses_number)
                rois_corr_coeffs, rois_corr_coeffs_significance = get_session_corr_coeffs_and_ci(summary, ses_number, corr_coeff_funct = pearsonr, verbose = False)
                ROIS_CORR_SIGNIFICANCE.append(rois_corr_coeffs_significance)
                ROIS_CORR_COEFFS.append(rois_corr_coeffs)

        ROIS_CORR_SIGNIFICANCE = np.concatenate(ROIS_CORR_SIGNIFICANCE)
        ROIS_CORR_COEFFS = np.concatenate(ROIS_CORR_COEFFS)


        color_list = [['lightgrey','navajowhite', pt.tab10(1)] , ['lightgrey','lightgreen', pt.tab10(2)]][i]

        x = ROIS_CORR_COEFFS 
        x_sign = ROIS_CORR_COEFFS[ROIS_CORR_SIGNIFICANCE]
        x_nonsign = ROIS_CORR_COEFFS[~ROIS_CORR_SIGNIFICANCE]
        x_sign_neg = [val for val in x_sign if val < 0]
        x_sign_pos = [val for val in x_sign if val > 0]

        fig, ax = plt.subplots(figsize=(6, 6))

        # hist
        style_nonsign = {'facecolor':  color_list[0], 'edgecolor': 'white', 'linewidth': 1}
        style_neg = {'facecolor':  color_list[1], 'edgecolor': 'black', 'linewidth': 1}
        style_pos = {'facecolor': color_list[2], 'edgecolor': 'black', 'linewidth': 1}
        ax.hist(x_nonsign, bins = 20, range = (-1,1), align = 'mid', **style_nonsign)
        ax.hist(x_sign_neg, bins = 10, range = (-1,0), align = 'left', **style_neg)
        ax.hist(x_sign_pos, bins = 10, range = (0,1), align = 'right', **style_pos)
        ax.set_ylabel('number of cells')
        ax.set_xlabel('Pearson correlation coefficient')

        # create inset plot
        sub_ax = inset_axes(
        parent_axes=ax,
        width="25%",
        height="25%",
        loc = "upper left",
        borderpad=2  
        )

        perc_nonsign = len(x_nonsign)/len(x) * 100
        perc_neg = len(x_sign_neg)/len(x) * 100
        perc_pos = len(x_sign_pos)/len(x) * 100


        wedges, texts = sub_ax.pie([perc_nonsign, perc_neg, perc_pos], 
                                labels = [f'{perc_nonsign:.1f}% \n n.s.',f'{perc_neg:.1f}% \n neg', f'{perc_pos:.1f}% \n pos'], 
                                colors = color_list, 
                                startangle = -150,
                                wedgeprops={"edgecolor":"black",'linewidth': 1, 'width' : 0.5},
                                textprops = {"fontsize" : 10})
        for text, color in zip(texts, color_list):
                text.set_color(color)

        sub_ax.set_axis_off()
        plt.show()

#---------------------------------------------Run event triggered analysis---------------------------------------#

def get_run_stationary_events_inds(data, return_bounds = False, ses_number = 0, time_from_other_events_cond = None, run_longer_than_cond = None) : 

    """
    To know : 
    stationary periods can begin at the closest timestamps from -0.5 or +1.5 sec of run periods. This value can be under these thresholds 
    """
    #get run indices and joined gaps < 0.5
    ind_run = np.concatenate(np.argwhere(data.running > 0.1))
    ind_run = [x.item() for x in ind_run] #simplier shape

    i=1
    joined_ind_run = [ind_run[0]] #the indice when run>0.1 joined
    while i < len(ind_run) : 
        if ind_run[i] == ind_run[i-1] + 1 :
            joined_ind_run.append(ind_run[i])
        else : 
            if data.t_dFoF[ind_run[i]] - data.t_dFoF[ind_run[i-1]] < 0.5 : 
                joined_ind_run.extend(np.arange(ind_run[i-1]+1,ind_run[i]+1).tolist()) #fill the gap 
            else : 
                joined_ind_run.append(ind_run[i]) #don't fill the gap
        i+=1


    #create list of list of run events bounds 
    run_events_ind_bounds = []
    bounds = np.concatenate(([0],np.concatenate(np.argwhere(np.diff(joined_ind_run) != 1)).tolist())) 


    if time_from_other_events_cond is not None : 
        run_events_ind_bounds.append([joined_ind_run[bounds[0]], joined_ind_run[bounds[1]]]) #first run period consider not following another one
        for val1, val2 in zip(bounds[1:-1], bounds[2:]) : 
            if data.t_dFoF[joined_ind_run[val1+1]] - data.t_dFoF[joined_ind_run[val1]] >= time_from_other_events_cond : #run periods must be distant from more than x sec
                run_events_ind_bounds.append([joined_ind_run[val1+1], joined_ind_run[val2]]) 

    if run_longer_than_cond is not None : 
        for val1, val2 in zip(bounds[:-1], bounds[1:]) : 
            if data.t_dFoF[joined_ind_run[val2]] - data.t_dFoF[joined_ind_run[val1+1]] >= run_longer_than_cond : #run periods must be longer than x sec
                run_events_ind_bounds.append([joined_ind_run[val1+1], joined_ind_run[val2]]) 


    if bounds[0] +1 == run_events_ind_bounds[0][0] : 
        run_events_ind_bounds[0][0] += -1 #to correct for first value being +1, if the first value has been conserved

    #create list of list of extended run events bounds (ie with 0.5 pre and 1.5 post)
    extended_run_events_bounds = []
    ind_distance_max_to_0p5 = int(0.5/np.max(np.diff(data.t_dFoF))) + 1 #to not look in the whole list every time 
    ind_distance_max_to_1p5 = int(1.5/np.max(np.diff(data.t_dFoF)))+ 1
    for i in range(len(run_events_ind_bounds)) : 
        istart, istop = run_events_ind_bounds[i]
        flag = 0
        if istart - ind_distance_max_to_0p5 < 0 : 
            istart_extended = 0
            flag = 1
        if istop + ind_distance_max_to_1p5 > len(data.t_dFoF) : 
            istop_extended = len(data.t_dFoF)
            flag = 2
        
        if flag != 1 : #not pretty
            istart_extended = ind_distance_max_to_0p5 - np.argmin(np.abs(data.t_dFoF[istart - ind_distance_max_to_0p5 : istart] - (data.t_dFoF[istart] - 0.5))) #get the ind of the timestamp the closest to istart -0.5
        if flag !=2 : 
            istop_extended = np.argmin(np.abs(data.t_dFoF[istop : istop + ind_distance_max_to_1p5 ] - (data.t_dFoF[istop] + 1.5))) #get the ind of the timestamp the closest to istop +1.5

        extended_run_events_bounds.append([istart - istart_extended, istop + istop_extended])


    #create list of list of stationary events bounds 
    stationary_events_ind_bounds = []

    for i in range(len(extended_run_events_bounds)-1) :
        if extended_run_events_bounds[i+1][0] - extended_run_events_bounds[i][1] > 0 : #take non empty gap between extended run events 
            stationary_events_ind_bounds.append([extended_run_events_bounds[i][1],extended_run_events_bounds[i+1][0]])

    #test for run/stationary period > 5%
    #test in timestamps length 
    dur_stationary = np.sum(np.diff(data.t_dFoF[stationary_events_ind_bounds]))
    dur_run = np.sum(np.diff(data.t_dFoF[run_events_ind_bounds]))
    dur_ses = data.t_dFoF[-1]

    if dur_stationary/dur_ses < 0.05 or dur_run/dur_ses < 0.05 :
        print("=================================== ses_number = " + str(ses_number) + " =========================================")
        print("Percentage of either locomotion or stationary periods is lower than 5% : session is discarded")
        print("=============================================================================================")
        return None, None  

    else : 
        if return_bounds : 
            return run_events_ind_bounds, stationary_events_ind_bounds
        else : 
            inds_run = np.concatenate([np.arange(event_bound[0], event_bound[1]) for event_bound in run_events_ind_bounds])
            inds_stationary = np.concatenate([np.arange(event_bound[0], event_bound[1]) for event_bound in stationary_events_ind_bounds])
            return inds_run, inds_stationary
        

def plot_bar_mean_deltaF_run_statio(folders, summary_protocol, protocol_control_cond, time_from_other_events_cond, run_longer_than_cond) : 

    if time_from_other_events_cond is not None : 
        run_cond_name = 'time_from_other_events_cond = ' + str(time_from_other_events_cond)
    if run_longer_than_cond is not None : 
        run_cond_name = 'run_longer_than_cond = ' + str(run_longer_than_cond)

    for i, folder in enumerate(folders) :

        summary = np.load('/home/user/DATA/Astrid/run_rest_summary/' + summary_protocol + '_' + folder + '_' + protocol_control_cond + '.npy', allow_pickle=True)  

        dFoF_parameters = dict(\
                roi_to_neuropil_fluo_inclusion_factor=1.15,
                neuropil_correction_factor = 0.7,
                method_for_F0 = 'sliding_percentile',
                percentile=5., # percent
                sliding_window = 5*60, # seconds
        )

        Frun = []
        Fstatio = []
        n_cell = []
        for ses_number in range(len(summary)) : 
            print(ses_number)

            filename = summary[ses_number]['datafile']
            data = physion.analysis.read_NWB.Data(filename, verbose=False) 
            data.build_dFoF(**dFoF_parameters, verbose=False)
            data.running = data.build_running(specific_time_sampling = data.t_dFoF)

            ind_run, ind_stationary = get_run_stationary_events_inds(data, return_bounds = True, 
                                                                    ses_number = ses_number, 
                                                                    time_from_other_events_cond = time_from_other_events_cond, 
                                                                    run_longer_than_cond = run_longer_than_cond)
            
            if ind_run is not None : 
                frun = np.mean([np.mean(data.dFoF[i][ind_run]) for i in range(len(data.dFoF))])
                fstatio = np.mean([np.mean(data.dFoF[i][ind_stationary]) for i in range(len(data.dFoF))])
                Frun.append(frun)
                Fstatio.append(fstatio)
                n_cell.append(len(data.dFoF))
            
            else : 
                Frun.append(np.nan)
                Fstatio.append(np.nan)

        #plot 
        i = i +1
        plt.figure(figsize = (3,6))
        plt.errorbar(x = [0,1], y = [np.nanmean(Fstatio), np.nanmean(Frun)], yerr =  [stats.sem(Fstatio, nan_policy = 'omit'), stats.sem(Frun, nan_policy = 'omit')], fmt='.', elinewidth=1, capthick=1,  capsize = 5, color = 'black')
        plt.bar(x = [0,1], height = [np.nanmean(Fstatio), np.nanmean(Frun)], width = 0.7, color = pt.tab10(i))
        plt.ylabel('$\\delta$ $\\Delta$F/F')
        plt.ylim(0,2.1)
        plt.xticks( [0,1],  ['rest','run'])
        plt.text(s=folder, x = -0.5, y = -0.5)
        plt.text(s='error bar = sem', x = -0.5, y = -0.6)
        plt.text(s=summary_protocol,  x = -0.5, y = -0.7)
        plt.text(s=run_cond_name,  x = -0.5, y = -0.8)
        plt.text(s='N='+str(np.sum(n_cell)),  x = -0.08, y = 0.05, rotation = 'vertical', fontsize = 12)
        plt.text(s='N='+str(np.sum(n_cell)),  x = 0.92, y = 0.05, rotation = 'vertical', fontsize = 12)
        plt.show()




def get_run_triggered_dfof_and_run(folder, summary, time_from_other_events_cond, run_longer_than_cond, return_only_sign = None) : 

    """
    -10sec to 10sec
    note : not included events starting before 10sec of experiment or finishing after the last 10sec of experiment
    """

    dFoF_parameters = dict(\
            roi_to_neuropil_fluo_inclusion_factor=1.15,
            neuropil_correction_factor = 0.7,
            method_for_F0 = 'sliding_percentile',
            percentile=5., # percent
            sliding_window = 5*60, # seconds
    )

    RUN = []
    TRACE = []
    ses_included = []
    
    for ses_number in range(len(summary)) : 

        print(ses_number)
        filename = summary[ses_number]['datafile']
        data = physion.analysis.read_NWB.Data(filename, verbose=False) 
        data.build_dFoF(**dFoF_parameters, verbose=False)
        data.running = data.build_running(specific_time_sampling = data.t_dFoF)

        run_events_ind_bounds, stationary_events_ind_bounds = get_run_stationary_events_inds(data, return_bounds = True, ses_number = ses_number, time_from_other_events_cond = time_from_other_events_cond, run_longer_than_cond = run_longer_than_cond)
        if run_events_ind_bounds is not None : 

            ses_included.append(ses_number)
            #exclude the run events to close to the beginning/ending of the experiment
            ind_first_10sec = np.argwhere(data.t_dFoF >= 10)[0]
            ind_last_10sec = np.argwhere(data.t_dFoF <= data.t_dFoF[-1] - 10)[-1]
            run_events_ind_bounds_filtered = [event for event in run_events_ind_bounds if event[0] > ind_first_10sec and event[1] < ind_last_10sec]

            run_timestamps_onsets = data.t_dFoF[np.array(run_events_ind_bounds_filtered)[:,0]]
            run_timestamps_onsets_minus10 = [np.argmin(np.abs(data.t_dFoF - (t_onset - 10))) for t_onset in run_timestamps_onsets]
            run_timestamps_onsets_plus10 = [np.argmin(np.abs(data.t_dFoF - (t_onset + 10))) for t_onset in run_timestamps_onsets]

            run_events_ind_plus_minus10 =  [np.arange(minus10,plus10+1) for minus10,plus10 in zip(run_timestamps_onsets_minus10,run_timestamps_onsets_plus10)]


            traces = [np.mean(data.dFoF[i][run_events_ind_plus_minus10], axis = 0) for i in range(len(data.dFoF))]
            if return_only_sign is not None :
                traces = [traces[i] for i in range(len(traces)) if summary[ses_number]['significant_ROIs'][i] == return_only_sign]
            run = np.mean(data.running[run_events_ind_plus_minus10], axis = 0)
            RUN.append(run)
            TRACE.append(traces)
    
    return np.array(RUN), TRACE, ses_included

#to correct, works but not clean 
def plot_average_run_triggered_dfof_and_run(summary, color, plot_diff_pre_post = True, time_from_other_events_cond = None, run_longer_than_cond = None, return_only_sign = None) : 

    summary_protocol, protocol_control_cond, folder = get_summary_info(summary)

    RUN, TRACE, ses_included = get_run_triggered_dfof_and_run(folder, summary, time_from_other_events_cond = time_from_other_events_cond, run_longer_than_cond = run_longer_than_cond, return_only_sign = return_only_sign) 

    mean_RUN = np.mean(RUN, axis = 0)
    TRACE = np.array(np.concatenate(TRACE))
    mean_TRACE = np.mean(TRACE, axis = 0)
    n_cell = len(TRACE)
    sign_ROIs = np.concatenate([summary[ses_number]['significant_ROIs'] for ses_number in ses_included])


    #plot run and dfof traces align to locomotion onset

    if time_from_other_events_cond is not None : 
        run_cond_name = 'time_from_other_events_cond = ' + str(time_from_other_events_cond)
    if run_longer_than_cond is not None : 
        run_cond_name = 'run_longer_than_cond = ' + str(run_longer_than_cond)

    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize = (15,10))

    ax0.plot(mean_RUN, color = color)
    ax0.fill_between(np.arange(0,len(mean_RUN)), mean_RUN + sem(RUN), mean_RUN - sem(RUN), alpha=0.3, color = 'grey')
    ax0.set_xticks(ticks = [0,len(mean_RUN)/4,len(mean_RUN)/2,(3*len(mean_RUN))/4,len(mean_RUN)], labels = [-10,-5,0,5,10])
    ax0.set_xlabel('time (s)')
    ax0.set_ylabel('Population average running speed')

    ax1.plot(mean_TRACE, color = 'black')
    ax1.set_xticks(ticks = [0,len(mean_TRACE)/4,len(mean_TRACE)/2,(3*len(mean_TRACE))/4,len(mean_TRACE)], labels = [-10,-5,0,5,10])
    ax1.fill_between(np.arange(0,len(mean_TRACE)), mean_TRACE + sem(TRACE), mean_TRACE - sem(TRACE), alpha=0.3, color = 'grey')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Population average $\\delta$ $\\Delta$F/F \n n cell = ' + str(n_cell))
    fig.text(s=folder, x = 0.27, y= 0.22, color = color) 
    fig.text(s=summary_protocol, x = 0.27, y= 0.19) 
    fig.text(s=run_cond_name, x = 0.27, y= 0.16)
    if return_only_sign is not None : 
         fig.suptitle('Response of cells ' + str(return_only_sign) + ' significant to visual stim', fontsize = 18)

    #plot diff of amplitude bar graph 
    if plot_diff_pre_post : 
        t = summary[0]
        #ind_onset= len([x for x in data.t_dFoF[:350] if x < 10]) #350 just not to look in the whole list 
        #ind_onset_m2 = len([x for x in data.t_dFoF[:250] if x < 8])
        #ind_onset_p2 = len([x for x in data.t_dFoF[:450] if x < 12])
        ind_onset = (10*TRACE.shape[1])//20 - 2 #simplified not entirely true indices
        ind_onset_m2 = (8*TRACE.shape[1])//20
        ind_onset_p2 = (12*TRACE.shape[1])//20

        var_pre_post_sign = [np.mean(tr[ind_onset:ind_onset_p2]) - np.mean(tr[ind_onset_m2:ind_onset]) for tr in TRACE[sign_ROIs]]
        var_pre_post_non_sign = [np.mean(tr[ind_onset:ind_onset_p2]) - np.mean(tr[ind_onset_m2:ind_onset]) for tr in TRACE[~sign_ROIs]]


        fig = plt.figure(figsize = (3,5))
        plt.errorbar(x = [0,1], y = [np.mean(var_pre_post_sign), np.mean(var_pre_post_non_sign)], yerr =  [sem(var_pre_post_sign), sem(var_pre_post_non_sign)], fmt='.', elinewidth=1, capthick=1,  capsize = 5, color = 'black')
        plt.bar(x = [0,1], height = [np.mean(var_pre_post_sign), np.mean(var_pre_post_non_sign)], width = 0.7, color = color, alpha = 0.5)
        plt.scatter(0.7*(np.arange(0,len(var_pre_post_sign))/len(var_pre_post_sign)) - 0.35, var_pre_post_sign, s= 2, color = color)
        plt.scatter(0.7*(np.arange(0,len(var_pre_post_non_sign))/len(var_pre_post_non_sign)) + 0.65, var_pre_post_non_sign, s= 2,  color = color)
        plt.ylabel('Variation $\\Delta$F/F (post-pre)')
        plt.xticks( [0,1],  ['responsive \n cells','non responsive \n cells'])
        fig.text(s=folder, x = 0.1, y = 0.1)
        fig.text(s='error bar = sem', x = 0.1, y = 0.05)
        fig.text(s=summary_protocol,  x = 0.1, y = 0)
        fig.text(s=run_cond_name,  x = 0.1, y = -0.05)
        fig.text(s='RESP : n_cells = '+ str(np.sum(sign_ROIs)) + ' / mean = ' + str(np.round(np.mean(var_pre_post_sign),3)),  x = 0.1, y = -0.1)
        fig.text(s='NON RESP : n_cells = '+ str(np.sum(~sign_ROIs)) + ' / mean = ' + str(np.round(np.mean(var_pre_post_non_sign),3)),  x = 0.1, y = -0.15)
        plt.show()


def get_summary_info(summary) :

    if 'shifted_angle' in summary[0].keys() : 
        summary_portocol = 'Tunings'
        protocol_control_cond = 'contrast'
    elif 'contrast' in summary[0].keys() : 
        summary_portocol = 'Selectivities'
        protocol_control_cond = 'angle'

    path = os.path.normpath(summary[0]['datafile'])
    folder = path.split(os.sep)[-3]

    return summary_portocol, protocol_control_cond, folder 
