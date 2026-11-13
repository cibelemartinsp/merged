#%%
import os, sys , shutil 
import multiprocessing
import numpy as np

import physion
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import physion.utils.plot_tools as pt

sys.path += ['./physion/src']
from physion.analysis.read_NWB\
                         import scan_folder_for_NWBfiles, Data
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.protocols.orientation_tuning import *


from scipy.signal import correlation_lags, correlate 
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import gaussian_filter1d

#%%

"""
1) plot percentage of responsiveness of the population 

#-----------------------------------------Cells responsivness profiles (to visual and crosscorelation to behavior)------------------------------------------------------#

2) compute for one session the crosscorelation of every ROI traces with all 3 behaviors + some responsivness metrics

3) get_population_responsiveness_profiles : use the fonction above, organise in a dict all sessions responsiveness profiles 

4) print relevant percentages of the above dict 

5) plot bar graph of the crosscorelation coefficient by resp/non resp (option with session mean or no)

6) plot a function of the % responsivness of sessions vs the mean crosscorelation coefficient of the session

#-----------------------------------------Responses's pval to tuning distribution ------------------------------------------------------#


7) get the pvalues of every ROI of every sessions and plot the distribution 

8) modified computing_tuning_response_per_cells to pass the pvalues


#-----------------------------------Effect of interval pre/post on nb significatif ROIs-----------------------------------------#

9) plot % of responsivness at different post_stimuli intervals 
"""



#%%

def plot_percentage_of_responsiveness(folder, summary_folder, base_path, colors) :
    keys = ['%s_contrast-1.0' % folder, 
                '%s_contrast-0.5' % folder]

    mean_perc_responsive = []
    plt.figure(figsize = (6,7))
    for i,key in enumerate(keys) :
        Tunings = np.load(os.path.join(summary_folder,'mccf_Tunings_%s.npy' % key), allow_pickle=True)
        perc_responsives = [np.sum(Tuning['significant_ROIs'])/len(Tuning['significant_ROIs']) for Tuning in Tunings]
        mean_perc_responsive.append(perc_responsives)

        plt.bar([i], np.mean(perc_responsives), color = colors[i], alpha = 0.5)
        plt.errorbar([i],np.mean(perc_responsives), yerr = np.nanstd(perc_responsives), fmt='.', elinewidth=1, capthick=1,  capsize = 5, color = 'black')

        x = (np.arange(0,len(perc_responsives))-(len(perc_responsives)/2))/(2.5*len(perc_responsives))
        plt.scatter(x+i, perc_responsives, color = colors[i])
        plt.text(x = i+0.02, y= np.mean(perc_responsives)+0.01, s= str(np.round(np.mean(perc_responsives),2)))



    mean_perc_responsive = [(a+b)/2 for a,b in zip(mean_perc_responsive[0],mean_perc_responsive[1])]
    plt.bar([3], np.mean(mean_perc_responsive), color = colors[0], alpha = 0.5)
    plt.errorbar([3],np.mean(mean_perc_responsive), yerr = np.nanstd(mean_perc_responsive), fmt='.', elinewidth=1, capthick=1,  capsize = 5, color = 'black')

    x = (np.arange(0,len(mean_perc_responsive))-(len(mean_perc_responsive)/2))/(2.5*len(mean_perc_responsive))
    plt.scatter(x+3, mean_perc_responsive, color = colors[0])
    plt.text(x = 3+0.02, y= np.mean(mean_perc_responsive)+0.01, s= str(np.round(np.mean(mean_perc_responsive),2)))


    plt.xticks(ticks = [0,1,3], labels = ['contrast = 1.0', 'contrast = 0.5', 'all contrasts'], rotation = 45)
    plt.ylabel('Percentage of responsive ROIs by sessions')
    plt.ylim((0,1))
    plt.text(x = -0.75, y= -0.25, s = folder)
    plt.text(x = -0.75, y= -0.30, s = 'multiple comparaison correction = False')
    plt.show()


#%%#-----------------------------------------Cells responsivness profiles (to visual and crosscorelation to behavior)------------------------------------------------------#

def get_session_responsiveness_profile(summary, ses_number, corr_coeff_funct = spearmanr):

    
    #build the session
    #filename = '/home/user/DATA/Astrid/Cibele_data/' + Tunings[ses_number]['datafile'][26:]
    #filename = '/home/user/DATA/Astrid/Cibele_data/' + Tunings[ses_number]['datafile'][26:]
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
    data.pupil= data.build_pupil(specific_time_sampling = data.t_dFoF)
    data.facemotion = data.build_facemotion(specific_time_sampling = data.t_dFoF)

    #compute responsivness to behavior
    behaviors = [data.running]
    behaviors_names =  ['running_speed','pupil_diameter','facemotion']
    responsiveness_to_behavior = []
    for j,behavior_ref in enumerate(behaviors) : 
        if behavior_ref is not None : 

            #compute ROIs traces lags 
            lags = []
            for i in range(len(data.dFoF)) : 
                x = behavior_ref
                y = data.dFoF[i]
                lags.append(correlation_lags(len(x), len(y))[np.argmax(correlate(x,y))])

            #compute shuffle behavior (to check pvalue coherence)
            rng = np.random.default_rng()
            behavior_ref_shuffled = rng.permutation(behavior_ref)


            #compute spearmanr corr coeff between shifted ROIs traces and behavior ref with/without shuffling
            spearcoeffs_withshifts = []
            spearcoeffs_withshifts_shuffled = []
            for i in range(len(data.dFoF)) : 
                #df = gaussian_filter1d(np.roll(data.dFoF[i], shift = lags[i]), sigma = 2)
                df = np.roll(data.dFoF[i], shift = lags[i])
                spearcoeffs_withshifts.append(corr_coeff_funct(behavior_ref, df))
                spearcoeffs_withshifts_shuffled.append(corr_coeff_funct(behavior_ref_shuffled, df))
            responsiveness_to_behavior.append(np.where(np.array(spearcoeffs_withshifts)[:,1] < 0.001)[0])
        
        else : 
            responsiveness_to_behavior.append(np.array([np.nan]))
            print(filename)
            print(behaviors_names[j])
            print('--------------')

    #get responsiveness to visual stim
    if 'selectivities' in summary[0].keys() :  
        responsiveness_to_stim = np.where(summary[ses_number]['significant_ROIs'] == True)[0]
    elif 'contrast' in summary[0].keys() : 
        responsiveness_to_stim = np.where(np.sum(summary[ses_number]['significant_pos'],axis = 1) + np.sum(summary[ses_number]['significant_neg'],axis = 1) != 0)[0] #can have a percentage > 100 if cells are sign neg for some contrast and sign pos for an other but i don't think that's possible (?)
    #get all cells 
    all_cells = np.arange(len(data.dFoF))

    #get dead cells
    resp_cells = np.unique(np.concatenate([np.concatenate(responsiveness_to_behavior),responsiveness_to_stim]))
    no_resp_cells = [x for x in all_cells if x not in resp_cells]

    ses_profile = [all_cells, no_resp_cells, responsiveness_to_stim, responsiveness_to_behavior]

    return ses_profile, spearcoeffs_withshifts

def get_population_responsiveness_profiles(folder, corr_coeff_funct = pearsonr, summary_protocol = 'Tunings', absolute_values = True) :

    """
    By default compute responsiveness to Tuning protocol
    """

    PROFILES = [[],[],[],[]]
    CORR_COEFFS = [[],[],[]]

    if summary_protocol == 'Tunings' :
        visual_control = 'contrast'
        visual_control_values = [0.5,1.0] # [0.,90.]

    for visual_control_val in visual_control_values: 
        summary = np.load('/home/user/DATA/Astrid/run_rest_summary/' + summary_protocol + '_' + folder + '_' + visual_control + '-' +  str(visual_control_val) + '.npy', allow_pickle=True)  

        for ses_number in range(len(summary)) : 
            print(ses_number)
            ses_profile, corr_coeff = get_session_responsiveness_profile(summary, ses_number, corr_coeff_funct = corr_coeff_funct)
            for i in range(4) : 
                PROFILES[i].append(ses_profile[i])

            CORR_COEFFS[0].append(visual_control_val)
            if absolute_values : 
                CORR_COEFFS[1].append(np.abs(np.array(corr_coeff)[:,0]))
            else : 
                CORR_COEFFS[1].append(np.array(corr_coeff)[:,0])

            if visual_control == 'angle' :
                sign_pos_and_neg = np.sum(summary[ses_number]['significant_pos'], axis = 1) + np.sum(summary[ses_number]['significant_neg'], axis = 1)
                CORR_COEFFS[2].append(sign_pos_and_neg.astype(np.bool))
            elif visual_control == 'contrast' : 
                CORR_COEFFS[2].append(summary[ses_number]['significant_ROIs'])


    resp_profiles = {'all_cells' : PROFILES[0],'no_resp_cells' : PROFILES[1],'visual_resp_cells': PROFILES[2], 'beh_resp_cells': PROFILES[3]}
    corr_dict = {visual_control : CORR_COEFFS[0], 'corr coeffs' : CORR_COEFFS[1], 'signficant ROIs' : CORR_COEFFS[2]}

    return resp_profiles, corr_dict

def print_population_responsiveness_relevant_percentages(resp_profiles, folder, summary_protocol) :
    n_ses = len(resp_profiles['all_cells'])
    perc_no_resp = [len(resp_profiles['no_resp_cells'][i])/len(resp_profiles['all_cells'][i]) for i in range(n_ses)]
    per_visual_resp = [len(resp_profiles['visual_resp_cells'][i])/len(resp_profiles['all_cells'][i]) for i in range(n_ses)]
    #to avoid division by zero 
    per_visual_resp_and_beh_resp = []
    for i in range(n_ses) :
        if len(resp_profiles['visual_resp_cells'][i])  != 0 : 
            per_visual_resp_and_beh_resp.append( len(np.intersect1d(resp_profiles['visual_resp_cells'][i],np.concatenate(resp_profiles['beh_resp_cells'][i])))/len(resp_profiles['visual_resp_cells'][i]) )
        else : 
            per_visual_resp_and_beh_resp.append(np.nan)

    #per_visual_resp_and_beh_resp = [len(np.intersect1d(resp_profiles['visual_resp_cells'][i],np.concatenate(resp_profiles['beh_resp_cells'][i]))) /len(resp_profiles['visual_resp_cells'][i]) for i in range(n_ses)]
    per_resp_all_beh =  [len(np.intersect1d(np.intersect1d(resp_profiles['beh_resp_cells'][i][0],resp_profiles['beh_resp_cells'][i][1]),resp_profiles['beh_resp_cells'][i][2]))/ len(resp_profiles['all_cells'][i]) for i in range(n_ses)]

    per_resp_to_run = [len(resp_profiles['beh_resp_cells'][i][0])/ len(resp_profiles['all_cells'][i]) for i in range(n_ses)]
    per_resp_to_pupil = [len(resp_profiles['beh_resp_cells'][i][1])/ len(resp_profiles['all_cells'][i]) for i in range(n_ses)]
    per_resp_to_facemotion = [len(resp_profiles['beh_resp_cells'][i][2])/ len(resp_profiles['all_cells'][i]) for i in range(n_ses)]

    print(folder)
    print('protocol = '  + summary_protocol)
    print()
    print('mean percentage no resp ' + str(np.mean(perc_no_resp)))
    print('total number of no resp cells : ' + str(np.sum([len(resp_profiles['no_resp_cells'][i]) for i in range(int(n_ses/2))])) + ' out of ' + str(np.sum([len(resp_profiles['all_cells'][i]) for i in range(int(n_ses/2))])))
    print('mean percentage visual resp ' +str(np.mean(per_visual_resp)))
    print('mean percentage visual resp contrast 0.5 : ' +str(np.mean(per_visual_resp[:int(len(per_visual_resp)/2)])))
    print('mean percentage visual resp contrast 1.0 :' +str(np.mean(per_visual_resp[int(len(per_visual_resp)/2):])))
    print('mean percentage visual resp and behavior resp ' + str(np.nanmean(per_visual_resp_and_beh_resp)))
    print('mean percentage resp to all 3 behaviors ' + str(np.mean(per_resp_all_beh)))
    print('mean percentage resp run ' + str(np.mean(per_resp_to_run)))
    print('mean percentage resp pupil ' + str(np.mean(per_resp_to_pupil)))
    print('mean percentage resp facemotion ' + str(np.mean(per_resp_to_facemotion)))


def plot_corr_coeffs_distribution_between_sign_non_sign(corr_dict, visual_control = 'contrast', visual_control_values = [0.5,1.0], plot_by_roi = False) :

    corr_coeffs_categ = [[],[],[],[]]
    if plot_by_roi == True : 
        for i in range(len(corr_dict[visual_control])) : 
            if corr_dict[visual_control][i] == visual_control_values[0] :
                corr_coeffs_categ[0].append(corr_dict['corr coeffs'][i][corr_dict['signficant ROIs'][i]])
                corr_coeffs_categ[1].append(corr_dict['corr coeffs'][i][~corr_dict['signficant ROIs'][i]])
            elif corr_dict[visual_control][i] == visual_control_values[1] :
                corr_coeffs_categ[2].append(corr_dict['corr coeffs'][i][corr_dict['signficant ROIs'][i]])
                corr_coeffs_categ[3].append(corr_dict['corr coeffs'][i][~corr_dict['signficant ROIs'][i]])
        for i in range(4):
            corr_coeffs_categ[i] = np.concatenate(corr_coeffs_categ[i])


    else:
        for i in range(len(corr_dict[visual_control])) : 
            if corr_dict[visual_control][i] == visual_control_values[0] :
                corr_coeffs_categ[0].append(np.nanmean(corr_dict['corr coeffs'][i][corr_dict['signficant ROIs'][i]]))
                corr_coeffs_categ[1].append(np.nanmean(corr_dict['corr coeffs'][i][~corr_dict['signficant ROIs'][i]]))
            elif corr_dict[visual_control][i] == visual_control_values[1] :
                corr_coeffs_categ[2].append(np.nanmean(corr_dict['corr coeffs'][i][corr_dict['signficant ROIs'][i]]))
                corr_coeffs_categ[3].append(np.nanmean(corr_dict['corr coeffs'][i][~corr_dict['signficant ROIs'][i]]))

    #plot
    if visual_control == 'contrast' : 
        labels = ['positive visual \n contrast 0.5','negative visual  \n contrast 0.5','positive visual  \n contrast 1.0','negative visual  \n contrast 1.0']
    elif visual_control == 'angle' : 
        labels = ['positive visual \n angle 0.0','negative visual  \n angle 0.0','positive visual  \n angle 90.0','negative visual  \n angle 90.0']

    MEANS = []
    fig, ax = plt.subplots(figsize = (7,7))
    colors = ['darkkhaki','lightsteelblue']*2
    alpha_par = [0,0,0,0]
    #y_par = [-0.15,-0.18, - 0.21,-0.24,-0.27]
    y_par = [-0.08,-0.10, - 0.12,-0.14,-0.16]
    #plot all ROIs
    for i in range(4) : 
        mean_val = np.nanmean(corr_coeffs_categ[i])
        ax.bar([i], mean_val, width = 0.5, color = colors[i], alpha = 0.4 )
        #ax.errorbar([i], mean_val, yerr = stats.sem(np.abs(corr_coeffs_categ[i])), fmt='.', elinewidth=1, capthick=1,  capsize = 5, color = 'black')
        ax.errorbar([i], mean_val, yerr = np.nanstd(corr_coeffs_categ[i]), fmt='.', elinewidth=1, capthick=1,  capsize = 5, color = 'black')
        x = (np.arange(0,len(corr_coeffs_categ[i]))-(len(corr_coeffs_categ[i])/2))/(2.5*len(corr_coeffs_categ[i]))
        ax.scatter(x+i, corr_coeffs_categ[i], alpha = 0.7, color = colors[i])
        round_mean = np.round(mean_val,2)
        ax.annotate(text = str(round_mean), xy = (i+0.02, round_mean +0.01))


    ax.set_ylabel(corr_coeff_funct.__name__ + ' sessions crosscorealtion coefficients \n with running')
    ax.set_xticks([0,1,2,3], labels = labels, rotation = 45)
    ax.set_xticks([0.2,1.2,2.2,3.2], labels = labels, rotation = 45)

    plt.text(s = 'SST-cells_cond-GluN1-KO_Adult_V1' , x = -1, y = y_par[0])
    plt.text(s = 'protocol = '  + summary_protocol , x = -1, y = y_par[1])
    plt.text(s = 'correlation function : ' + str(corr_coeff_funct.__name__), x = -1, y =y_par[2])
    plt.text(s = "sessions averages of absolute crosscorelation coefficients between rois traces and running", x = -1, y = y_par[3])
    plt.text(s = "error bar = std", x = -1, y = y_par[4])
    plt.show()

def plot_correlation_functof_responsiveness(corr_dict, folder, colors):
    mean_corr_coeffs = []
    perc_resp = []

    for i in range(len(corr_dict['contrast'])): 
        mean_corr_coeffs.append(np.mean(corr_dict['corr coeffs'][i]))
        perc_resp.append(np.sum(corr_dict['signficant ROIs'][i])/ len(corr_dict['signficant ROIs'][i]))

    plt.figure(figsize=(7,7))
    plt.scatter(mean_corr_coeffs[:14], perc_resp[:14], color = colors[0], lw = 2)
    plt.scatter(mean_corr_coeffs[14:], perc_resp[14:], color = colors[1], lw = 2)


    plt.ylabel("Sessions's" +  ' % ' + "of responsiveness")
    plt.xlabel("Sessions's mean correlation with running")
    grey_patch = mpatches.Patch(color=colors[0], label='contrast 0.5')
    green_patch = mpatches.Patch(color=colors[1], label='contrast 1.0')
    plt.legend(handles=[grey_patch, green_patch])

    plt.ylim((0,1))
    plt.text(s = folder, x = 0, y = -0.2)
    plt.text(s = 'Multiple comparaison correction = False', x = 0, y = -0.25)


    plt.show()

#%%#-----------------------------------------Responses's pval to tuning distribution ------------------------------------------------------#

def get_pval_of_tuning_response(folder, summary_folder, plot_distribution = True, color = ['lightgrey', pt.tab10(0)]) : 

    dFoF_parameters = dict(\
            roi_to_neuropil_fluo_inclusion_factor=1.15,
            neuropil_correction_factor = 0.7,
            method_for_F0 = 'sliding_percentile',
            percentile=5., # percent
            sliding_window = 5*60, # seconds
    )

    response_significance_threshold=5e-2
    stat_test_props=dict(interval_pre=[-1.,0],
                            interval_post=[1.,2.],                                   
                            test='ttest',                                            
                            sign='positive')

    nMin_episodes = 2
    start_angle=-22.5
    angle_range=180

    PVAL_pref_angle, PVAL_min, iANGLE_of_pval_min = [[],[]], [[],[]], [[],[]]

    Tunings = np.load(summary_folder + '/Tunings_' + folder + '_contrast-1.0.npy', allow_pickle=True)   

    for ses_number in range(len(Tunings)) : 
        print(ses_number)
        filename = '/home/user/DATA/Astrid/Cibele_data/' + Tunings[ses_number]['datafile'][35:]
        #filename = '/home/user/DATA/Astrid/Cibele_data/' + Summaries[ses_number]['datafile'][26:]

        data = physion.analysis.read_NWB.Data(filename, verbose=False) 
        data.build_dFoF(**dFoF_parameters, verbose=False)

        Episodes = EpisodeData(data, 
                                quantities=['dFoF'], 
                                protocol_name = data.protocols[0],
                                verbose=False)
        
        for i,contrast in enumerate([0.5,1.0]) : 


            summary = compute_tuning_response_per_cells_pval(data, Episodes, 
                                                quantity='dFoF', 
                                                stat_test_props = stat_test_props, 
                                                response_significance_threshold =\
                                                    response_significance_threshold, 
                                                contrast = contrast)

            prefered_angles_shifted = np.array([shift_orientation_according_to_pref(r, pref_angle=-start_angle,
                                                start_angle=start_angle,
                                                angle_range=angle_range)\
                                                for r in summary['prefered_angles']])

            iangles = [np.argwhere(prefered_angles_shifted[roi] == summary['shifted_angle'])[0][0] for roi in range(summary['pval'].shape[0])]
            PVAL_pref_angle[i].append([summary['pval'][roi,iangles[roi]] for roi in range(summary['pval'].shape[0])])
            PVAL_min[i].append([np.min(summary['pval'][roi,:]) for roi in range(summary['pval'].shape[0])])
            iANGLE_of_pval_min[i].append([np.argmin(summary['pval'][roi,:]) for roi in range(summary['pval'].shape[0])])

            if plot_distribution : 
                
                plt.figure(figsize=(7,7))
                for ses_number in range(len(Tunings)) : 

                    for icontrast in range(2) : 
                        y = PVAL_min[icontrast][ses_number]
                        x = np.arange(len(y))/len(y)
                        plt.scatter(x+(icontrast*2), y, color = color[icontrast])

                plt.hlines(y = 0.05, xmin = 0, xmax = 3)
                plt.xticks([0.5,2.5], labels = ['contrast 0.5', 'contrast 1.0'], rotation = 45)
                #plt.ylabel('ROIs pvalues at prefered orientations')
                plt.ylabel('ROIs min pvalues')
                plt.ylim((0,0.62))
                plt.text(s = folder, x = -0.2, y = -0.2)
                #plt.text(s = 'multiple_comparison_correction = False', x = -0.2, y = -0.2)
                nROIs = len(np.concatenate(PVAL_min[0]))
                plt.text(s = 'total nROIs = ' + str(nROIs) , x = -0.2, y = -0.25 )

        return PVAL_pref_angle, PVAL_min, iANGLE_of_pval_min
        

def compute_tuning_response_per_cells_pval(data, Episodes,
                                      stat_test_props,
                                      response_significance_threshold = 0.05,
                                      filtering_cond=None,
                                      quantity='dFoF',
                                      contrast=1.0,
                                      nMin_episodes = 2,
                                      start_angle=-22.5, 
                                      angle_range=180,
                                      verbose=False):
    """

    All cells are considered in this analysis !!
      --> think about filtering them by resp['significant_ROIs'] when needed !!

    """

    shifted_angle = np.array(\
        [shift_orientation_according_to_pref(r, pref_angle=-start_angle,
                                             start_angle=start_angle,
                                             angle_range=angle_range)\
                    for r in Episodes.varied_parameters['angle']])

    if verbose:
        print('  - shifted_angle correspond to : ', shifted_angle)

    if filtering_cond is None:
        filtering_cond = Episodes.find_episode_cond() # True everywhere

    cond = Episodes.find_episode_cond(key='contrast', 
                                        value=contrast) &\
                                        filtering_cond
    
    summary = Episodes.pre_post_statistics(stat_test_props = stat_test_props,
                                                episode_cond=cond,
                                                repetition_keys=['repeat', 'contrast'],
                                                response_args=dict(quantity=quantity),
                                                response_significance_threshold=response_significance_threshold,
                                                multiple_comparison_correction=False,
                                                loop_over_cells=True,
                                                nMin_episodes = nMin_episodes,
                                                verbose=verbose)
        
    # if significant in at least one orientation
    significant = (np.sum(summary['significant'], axis=1)>0)

    # find preferred angle:
    ipref = np.argmax(summary['value'], axis=1).flatten()
    #print(ipref)

    prefered_angles = np.array(\
            [summary['angle'][i] for i in ipref])

    selectivities = np.array([\
        selectivity_index(summary['angle'],
                          summary['value'][roi, :])\
                            for roi in range(data.nROIs)])

    RESPONSES, semRESPONSES, Ntrials, Pval = [], [], [], []
    for roi in range(data.nROIs):

        RESPONSES.append(np.zeros(len(shifted_angle)))
        semRESPONSES.append(np.zeros(len(shifted_angle)))
        Ntrials.append(np.zeros(len(shifted_angle)))
        Pval.append(np.zeros(len(shifted_angle)))

        for angle, value, std, ntrials, pval in zip(\
            summary['angle'],
            summary['value'][roi,:], 
            summary['std-value'][roi,:],
            summary['ntrials'],
            summary['pval'][roi,:]):

            new_angle = shift_orientation_according_to_pref(angle,
                                                    pref_angle=prefered_angles[roi],
                                                    start_angle=start_angle,
                                                    angle_range=angle_range)
            iangle = np.flatnonzero(shifted_angle==new_angle)[0]

            RESPONSES[-1][iangle] = value
            semRESPONSES[-1][iangle] = std/np.sqrt(ntrials)
            Ntrials[-1][iangle] = ntrials
            Pval[-1][iangle] = pval 


    return {'Responses':np.array(RESPONSES),
            'semResponses':np.array(semRESPONSES),
            'selectivities':np.array(selectivities),
            'shifted_angle':np.array(shifted_angle),
            'prefered_angles':np.array(prefered_angles),
            'significant_ROIs':np.array(significant),
            'std-values':summary['std-value'], 
            'ntrials': np.array(Ntrials[0]),
            'pval' : np.array(Pval)}

#%% #-----------------------------------Effect of interval pre/post on nb significatif ROIs-----------------------------------------#


def plot_responsiveness_at_different_post_stimuli_interval(folder, summary_folder,plot = True): 

    visual_control_values = [0.5,1.0]
    Summaries = np.load(summary_folder + '/Tunings_' + folder + '_contrast-0.5.npy', allow_pickle=True)   

    dFoF_parameters = dict(\
            roi_to_neuropil_fluo_inclusion_factor=1.15,
            neuropil_correction_factor = 0.7,
            method_for_F0 = 'sliding_percentile',
            percentile=5., # percent
            sliding_window = 5*60, # seconds
    )

    response_significance_threshold=5e-2
    stat_test_props=dict(interval_pre=[-1.,0],
                            interval_post=[1.,2.],                                   
                            test='ttest',                                            
                            sign='positive')

    list_intervals = [[[-1.,0],[1.,2.]],
                    [[-1.,0],[0.,1.]],
                    [[-1.,0],[0,0.5]]]

    NB_responsive_control_cond1 = []
    NB_responsive_control_cond2 = []

    for intervals in list_intervals : 
        
        nb_responsive_control_cond = [[],[]]

        stat_test_props=dict(interval_pre=intervals[0],
                            interval_post=intervals[1],                                   
                            test='ttest',                                            
                            sign='positive')

        for ses_number in range(len(Summaries)) : 

            filename = '/home/user/DATA/Astrid/Cibele_data/' + Summaries[ses_number]['datafile'][35:]
            #filename = '/home/user/DATA/Astrid/Cibele_data/' + Summaries[ses_number]['datafile'][26:]

            data = physion.analysis.read_NWB.Data(filename, verbose=False) 
            data.build_dFoF(**dFoF_parameters, verbose=False)

            Episodes = EpisodeData(data, 
                                    quantities=['dFoF'], 
                                    protocol_name = data.protocols[0],
                                    verbose=False)
        
            for i,visual_control_val in enumerate(visual_control_values) : 

                summary = compute_tuning_response_per_cells(data, Episodes, 
                                                    quantity='dFoF', 
                                                    stat_test_props = stat_test_props, 
                                                    response_significance_threshold =\
                                                        response_significance_threshold, 
                                                    contrast = visual_control_val)
                nb_responsive_control_cond[i].append(np.sum(summary['significant_ROIs'])/len(summary['significant_ROIs']))


        NB_responsive_control_cond1.append(nb_responsive_control_cond[0])
        NB_responsive_control_cond2.append(nb_responsive_control_cond[1])

    #print perc visual resp values
    print(folder)
    print('Selectivities')
    print('multiple comparaison correction = False')
    print()
    for i in range(3):
        print('mean percentage visual resp with interval' + str(list_intervals[i][1]) + ' = ' + str(np.round(np.mean(np.concatenate((NB_responsive_control_cond1[i],NB_responsive_control_cond2[i]))),3)))

    for i in range(3):
        print()
        print('with interval : ' + str(list_intervals[i][1]) )
        print('mean percentage visual resp = ' + str(np.round(np.mean(np.concatenate((NB_responsive_control_cond1[i],NB_responsive_control_cond2[i]))),3)))

        print('mean percentage visual resp ' + visual_control + ' ' + str(visual_control_values[0]) + ' = '  + str(np.round(np.mean(NB_responsive_control_cond1[i]),3)))
        print('mean percentage visual resp ' + visual_control + ' ' + str(visual_control_values[1]) + ' = '  + str(np.round(np.mean(NB_responsive_control_cond2[i]),3)))
        print()
        print('--------')
        print()

    if plot : 

        colors=['lightgrey', 'lightgrey', 'lightgrey', pt.tab10(i), pt.tab10(i), pt.tab10(i)]
        fig, ax = plt.subplots(figsize = (7,7))
        labels = [str(list_intervals[0][1]), str(list_intervals[1][1]), str(list_intervals[2][1]),str(list_intervals[0][1]), str(list_intervals[1][1]), str(list_intervals[2][1])]
        NB_responsive = np.concatenate((NB_responsive_control_cond1,NB_responsive_control_cond2))
        alphas = [0.35,0.5,0.65]*2
        for i in range(6) : 

            mean_val = np.round(np.mean(NB_responsive[i]),3)
            ax.bar([0,1,2,4,5,6][i], mean_val, width = 0.6, color = colors[i], alpha = alphas[i])
            #ax.errorbar([i], mean_val, yerr = stats.sem(np.abs(corr_coeffs_categ[i])), fmt='.', elinewidth=1, capthick=1,  capsize = 5, color = 'black')
            ax.errorbar([0,1,2,4,5,6][i], mean_val, yerr = np.std(NB_responsive[i]), fmt='.', elinewidth=1, capthick=1,  capsize = 5, color = 'black')
            x = (np.arange(0,len(NB_responsive[i]))-(len(NB_responsive[i])/2))/(2.5*len(NB_responsive[i]))
            ax.scatter(x+[0,1,2,4,5,6][i], NB_responsive[i], alpha = alphas[i]+0.35, color = colors[i])
            round_mean = mean_val
            ax.annotate(text = str(round_mean), xy = ([0,1,2,4,5,6][i]+0.02, round_mean +0.01))

        ax.set_xticks([0,1,2,4,5,6], labels = ['post interval = \n' + str(list_intervals[0][1]), 'post interval = \n' + str(list_intervals[1][1]), 'post interval = \n' + str(list_intervals[2][1])]*2, rotation = 45)

        ax.set_ylabel('Percentage of responsive cells per session')
        plt.text(s = visual_control + ' ' + str(visual_control_values[0]) , x = 0.75, y = -0.28)
        plt.text(s = visual_control + ' ' + str(visual_control_values[1])  , x = 4.75, y = -0.28)
        plt.text(s = folder, x = -0.5, y = -0.4)
        plt.text(s = 'protocol = '  + summary_protocol , x = -0.5, y = -0.45)
        plt.text(s = 'multiple comparaison correction = False', x = -0.5, y = -0.5)

        plt.show()


