import numpy as np 
from physion.analysis.protocols.orientation_tuning import get_tuning_responses
from physion.analysis.protocols.contrast_sensitivity import get_responses


def session_sem_with_indepedance_hypothesis_universal(Summaries, average_by = 'sessions') : #done very detailed because easily confusable


    #compute sessions means sems
    sessions_sems = []
    if 'shifted_angle' in Summaries[0].keys():
        visual_stim = 'shifted_angle'
    elif 'contrast' in Summaries[0].keys():
        visual_stim = 'contrast'

    for s in Summaries :

        #compute sem of each ROI mean. ie ROI mean std due to eps. ie propagated eps std / square number of eps
        ROIs_sems = s['std-values'] / np.sqrt(s['ntrials']) #does not perform the same way for both summaries bc in tuning ntrials is two dim while in sensitivity it is 1 dim 

        #compute between-ROIs-variance-due-to-eps. ie the contributions of each ROIs-variance-due-to-eps to the between-ROI variance. is = to the mean of the variances due to eps. it is NOT the second propagation. bc we are in between ROIs not after ROIs mean 
        ROIs_variance_due_to_eps = np.nanmean(ROIs_sems**2, axis = 0) #mean of squared stds ie mean of variances ie is a variance

        #compute observed between ROIs variance
        observed_ROIs_variance = np.nanvar(s['Responses'], axis=0, ddof = 1)

        #compare observed-between-ROIs-variance to the mean of ROIs-variances-due-to-eps
        session_variance = np.nanmax((observed_ROIs_variance, ROIs_variance_due_to_eps), axis = 0)  #check for the nans propagation law
        #compute sem of the session mean 
        n_rois_per_visual_stim = [np.sum(np.isnan(s['Responses'][:,c]) == False) for c in range(s[visual_stim].shape[0])] #to account for nans in the following divisions
        session_sem = np.sqrt(session_variance) / np.sqrt(n_rois_per_visual_stim) #sqrt of the std divided by sqrt number of ROIs ie sem of the session
        
        sessions_sems.append(session_sem)

    sessions_sems = np.array(sessions_sems)


    #compute population mean sem 

    #compute observed-between-sessions-variance 
    if visual_stim == 'shifted_angle' : 
        Responses = np.array(get_tuning_responses(Summaries, average_by=average_by))
    elif visual_stim == 'contrast' : 
        Responses = np.array(get_responses(Summaries, average_by=average_by))

    observed_sessions_variance = np.nanvar(Responses, axis=0, ddof = 1)

    #compute between-sessions-variance-due-to-each-respective-sems ie the contributions of each session sem to the between-sessions variance. #is the mean of the previously computed squared sems (because sems is std and here we work with variance)
    sessions_variance_due_to_sems = np.nanmean(sessions_sems**2, axis = 0)

    #compare observed-between-sessions-variance to mean of sessions-variances computed before
    population_variance = np.nanmax((observed_sessions_variance, sessions_variance_due_to_sems), axis = 0)  #check for the nans propagation law

    #compute population sem
    n_sessions_per_visual_stim = [np.sum(np.isnan(Responses[:,c]) == False) for c in range(s[visual_stim].shape[0])] #to account for nans in the following divisions
    population_sem = np.sqrt(population_variance) / np.sqrt(n_sessions_per_visual_stim) #std divided by sqrt number of sessions ie sem of the session

    return population_sem