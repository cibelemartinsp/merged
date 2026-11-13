import numpy as np
from general_tools.general_tools import get_protocol_keys
from physion.analysis.read_NWB import scan_folder_for_NWBfiles, Data


def check_presence_locomotion_values(data):
    if np.sum(data.running) == 0 : 
        raise ValueError('All locomotion values equal zero in this file')

        
def compute_arousal_masks(Episodes, speed_thr = 0.5) : 

    n_ep = len(Episodes.repeat)
    temporal_mask = (0 <= Episodes.t) & (Episodes.t <= Episodes.time_duration[0]) #only consider locomotion during the visual stimuli

    run_mask = [np.mean(Episodes.running[ep][temporal_mask]) >= speed_thr for ep in range(n_ep)]
    run_mask = np.array(run_mask).astype(bool)
    rest_mask = ~run_mask

    return run_mask, rest_mask 


def get_perc_rest_from_rest_summary(summary_filenames, summary_protocol) :
    #could be bettered with automatic summary protocol detection => would allow to pass filenames with different protocols
    means = []
    stds = []
    n_trials = 15*8 #hard coded for tuning and contrast protocol..
    keys = get_protocol_keys(summary_protocol = summary_protocol)

    for summary_filename in summary_filenames : 
        
        for j, key in enumerate(keys):
            rest_summaries = np.load(summary_filename + key + '.npy', allow_pickle = True )
    
            sessions_perc_rest = [np.sum(r_s['ntrials'][0])/n_trials for r_s in rest_summaries] #calcul for roi 0 cause all rois have the same nb of episode filtered 
            means.append(np.round(np.nanmean(sessions_perc_rest)*100,1))
            stds.append(np.round(np.nanstd(sessions_perc_rest)*100,1))

    return means, stds



def compute_perc_rest_from_nwb(folder, base_path, protocol = 'Tunings', compute = 'time', speed_thr = 0.5) :

    if protocol == 'Tunings' :
        control_cond = '8orientation'
    if protocol == 'Sensitivities' :
        control_cond = '8contrasts'

    DATASET = scan_folder_for_NWBfiles(base_path + '/' + folder + '/NWBs')
    filenames = [DATASET['files'][i] for i in range(len(DATASET['files'])) if control_cond in DATASET['protocols'][i][0]]

    perc_rest = []

    for filename in filenames :

        data = physion.analysis.read_NWB.Data(filename, verbose=False) 
        data.build_running()
        check_presence_locomotion_values(data)

        Episodes = EpisodeData(data, 
                                quantities=['running'], 
                                protocol_name = data.protocols[0],
                                verbose=False)

        if compute == 'time' :
            #percentage of time running during the entire session
            perc_rest.append(np.round((np.sum([data.running >= speed_thr]) * 100 )/ len(data.running),2))

        if compute == 'episodes' :
            #percentage of episodes considered runned
            perc_rest.append(np.round((np.sum(compute_arousal_masks(Episodes, speed_thr = speed_thr)[0]) *100) / len(Episodes.repeat),2))

    return perc_rest