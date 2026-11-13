#%%






def plot_bar_comparaison_average_run_triggered_dfof(folders) : 


    plt.figure(figsize = (5,7))

    time_from_other_events_cond = 0
    for i,folder in enumerate(folders[1:]) : 
        summary_protocol = 'Tunings'
        protocol_control_cond = 'contrast-1.0'
        summary = np.load('/home/user/DATA/Astrid/run_rest_summary/' + summary_protocol + '_' + folder + '_' + protocol_control_cond + '.npy', allow_pickle=True)  

        summary_portocol, protocol_control_cond = get_summary_info(summary)

        RUN, TRACE, ses_included = get_run_triggered_dfof_and_run(folder, summary, time_from_other_events_cond = time_from_other_events_cond) 
        sign_ROIs = np.concatenate([summary[ses_number]['significant_ROIs'] for ses_number in ses_included])
        TRACE = np.array(np.concatenate(TRACE))
        mean_TRACE = np.mean(TRACE, axis = 0)
        n_cell = len(TRACE)

        
        ind_onset = (10*TRACE.shape[1])//20 - 2
        ind_onset_m2 = (8*TRACE.shape[1])//20
        ind_onset_p2 = (12*TRACE.shape[1])//20

        var_pre_post_sign = [np.mean(tr[ind_onset:ind_onset_p2]) - np.mean(tr[ind_onset_m2:ind_onset]) for tr in TRACE[sign_ROIs]]

        color = pt.tab10(i+1)
        plt.errorbar(x = [i], y = [np.mean(var_pre_post_sign)], yerr =  [sem(var_pre_post_sign)], fmt='.', elinewidth=1, capthick=1,  capsize = 5, color = 'black')
        plt.bar(x = [i], height = [np.mean(var_pre_post_sign)], width = 0.7, color = color, alpha = 0.5)
        plt.scatter(0.7*(np.arange(0,len(var_pre_post_sign))/len(var_pre_post_sign)) - 0.35 + i, var_pre_post_sign, s= 2, color = color)
        plt.text(s='n_cells = '+ str(np.sum(sign_ROIs)) + ' / mean = ' + str(np.round(np.mean(var_pre_post_sign),3)),  x = -0.5 , y = -0.-0.1*i, color = color)
    
    plt.ylabel('Variation $\\Delta$F/F (post-pre)')
    plt.xticks( [0,1],  ['responsive cells \n ' + str(folders[0]),'responsive cells \n ' + str(folders[1])])
    #plt.text(s=folder, x = -0.5, y = -1.1)
    plt.text(s='error bar = sem', x = -0.5, y = -0.6)
    plt.text(s=summary_protocol,  x = -0.5, y = -0.7)
    plt.text(s='time_from_other_events_cond = ' + str(time_from_other_events_cond),  x = -0.5, y = -0.8)
    #plt.text(s='error bar = sem', x = -0.5, y = -1.1)
    #plt.text(s=summary_protocol,  x = -0.5, y = -1.2)
    #plt.text(s='time_from_other_events_cond = ' + str(time_from_other_events_cond),  x = -0.5, y = -1.3)
    #plt.text(s='RESP : n_cells = '+ str(np.sum(sign_ROIs)) + ' / mean = ' + str(np.round(np.mean(var_pre_post_sign),3)),  x = -0.5, y = -1.5)
    #plt.text(s='NON RESP : n_cells = '+ str(np.sum(~sign_ROIs)) + ' / mean = ' + str(np.round(np.mean(var_pre_post_non_sign),3)),  x = -0.5, y = -1.6)
    plt.tight_layout()
    plt.show()
