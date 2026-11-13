#%%


Tunings1_KO = np.load('/home/user/DATA/Astrid/Cibele_data/summary/both_signi_Tunings_SST-cells_cond-GluN1-KO_Adult_V1_contrast-1.0.npy', allow_pickle = True)

Tunings1_KO_2 = np.load('/home/user/DATA/Astrid/Cibele_data/summary/2both_signi_Tunings_SST-cells_cond-GluN1-KO_Adult_V1_contrast-1.0.npy', allow_pickle = True)



#%% load Tuning
contrast = 1.0
Tunings = np.load('/home/user/DATA/Astrid/Cibele_data/summary/2both_signi_Tunings_SST-cells_cond-GluN1-KO_Adult_V1_contrast-%s.npy' % contrast, allow_pickle = True)

Tunings = np.load('/home/user/DATA/Astrid/Cibele_data/summary/2both_signi_Tunings_SST-cells_WT_Adult_V1_contrast-%s.npy' % contrast, allow_pickle = True)

#%% plot evoked responses

for n_session in range(len(Tunings)) : 

    if n_session != 4 :
        Tuning = Tunings[n_session]
        data, Episodes = create_data_and_Episodes(Tuning) 
        ind_sign_and_neg = find_ind_sign_and_neg(Tuning)

        fig, axes = plt.subplots(figsize = (13,2*(len(ind_sign_and_neg)//5)+3), nrows = len(ind_sign_and_neg)// 5 + 1 , ncols = 5 ,  layout = 'constrained')
        fig.suptitle('SST WT contrast = %s' % contrast + '\n session n° %s' % n_session , fontsize = 15)
        for i in range(len(ind_sign_and_neg)): 

            ind = ind_sign_and_neg[i]

            n_cell = ind[0]
            resp_to_match = Tuning['Responses'][n_cell, ind[1]]
            evoked_response, resp = find_evoked_response_matching_resp_mean(resp_to_match, Episodes, n_cell, contrast) 

            mean_evoked_response = evoked_response.mean(axis = 0)
            max_e_r = np.max(mean_evoked_response) 
            min_e_r = np.min(mean_evoked_response) 

            axes.flatten()[i].fill_between([1,2], min_e_r, max_e_r +0.1, color = 'orange', alpha = 0.2)
            axes.flatten()[i].fill_between([-1,0], min_e_r, max_e_r +0.1, color = 'green', alpha = 0.2)

            axes.flatten()[i].plot(Episodes.t, mean_evoked_response, color = 'black')
            axes.flatten()[i].annotate(text = 'n°' + str(n_cell), xy = (4,max_e_r), fontsize = 15)
        plt.show()

#%%
def create_data_and_Episodes(Tuning) : 
      
    filename = Tuning['datafile']
    data = physion.analysis.read_NWB.Data(filename, verbose=False) 
    dFoF_parameters = dict(\
            roi_to_neuropil_fluo_inclusion_factor=1.15,
            neuropil_correction_factor = 0.7,
            method_for_F0 = 'sliding_percentile',
            percentile=5., # percent
            sliding_window = 5*60, # seconds
    )
    data.build_dFoF(**dFoF_parameters, verbose=False)

    Episodes = EpisodeData(data, 
                            quantities=['dFoF'], 
                            protocol_name = data.protocols[0],
                            verbose=False)

    return data, Episodes

def find_ind_sign_and_neg(Tuning) : 

    Tuning['significant_ROIs'] = Tuning['significant_ROIs'].astype(bool)

    ind_sign = np.argwhere(Tuning['significant_ROIs'] == True) 
    ind_sign_and_neg = []
    for ind in ind_sign:
        if Tuning['Responses'][ind[0], ind[1]] < 0 : 
            ind_sign_and_neg.append(ind)

    return ind_sign_and_neg


def find_evoked_response_matching_resp_mean(resp_to_match, Episodes, n_cell, contrast) :

    i = 0
    resp = np.nan
    while resp != np.round(resp_to_match, 5): 
        angle_neg = np.array([-22.5,   0. ,  22.5,  45. ,  67.5,  90. , 112.5, 135. ])[i] + 22.5
        cond = (Episodes.angle == angle_neg) & (Episodes.contrast == contrast)
        
        evoked_response = Episodes.dFoF[cond][:, n_cell, :]
        evoked_response_pre = evoked_response[:, 998:1999]
        evoked_response_post = evoked_response[:, 2998:3999]

        x = evoked_response_pre.mean(axis = 1)
        y = evoked_response_post.mean(axis = 1) 
        resp = np.round(np.mean(y-x), 5)
        i+=1

    return evoked_response, resp

#%% Compute % of significant evoked response that are negative 


Tunings1 = np.load('/home/user/DATA/Astrid/Cibele_data/summary/2both_signi_Tunings_SST-cells_WT_Adult_V1_contrast-1.0.npy', allow_pickle = True)
Tunings05 = np.load('/home/user/DATA/Astrid/Cibele_data/summary/2both_signi_Tunings_SST-cells_WT_Adult_V1_contrast-0.5.npy', allow_pickle = True)
Tunings1_KO = np.load('/home/user/DATA/Astrid/Cibele_data/summary/test_both_sign_Tunings_SST-cells_cond-GluN1-KO_Adult_V1_contrast-1.0.npy', allow_pickle = True)
Tunings05_KO = np.load('/home/user/DATA/Astrid/Cibele_data/summary/2both_signi_Tunings_SST-cells_cond-GluN1-KO_Adult_V1_contrast-0.5.npy', allow_pickle = True)
colors = [(to_rgb('#12522eff'), 1), 'lightgrey',
    (to_rgb('#5d1490ff'), 1),'lightgrey']


#%% Taddy's 
summary_folder =  "/home/user/DATA/Astrid/Taddy_data/OneDrive_1_9-3-2026/summary/"
Tunings1 = np.load(summary_folder + 'both_sign_Tunings_Wild-Type_contrast-1.0.npy', allow_pickle = True)
Tunings05 = np.load(summary_folder + 'both_sign_Tunings_Wild-Type_contrast-0.5.npy', allow_pickle = True)
Tunings1_KO = np.load(summary_folder + 'both_sign_Tunings_GluN1-KO_contrast-1.0.npy', allow_pickle = True)
Tunings05_KO = np.load(summary_folder + 'both_sign_Tunings_GluN1-KO_contrast-0.5.npy', allow_pickle = True)
colors = [(pt.tab10(1)[:3], 1), 'lightgrey',
    (pt.tab10(4)[:3], 1), 'lightgrey']

#%%
PERC_neg_sign = []
PERC_sign_neg = []
N_RESP_sign = []
N_CELL_sign = []
for Tunings in [Tunings1, Tunings05, Tunings1_KO, Tunings05_KO]:

        
    perc_neg_sign = []
    perc_sign_neg = []
    N_resp_sign = []
    N_cell_sign = []
    for i, T in enumerate(Tunings) : 
            
        if i == 4 and 'GluN1-KO' in T['datafile'] : 
            continue
        else : 
            n_cell_sign = np.sum(np.sum(T['significant_ROIs'], axis = 1) > 0)
            T['significant_ROIs'] = T['significant_ROIs'].astype(bool)
            n_resp_tot = T['Responses'].shape[0]*8
            n_resp_sign = np.sum(T['significant_ROIs'])

            n_resp_neg = np.sum(T['Responses'] < 0)
            n_resp_neg_sign = np.sum(T['Responses'][T['significant_ROIs']] < 0)

            n_resp_pos = np.sum(T['Responses'] > 0)
            n_resp_pos_sign = np.sum(T['Responses'][T['significant_ROIs']] >= 0)

            #perc_neg_sign.append((n_resp_neg_sign / n_resp_neg) *100)
            perc_sign_neg.append((n_resp_neg_sign / (n_resp_neg_sign + n_resp_pos_sign)) * 100)
            N_resp_sign.append(n_resp_sign)
            N_cell_sign.append(n_cell_sign)
    N_RESP_sign.append(N_resp_sign)
    N_CELL_sign.append(N_cell_sign)
    #PERC_neg_sign.append(perc_neg_sign)
    PERC_sign_neg.append(perc_sign_neg)

#%% plot

mean_PERC_sign_neg = [np.mean(perc) for perc in PERC_sign_neg]



plt.figure(figsize=(5,5))
x = [0,1,3,4]
plt.bar(x = x, height = mean_PERC_sign_neg, width= 0.7, color = colors, alpha = 0.5)
for i, perc in enumerate(PERC_sign_neg) : 
        range_x = (np.arange(len(perc)) *0.7 / len(perc) ) + x[i] - 0.35
        plt.scatter(range_x, perc, color = colors[i])
plt.ylabel("%" + " of significant resp that are negative")
plt.xticks(ticks = x, labels=['SST WT \n contrast 1', 'SST WT \n contrast 0.5', 'SST KO \n contrast 1', 'SST KO \n contrast 0.5'], rotation = 45)
plt.ylim(0,65)

n_resp_sign = [np.sum(n_sign) for n_sign in N_RESP_sign]
n_cell_sign = [np.sum(n_sign) for n_sign in N_CELL_sign]
plt.annotate(text = 'N significants cells : ', xy = (0,65), fontsize = 6)

plt.annotate(text = 'N significants responses : ', xy = (0,58), fontsize = 6)
for i in range(len(n_resp_sign)) : 
    plt.annotate(text = str(n_cell_sign[i]), xy = (0 + i*0.5,62), fontsize = 6, color = colors[i])

    plt.annotate(text = str(n_resp_sign[i]), xy = (0 + i*0.5,55), fontsize = 6, color = colors[i])
