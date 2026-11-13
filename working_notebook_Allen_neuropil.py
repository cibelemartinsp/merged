#%%

from Allen_Institute_r_neuropil import *
from general_tools.cell_type_color_codes import get_working_data_defaults
from general_tools.general_tools import *
#%%
T = 50000
N = 10
r_truth_vals, r_est_vals = validate_with_synthetic_F(T, N)

#%% effect of lambda on r

"""
self.F_M is a list of len folds
self.T = len(Fm)
"""
Tunings_wt_1 = np.load('/home/user/DATA/Astrid/Cibele_data/summary/Tunings_SST-cells_WT_Adult_V1_contrast-1.0.npy', allow_pickle = True) 
Tunings_wt_05 = np.load('/home/user/DATA/Astrid/Cibele_data/summary/Tunings_SST-cells_WT_Adult_V1_contrast-0.5.npy', allow_pickle = True) 
Tunings_ko_1 = np.load('/home/user/DATA/Astrid/Cibele_data/summary/Tunings_SST-cells_cond-GluN1-KO_Adult_V1_contrast-1.0.npy', allow_pickle = True) 
Tunings_ko_05 = np.load('/home/user/DATA/Astrid/Cibele_data/summary/Tunings_SST-cells_cond-GluN1-KO_Adult_V1_contrast-0.5.npy', allow_pickle = True) 
Tunings_ko_1_taddy = np.load('/home/user/DATA/Astrid/Taddy_data/OneDrive_1_9-3-2026/summary/Tunings_GluN1-KO_contrast-1.0.npy', allow_pickle = True) 
Tunings_ko_05_taddy = np.load('/home/user/DATA/Astrid/Taddy_data/OneDrive_1_9-3-2026/summary/Tunings_GluN1-KO_contrast-0.5.npy', allow_pickle = True) 
Tunings_pv_1 = np.load('/home/user/DATA/Astrid/Cibele_data/summary/Tunings_PV-cells_WT_Adult_V1_contrast-1.0.npy', allow_pickle = True) 
Tunings_pv_05 = np.load('/home/user/DATA/Astrid/Cibele_data/summary/Tunings_PV-cells_WT_Adult_V1_contrast-0.5.npy', allow_pickle = True) 
Tunings_wt_1_taddy = np.load('/home/user/DATA/Astrid/Taddy_data/OneDrive_1_9-3-2026/summary/Tunings_Wild-Type_contrast-1.0.npy', allow_pickle = True) 


#%%

from scipy.stats import sem

MEANS_R = []
SEM_R = []
OVERALL_R_LIST = []
for Tunings in [Tunings_wt_1, Tunings_ko_1, Tunings_wt_1_taddy, Tunings_ko_1_taddy, Tunings_pv_1] : 
    
    mean_r_with_lam = []
    sem_r_with_lam = []
    for lam in [5,20]:
        overall_r_list = []

    
        for T in Tunings : 

            filename = T['datafile']
            data = physion.analysis.read_NWB.Data(filename, verbose=False) 
            dFoF_parameters = dict(\
                roi_to_neuropil_fluo_inclusion_factor=1.15,
                neuropil_correction_factor = 0.7,
                roi_specific_neuropil_correction_factor = True,
                lam=lam,
                method_for_F0 = 'sliding_percentile',
                percentile=5., # percent
                sliding_window = 5*60, # seconds
                    )

            data.build_dFoF(**dFoF_parameters, verbose=False)
            r_list_valid = np.concatenate(data.neuropil_correction_factor)
            print(len(data.valid_roiIndices)/data.original_nROIs)
            overall_r_list.append(r_list_valid)

        OVERALL_R_LIST.append(overall_r_list)
        mean_r_with_lam.append(np.mean(np.concatenate(overall_r_list)))
        sem_r_with_lam.append(sem(np.concatenate(overall_r_list)))

    MEANS_R.append(mean_r_with_lam)
    SEM_R.append(sem_r_with_lam)


#%%


summary_path, folder, colors = get_working_data_defaults(experimenters = ['Cibele','Cibele','Taddy','Taddy','Cibele'],
                                                        folders_surnames = ['SST_WT', "SST_GluN1_KO", 'SST_WT', "SST_GluN1_KO", 'PV_WT'],
                                                        color_with_grey=False)
new_colors = []
for c in colors :
    new_colors.append(c)
    #new_colors.append(c)
lam_list =[5,20]
#%%
STD_R = [np.std(np.concatenate(OVERALL_R_LIST[i])) for i in range(len(OVERALL_R_LIST))]
STD_R_reshaped = np.array(STD_R).reshape(5,2)
#%%
plt.figure(figsize = (9,9))
for i in range(len(MEANS_R)) : 
    plt.plot(lam_list, MEANS_R[i], color = new_colors[i], label = folder[i])
    plt.errorbar(lam_list, MEANS_R[i], yerr = SEM_R[i],  color = new_colors[i])
plt.xticks(ticks=lam_list, labels = lam_list, rotation = 90)
plt.xlim(2,23)
plt.grid(True, axis = 'y', linewidth = 1)
plt.xlabel('lambda penalization value')
plt.ylabel('mean neuropil correction factor (over all valid cells)')
plt.legend(loc = 'upper left')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filename = Tunings_wt_1[0]['datafile']
data = physion.analysis.read_NWB.Data(filename, verbose=False) 
dFoF_parameters = dict(\
        roi_to_neuropil_fluo_inclusion_factor=1.15,
        neuropil_correction_factor = 0.7,
        roi_specific_neuropil_correction_factor = False,
        method_for_F0 = 'sliding_percentile',
        percentile=5., # percent
        sliding_window = 5*60, # seconds
)
data.build_dFoF(**dFoF_parameters, verbose=False)

#%%
filename = Tunings_wt_1[0]['datafile']
data_allen = physion.analysis.read_NWB.Data(filename, verbose=False) 
dFoF_parameters = dict(\
        roi_to_neuropil_fluo_inclusion_factor=1.15,
        neuropil_correction_factor = 0.7,
        roi_specific_neuropil_correction_factor = True,
        method_for_F0 = 'sliding_percentile',
        percentile=5., # percent
        sliding_window = 5*60, # seconds
)
data_allen.build_dFoF(**dFoF_parameters, verbose=False)

#%%

for roi in range(data.nROIs):
    plt.figure(figsize=(4,4))
    plt.plot(data.dFoF[roi][1000:1100])
    plt.plot(data_allen.dFoF[roi][1000:1100], color = 'purple')


for roi in range(data.nROIs) : 
    plt.figure(figsize = (4,4))
    res = estimate_contamination_ratios(data.rawFluo[roi,:], data.neuropil[roi,:])
    plt.scatter(res['r_vals'], res['err_vals'])
    i_min_error = np.argwhere(res['err_vals'] == res['min_error'])[0][0]
    plt.scatter(res['r_vals'][i_min_error], res['err_vals'][i_min_error], color = 'red')