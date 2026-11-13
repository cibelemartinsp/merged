# %% [markdown]
# # Build Sensitivity Dataset across Conditions

# %%
import os, sys , shutil 
import multiprocessing
import numpy as np


# physion and the merged modules are taken from merged/ (parent of create_summaries/)
try:
    summaries_dir = os.path.dirname(os.path.abspath(__file__))
except NameError: # interactive window
    summaries_dir = os.path.join(os.path.expanduser('~'), 'code', 'merged', 'create_summaries')
merged_root = os.path.dirname(summaries_dir)
for path in [summaries_dir, merged_root, os.path.join(merged_root, 'physion', 'src')]:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
from physion.analysis.read_NWB\
                         import scan_folder_for_NWBfiles, Data
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.protocols.contrast_sensitivity\
                        import compute_sensitivity_per_cells

from physion.analysis.episodes.build import EpisodeData
from dataset_tools import (get_summary_prefix_name, 
                            get_filtering_cond, 
                            build_filtering_cond_quantities)
from contrast_dataset_tools import *

parallelized, debug = False, False 

# load the dataset locations:
from Dataset_Organization import datasets_func, quantity, summary_folder, filtering_cond_name
datasets = datasets_func('angle', [0., 90.])

from Preprocessing_Settings import get_dFoF_params, get_stat_test_props

if sys.argv == ['Contrast-Dataset.py']:
    special_dFoF_params = None
else : 
    special_dFoF_params = sys.argv 
# %%
# to be a valid dataset:
nMIN_DATAFILES = 2

def process_file(filename, i, c, quantity, filtering_cond_name):

    # to be a valid datafile:
    nMIN_ROIs = 4

    # CELL-dependent calcium pre-processing params 
    dFoF_parameters = get_dFoF_params(c, special_dFoF_params)

    # statistical test for visually-evoked-responses
    stat_test_props= get_stat_test_props(c)


    response_significance_threshold=5e-2

    print('%i) ' % (i+1), 'analyzing file: %s  [...] ' % filename)
    data = Data(filename, verbose=False)
    protocol_name=[p for p in data.protocols if '8contrast' in p][0]
    # Zscore of F0 or correctedFluo needs them kept (with the ROI selection of dFoF)
    data.build_dFoF(**dFoF_parameters, with_correctedFluo_and_F0=True, verbose=False)

    if quantity[:6] == 'Zscore' :
        data.build_Zscore(quantity[7:])
        # Episodes need the time of the signal : ophys quantities are on the dFoF time
        setattr(data, 't_%s' % quantity, data.t_dFoF)

    # name of the signal in data/Episodes (quantity still names the temp files)
    signal = quantity

    if quantity[:11] == 'Deconvolved':
        #setattr(data, quantity, data.correctedFluo - data.correctedFluo0)
        data.build_Deconvolved(Tau = 1.5, quantity = quantity[12:])
        stat_test_props['interval_post']=[0.0, 1.0]
        signal = quantity[:11] # stored as data.Deconvolved

    quantities = [signal]

    if data.nROIs>=nMIN_ROIs:

        try:
            if filtering_cond_name :
                data, quantities = build_filtering_cond_quantities(filtering_cond_name, data, quantities)

            Episodes = EpisodeData(data, 
                                    quantities=quantities,
                                    protocol_name=protocol_name, 
                                    verbose=False)
            
            filtering_cond = get_filtering_cond(filtering_cond_name, Episodes)

            Sensitivity = compute_sensitivity_per_cells(data, Episodes,
                                                        quantity=signal,
                                                        stat_test_props=stat_test_props, 
                                                        response_significance_threshold = response_significance_threshold, 
                                                        filtering_cond = filtering_cond, 
                                                        angle = float(c.split('angle-')[1][:3]))
            if filtering_cond_name :
                Sensitivity = fill_missing_contrast_values_with_nans(Sensitivity, Episodes)

            Sensitivity['datafile'] = filename
            Sensitivity['nROIs_original'] = data.original_nROIs
            Sensitivity['nROIs_final'] = data.nROIs
            Sensitivity['subject'] = data.nwbfile.subject.subject_id

            if  dFoF_parameters['computed_neuropil_fact_method'] : #dFoF_parameters['roi_specific_neuropil_correction_factor'] or dFoF_parameters['with_computed_neuropil_fact'] : 
                Sensitivity['neuropil_correction_factors'] = data.neuropil_correction_factor

            np.save(os.path.join(summary_folder, 'temp', 
                                 'Sensitivity-%s-%s-%i.npy' % (quantity, c, i)),
                    Sensitivity)
            print('      [v] --> included, n=%i ROIs ' % data.nROIs)

        except ValueError as ve: #value error from no locomotion value or no prefered angles for filtered summaries
            print(f"Error: {ve}")
            print('File: %s' % filename, ' discarded')
        
        # except BaseException as be:
        #     print('                        [-------------------------------]')
        #     print(be)
        #     print()
        #     print('      [X] --> discarded, problem in datafile, CHECK [!!]')
        #     print('                        [-------------------------------]')

    else:
        print('      [X] --> discarded, n=%i ROIs ' % data.nROIs)

if __name__=='__main__':

    from physion.assembling.dataset import read_spreadsheet
    cpus = multiprocessing.cpu_count()-1 # leaving 1 cpu for the rest

    # temporary folder for parallelization
    os.makedirs(os.path.join(summary_folder, 'temp'), exist_ok=True)

    Nstart = 0
    Nend = len(datasets) 

    for n in range(Nstart, Nend):

        c = list(datasets.keys())[n]

        table = datasets[c]['datafolder'].replace('NWBs', 'DataTable.xlsx')

        #dataset_table, subjects_table, analysis =\
        #        read_spreadsheet(table, get_metadata_from='table')
        print()
        print()
        print('=================================================================')
        print('-----------------------------------------------------------------')
        print('------- %i) computing : %s ' % (n+1, c))
        print('-----------------------------------------------------------------')
        print()

        DATASET = scan_folder_for_NWBfiles(datasets[c]['datafolder'])
        
        # FILTER
        # 1) protocol type: contrast sensitivity
        cond = np.array([np.sum(['ff-gratings-2orientations-8contrasts-15repeats' in p for p in protocols])\
                        for protocols in DATASET['protocols']], dtype=bool)
        # 2) age condition
        if datasets[c]['age_interval'] is not None:
            cond = cond &\
                (DATASET['ages']>=datasets[c]['age_interval'][0]) &\
                (DATASET['ages']<=datasets[c]['age_interval'][1])

        if len(DATASET['files'][cond])>nMIN_DATAFILES:

            if parallelized:
                ################################################
                ###    parallelization here !   #################
                ################################################
                nruns = int(len(DATASET['files'][cond])/cpus)+1

                for r in range(nruns):
                    i0 = r*cpus
                    imax = np.min([i0+cpus, len(DATASET['files'][cond])]) 
                    print(' - running set of files %i:%i' % (i0, imax))

                    # start the processes
                    procs = []
                    for i in range(i0,imax):
                        proc = multiprocessing.Process(\
                                            target=process_file, 
                                            args=(DATASET['files'][cond][i], 
                                                  i, c, quantity, filtering_cond_name))
                        procs.append(proc)
                        proc.start()

                    # complete the processes
                    for proc in procs:
                        proc.join()
            else:
                #####################################
                ###### UN-PARALLELIZED VERSION ######
                for i, f in enumerate(DATASET['files'][cond]):
                    process_file(f, i, c, quantity, filtering_cond_name)
                #####################################

            # now that we have stored all datafile outputs
            Sensitivities = []
            for i, f in enumerate(DATASET['files'][cond]):

                if os.path.isfile(os.path.join(summary_folder, 'temp', 
                                              'Sensitivity-%s-%s-%i.npy' % (quantity, c, i))):
                    Sensitivity = np.load(os.path.join(summary_folder, 'temp', 
                                                'Sensitivity-%s-%s-%i.npy' % (quantity, c, i)),
                                        allow_pickle=True).item()
                    Sensitivities.append(Sensitivity)

            # # saving data
            summary_prefix_name = get_summary_prefix_name(quantity, filtering_cond_name, special_dFoF_params)
            
            np.save(os.path.join(summary_folder, summary_prefix_name + 'Sensitivities_%s.npy' % c), 
                    Sensitivities)

        else:
            print()
            print('   [!!]   DATASET NOT LARGE ENOUGH   [!!] ')
            print('               only N=%i sessions available' %\
                                        len(DATASET['files'][cond]))
            print('   [!!]   DATASET not analyzed       [!!] ')
            print()

        print('-----------------------------------------------------------------')
        print('=================================================================')
    # shutil.rmtree(os.path.join(summary_folder, 'temp'))

# %%
#f = '/home/user/CURATED/Cibele/SST-cells_cond-GluN1-KO_Adult_V1/NWBs/2026_02_18-14-42-43.nwb'
#d = Data(f, verbose=True)

# %%
if False:
    from Dataset_Organization_cibele import summary_folder
    from physion.analysis.protocols.contrast_sensitivity\
            import plot_contrast_sensitivity, plot_contrast_responsiveness
    fig, ax = plot_contrast_sensitivity(\
                            ['PV-cells_WT_Adult_V1_contrast-1.0', 
                             'PV-cells_WT_Adult_V1_contrast-0.5'],
                            #   average_by='ROIs',
                            path=summary_folder)


# %%
