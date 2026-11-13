# %% [markdown]
# # Build Tuning Dataset across Conditions

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
from physion.analysis.protocols.orientation_tuning\
                import compute_tuning_response_per_cells
from dataset_tools import (get_summary_prefix_name, 
                                    get_filtering_cond, 
                                    build_filtering_cond_quantities)
from tuning_dataset_tools import *

parallelized, debug = False, False 

# load the dataset locations:
from Dataset_Organization import datasets_func, quantity, summary_folder, filtering_cond_name
datasets = datasets_func('contrast', [0.5, 1.0])

if filtering_cond_name is not None : 
    from tuning_dataset_tools import (get_prefered_angles_dataset,
                                        get_prefered_angles)

from Preprocessing_Settings import get_dFoF_params, get_stat_test_props


if sys.argv == ['Tuning-Dataset.py']:
    special_dFoF_params = None
else : 
    special_dFoF_params = sys.argv 
# %%
def process_file(filename, i, c, quantity, filtering_cond_name):
    # to be a valid datafile:
    nMIN_ROIs = 4

    # CELL type-dependent calcium pre-processing params 
    dFoF_parameters = get_dFoF_params(c, special_dFoF_params)
    # CELL type-dependent statistical test for visually-evoked-responses
    stat_test_props= get_stat_test_props(c)

    response_significance_threshold=5e-2

    print('%i) ' % (i+1), 'analyzing file: %s  [...] ' % filename)
    data = Data(filename, verbose=False)
    protocol_name=[p for p in data.protocols if '8orientation' in p][0]

    data.build_dFoF(**dFoF_parameters, verbose=False)
    
    if quantity[:6] == 'Zscore' :
        data.build_Zscore(quantity[7:])

    elif quantity[:11] == 'Deconvolved':
        data.build_Deconvolved(Tau = 0.8, quantity = quantity[12:])
        stat_test_props['interval_post']=[0.0, 1.0]  
        quantity = quantity[:11]             

    quantities = [quantity]
    prefered_angles = None

    if data.nROIs>=nMIN_ROIs:

        try:
            
            if filtering_cond_name :
                data, quantities = build_filtering_cond_quantities(filtering_cond_name, data, quantities)
                prefered_angles = get_prefered_angles(prefered_angles_dataset, filename)

            Episodes = EpisodeData(data, 
                                    quantities=quantities,
                                    protocol_name=protocol_name, 
                                    verbose=debug)
            
            filtering_cond = get_filtering_cond(filtering_cond_name, Episodes)

            Tuning = compute_tuning_response_per_cells(data, Episodes, 
                                                        quantity=quantity, 
                                                        stat_test_props = stat_test_props, 
                                                        prefered_angles=prefered_angles,
                                                        response_significance_threshold =\
                                                            response_significance_threshold, 
                                                        filtering_cond=filtering_cond,
                                                        contrast =\
                                                            float(c.split('contrast-')[1][:3]),
                                                        nMin_episodes = 2,
                                                        keep_all_orientations_significances = False, 
                                                        verbose=debug)
            Tuning['datafile'] = filename
            Tuning['nROIs_original'] = data.original_nROIs
            Tuning['nROIs_final'] = data.nROIs
            Tuning['nROIs_responsive'] = np.sum(Tuning['significant_ROIs'])
            Tuning['subject'] = data.nwbfile.subject.subject_id

            if  dFoF_parameters['computed_neuropil_fact_method'] : #dFoF_parameters['roi_specific_neuropil_correction_factor'] or dFoF_parameters['with_computed_neuropil_fact'] : 
                Tuning['neuropil_correction_factors'] = data.neuropil_correction_factor

            np.save(os.path.join(summary_folder, 'temp', 
                                 'Tuning-%s-%i.npy' % (c, i)),
                    Tuning)
            print('      [v] --> included, n=%i ROIs ' % data.nROIs)

        except ValueError as ve: #value error from no locomotion value or no prefered angles for filtered summaries
            print(f"Error: {ve}")
            print('File: %s' % filename, ' discarded')
        
        except BaseException as be:
            print('                        [-------------------------------]')
            print(be)
            print()
            print(filename)
            print('nROIs=%i' % data.nROIs, ', protocols=%s' % data.protocols) 
            print(Episodes.varied_parameters)
            print('      [X] --> discarded, problem in datafile, CHECK [!!]')
            print('                        [-------------------------------]')

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
                #read_spreadsheet(table, get_metadata_from='table')
        print()
        print()
        print('=================================================================')
        print('-----------------------------------------------------------------')
        print('------- %i) computing : %s ' % (n+1, c))
        print('-----------------------------------------------------------------')
        print()

        DATASET = scan_folder_for_NWBfiles(datasets[c]['datafolder'])
        
        if filtering_cond_name:
            prefered_angles_dataset = get_prefered_angles_dataset(datasets[c]['datafolder'], summary_folder, c.split('_contrast-')[1])

        #   FILTER:
        # ----------
        # 1) protocol type: orientation tuning
        cond = np.array([np.sum(['ff-gratings-8orientation-2contrasts-15repeats' in p for p in protocols])\
                        for protocols in DATASET['protocols']], dtype=bool)
        # 2) age condition (not None only if young)
        if (datasets[c]['age_interval'] is not None):
            cond = cond &\
                (DATASET['ages']>=datasets[c]['age_interval'][0]) &\
                (DATASET['ages']<=datasets[c]['age_interval'][1])

        if np.sum(cond)>0:
            # MINIMUM NUMBER OF SESSION (1 for now)

            if parallelized:
                ################################################
                ###    parallelization here !  #################
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
                                            args=(DATASET['files'][cond][i], i, c, quantity, filtering_cond_name))
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
            Tunings = []
            for i, f in enumerate(DATASET['files'][cond]):

                if os.path.isfile(os.path.join(summary_folder, 'temp', 
                                                'Tuning-%s-%i.npy' % (c, i))):
                    Tuning = np.load(os.path.join(summary_folder, 'temp', 
                                                'Tuning-%s-%i.npy' % (c, i)),
                                        allow_pickle=True).item()
                    Tunings.append(Tuning)

            # # saving data
            summary_prefix_name = get_summary_prefix_name(quantity, filtering_cond_name, special_dFoF_params)

            np.save(os.path.join(summary_folder, summary_prefix_name + 'Tunings_%s.npy' % c), Tunings)

        else:
            print()
            print('   [!!]   DATASET NOT LARGE ENOUGH   [!!] ')
            print('               only N=%i sessions available' %\
                                        len(DATASET['files'][cond]))
            print('   [!!]   DATASET not analyzed       [!!] ')
            print()

        print('-----------------------------------------------------------------')
        print('=================================================================')
    # remove only the temp files of this script: summary/temp is shared with the other
    # summary scripts (Temporal-Dynamics, Deconv-Contrast, ...) that may be running at the same time
    for temp_file in os.listdir(os.path.join(summary_folder, 'temp')):
        if temp_file.startswith('Tuning-'):
            os.remove(os.path.join(summary_folder, 'temp', temp_file))

# %%
if False:
    from Dataset_Organization import summary_folder
    from physion.analysis.protocols.orientation_tuning\
        import plot_orientation_tuning_curve, plot_selectivity

    folder = "PYR-PV-SynGCaMP_WT_Young_V1"
    colors = [pt.tab10(1), 'lightgrey']
    fig, ax = plot_selectivity(\
                            [folder + '_contrast-1.0', 
                             folder + '_contrast-0.5'],
                            #   average_by='ROIs',
                            #  using='fit',
                            path=summary_folder, 
                            colors = colors)
        
    fig, ax = plot_orientation_tuning_curve(\
                            [folder + '_contrast-1.0', 
                             folder + '_contrast-0.5'],
                                            # average_by='ROIs',
                            path=summary_folder, 
                            colors = colors)
