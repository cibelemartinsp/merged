import os, sys
# physion and the merged modules are taken from merged/ (parent of general_tools/)
try:
    merged_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError: # interactive window
    merged_root = os.path.join(os.path.expanduser('~'), 'code', 'merged')
for path in [merged_root, os.path.join(merged_root, 'physion', 'src')]:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import re
from matplotlib.colors import to_rgb
import pandas as pd 
import physion.utils.plot_tools as pt



data_from_list = ['Cibele']*16 + ['Taddy']*2 + ['Cibele']*2

# summaries of all groups (Cibele and Taddy) are in ~/CURATED/Cibele/summary
base_path = [os.path.join(os.path.expanduser('~'), 'CURATED', 'Cibele')]*20

# folder names = dataset names used in the summary filenames (e.g. Tunings_<folder>_contrast-1.0.npy)
folders = [
    "SST-cells_WT_Adult_V1",
    "SST-cells_cond-GluN1-KO_Adult_V1",
    "PV-cells_WT_Adult_V1",
    "PV-cells_cond-GluN1-KO_Adult_V1",
    "PYR-SynGCaMP_WT_Adult_V1",
    "SST-cells_WT_P15-P19_V1",
    "SST-cells_WT_P20-P23_V1",
    "SST-cells_WT_P24-P27_V1",
    "PV-cells_WT_P20-P23_V1",
    "PV-cells_WT_P24-P27_V1",
    "PYR-PV-SynGCaMP_WT_P15-P19_V1",
    "PYR-PV-SynGCaMP_WT_P20-P23_V1",
    "PYR-PV-SynGCaMP_WT_P24-P27_V1",
    "SST-cells_cond-GluN1-KO_P15-P19_V1",
    "SST-cells_cond-GluN1-KO_P20-P23_V1",
    "SST-cells_cond-GluN1-KO_P24-P27_V1",
    "SST-cells_WT_Adult_V1_Taddy",
    "SST-cells_cond-GluN1-KO_Adult_V1_Taddy",
    "GCaMP8m-Test_V1",
    "GCaMP8s-Test_V1"
    ]

folders_surnames = [
    "SST_WT",
    "SST_GluN1_KO",
    "PV_WT", 
    "PV_GluN1_KO",
    "PYR_WT",
    "SST_WT_P15",
    "SST_WT_P20",
    "SST_WT_P24",
    "PV_WT_P20",
    "PV_WT_P24",
    "PYR_WT_P15",
    "PYR_WT_P20",
    "PYR_WT_P24",
    "SST_GluN1_KO_P15",
    "SST_GluN1_KO_P20",
    "SST_GluN1_KO_P24",
    "SST_WT",
    "SST_GluN1_KO",
    "GCaMP8m",
    "GCaMP8s"
    ]


colors = [
    (*to_rgb('#12522eff'), 1),
    (*to_rgb('#5d1490ff'), 1),
    (*to_rgb('#b30a7bff'), 1),
    (*to_rgb('#9e0004ff'), 1),
    (*to_rgb('#005d8aff'), 1),
    (*to_rgb('#c6e7bbff'), 1),
    (*to_rgb('#99b993ff'), 1),
    (*to_rgb('#428e58ff'), 1),
    (*to_rgb('#ff9ab5ff'), 1),
    (*to_rgb('#e31593ff'), 1),
    (*to_rgb('#b8cfe0ff'), 1),
    (*to_rgb('#6caaccff'), 1),
    (*to_rgb('#008fbbff'), 1),
    (*to_rgb('#dbc0f2ff'), 1),
    (*to_rgb('#b38fd6ff'), 1),
    (*to_rgb('#9740d2ff'), 1),
    pt.tab10(1), 
    pt.tab10(4), 
    (*to_rgb('k'), 1),
    (*to_rgb('lightgrey'), 1),
    ]



data = {'experimenter' : data_from_list, 'base_path' : base_path,'folder_name' : folders, 'folder_surname' : folders_surnames, 'color' : colors }
working_data_defaults = pd.DataFrame(data)



def get_working_data_defaults(experimenters = ['Cibele'], 
                              folders_surnames = ['SST_WT'], 
                              color_with_grey = False, 
                              alpha = 1) : 


    #correcting and checking input form 
    if type(experimenters) == str: 
        experimenters = [experimenters]

    if type(folders_surnames) == str: 
        folders_surnames = [folders_surnames]

    if len(experimenters) != len(folders_surnames) : 
        print("experimenters and folders surnames length do not match \n assuming %s experimenter for all folders " % experimenters[0])
        experimenters = [experimenters[0]] * len(folders_surnames)

    #collecting relevant values
    summary_folder = []
    folders = []
    colors = []
    for experimenter, folder_surname in zip(experimenters, folders_surnames) : 

        cond = (working_data_defaults['experimenter'] == experimenter) & \
                (working_data_defaults['folder_surname'] == folder_surname)
        
        summary_folder.append(working_data_defaults['base_path'][cond].values.item() + '/summary')
        folders.append(working_data_defaults['folder_name'][cond].values.item())
        colors.append(working_data_defaults['color'][cond].values.item())

    if alpha != 1 :
        colors = [(*c[:3], alpha) for c in colors]

    if color_with_grey : 
        colors = [[c, (*to_rgb('lightgrey'),1)] for c in colors]

    return summary_folder, folders, colors 


def get_group_surname(dataset_name) :
    """ folder surname of a dataset, from any of its names
        e.g. 'Deconvolved_SST-cells_cond-GluN1-KO_P20-P23_V1_angle-0.0' -> 'SST_GluN1_KO_P20'
             'PYR-PV-SynGCaMP_WT_Adult_V1' -> 'PYR_WT' """

    # GCaMP indicator test groups (no cell type in their name)
    indicator = re.search(r'(GCaMP8[ms])-Test', dataset_name)
    if indicator is not None :
        return indicator.group(1)

    cell_type = re.search(r'(SST|PYR|PV)-', dataset_name)
    if cell_type is None :
        raise ValueError('no cell type (SST/PV/PYR) found in "%s"' % dataset_name)

    genotype = 'GluN1_KO' if re.search(r'GluN1[-_]KO', dataset_name) else 'WT'

    age = re.search(r'_(P\d+)-P\d+', dataset_name)
    if age is not None :
        age = '_' + age.group(1)
    elif 'Young' in dataset_name :
        age = '_Young' # pooled ages, no color of its own
    else :
        age = '' # adult

    # other injection protocol, not the same group
    protocol = '_Ventricule' if 'Ventricule' in dataset_name else ''

    return '%s_%s%s%s' % (cell_type.group(1), genotype, age, protocol)


def get_color(dataset_names,
              experimenter = 'Cibele',
              alpha = 1,
              default = None) :
    """ color of the group of a dataset (or a list of colors for a list of datasets),
        taken from working_data_defaults through the group surname
        groups without a color raise a KeyError, unless a default color is given """

    if type(dataset_names) == str :
        return get_color([dataset_names], experimenter = experimenter, alpha = alpha, default = default)[0]

    colors = []
    for dataset_name in dataset_names :
        surname = get_group_surname(dataset_name)
        cond = (working_data_defaults['experimenter'] == experimenter) & \
                (working_data_defaults['folder_surname'] == surname)
        if not cond.any() and default is not None :
            colors.append(default)
            continue
        if not cond.any() :
            raise KeyError('no color for group "%s" (from "%s") in working_data_defaults' % (surname, dataset_name))
        colors.append((*working_data_defaults['color'][cond].values.item()[:3], alpha))

    return colors


def reshape_colors_with(colors,
                        alpha = 1,
                       second_cond_grey = False,
                       second_cond_low_alpha = False, 
                       second_cond_alpha = 0.3) : 

    #check and initialisation
    if second_cond_grey & second_cond_low_alpha :
        print('second_cond_grey and second_cond_low_alpha are mutually exclusive')
        return

    if len(colors[0]) == 2  : 
        colors = [c[0] for c in colors] #keep only cell type color, remove control cond color
        if (not second_cond_grey) & (not second_cond_low_alpha) : 
            print('warning : second cond color removed')

    #do reshaping
    if alpha != 1 :
        colors = [(*c[:3], alpha) for c in colors]

    if second_cond_grey : 
        colors = [[c, (*to_rgb('lightgrey'),1)] for c in colors]

    if second_cond_low_alpha : 
        colors = [[c, (*c[:3], second_cond_alpha)] for c in colors]

    return colors

