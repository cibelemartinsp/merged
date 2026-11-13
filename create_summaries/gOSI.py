import numpy as np
from scipy.stats import sem 
from physion.analysis.protocols.orientation_tuning import selectivity_index
from general_tools.general_tools import get_protocol_keys

def get_cell_gOSI(Radians, Responses) : 
    """
    Responses and Radians are both 1D arrays of dim = (1*8)
    """
    cell_gOSI = np.sqrt(np.sum(Responses*np.cos(Radians))**2 + np.sum(Responses*np.sin(Radians))**2) / np.sum(Responses)

    return cell_gOSI


def get_OSIs_or_gOSIs(Tunings, osi_type = 'gOSI', only_significant = True, clip_negative = True) : 

    OSIs = []

    angles = Tunings[0]['shifted_angle']
    radians = [np.deg2rad(2*(angle + 22.5)) for angle in angles]

    for T in Tunings : 
        session_OSIs = []

        Responses = T['Responses']
        if only_significant : 
            Responses = Responses[T['significant_ROIs']]
        if clip_negative : 
            Responses = np.array([[max(x,0) for x in r ] for r in Responses]) 

        for R in Responses : #warning : do not work with summary with 2d significances
            if osi_type == 'gOSI' : 
                cell_OSI = get_cell_gOSI(radians, R)
            elif osi_type == 'OSI':
                cell_OSI = selectivity_index(angles, R)
            session_OSIs.append(cell_OSI)

        OSIs.append(session_OSIs)
    
    return OSIs


def plot_OSI_or_gOSI(ax, summary_filenames, colors, osi_type = 'gOSI', labels = None, x = None, mean_by = 'sessions', clip_negative = True, add_keys = True): 

    height = []
    height_error = []
    for filename in summary_filenames : 
        if add_keys : 
            keys = [key + '.npy' for key in get_protocol_keys(summary_protocol = 'Tunings')]
        else : 
            keys = ['']
            
        for key in keys : 
            Tunings = np.load(filename + key, allow_pickle = True)

            OSIs = get_OSIs_or_gOSIs(Tunings, osi_type = osi_type, clip_negative = clip_negative)

            if mean_by == 'cells': 
                OSIs = np.concatenate(OSIs)
                height.append(np.nanmean(OSIs))
                height_error.append(np.nanstd(OSIs))
                ax.set_title('mean by cells')

            elif mean_by == 'sessions':
                OSIs = [np.mean(session_gosi) for session_gosi in OSIs]
                height.append(np.nanmean(OSIs))
                height_error.append(sem(OSIs, nan_policy = 'omit'))

    if x is None : 
        x = np.arange(len(height))

    ax.grid(True, linewidth = 1, axis = 'y')
    ax.bar(x = x, height = height, width = 0.75, color = colors)
    ax.errorbar(x=x, y= height, yerr=height_error, fmt='.', elinewidth=1, capthick=1,  capsize = 5, color = 'black')
    if labels is not None : 
        ax.set_xticks(ticks = x, labels = labels, rotation = 45)
    ax.set_ylabel(osi_type)
    ax.set_xticks([])
    ax.set_title('negative clipped')

