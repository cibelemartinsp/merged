#%%
import sys, os 
# physion and the merged modules are taken from merged/ (parent of plots/)
try:
    merged_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError: # interactive window
    merged_root = os.path.join(os.path.expanduser('~'), 'code', 'merged')
for path in [merged_root, os.path.join(merged_root, 'physion', 'src')]:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
os.chdir(merged_root)
# forget modules already imported from another copy (e.g. astrid/lab-notebook) in this kernel
for name in list(sys.modules):
    if name.split('.')[0] in ['general_tools', 'create_summaries', 'notebooks', 'physion']:
        module = sys.modules[name]
        origin = getattr(module, '__file__', None) or next(iter(getattr(module, '__path__', None) or []), '')
        if not str(origin).startswith(merged_root):
            del sys.modules[name]

import numpy as np
import string
alphabet_upper = list(string.ascii_uppercase)

from general_tools.general_plot_tools import *
from general_tools.cell_type_color_codes import *
from create_summaries.gOSI import *
from create_summaries.arousal import *


#%%
fig_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD') + '/'
os.makedirs(fig_path, exist_ok=True)
summary_paths, folders, colors= get_working_data_defaults(experimenters = ['Cibele'],
                                                        folders_surnames = ['SST_WT'], 
                                                        color_with_grey=True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#TEMPLATE profil response 1 pop

summary_protocol = 'Tunings' #'Sensitivities' 

mosaic = """
    AAAAAABBD
    AAAAAABBD
    AAAAAACCD
    AAAAAACCD
    """

fig, ax_dict = create_figure(mosaic, figsize = (8,5))
summary_filenames = get_summary_names(folders, summary_paths, summary_protocols = [summary_protocol])
#plot responses
plot_responses(ax_dict["A"],
            summary_filenames, 
            colors,  
            summary_protocol = summary_protocol,
            ) 

#plot resp to visual pies
plot_responsiveness_pies([ax_dict["B"], ax_dict["C"]], summary_filenames, colors, summary_protocol = summary_protocol, total_on = 'nROIs_final')

#plot raw fluorescence amplitude
plot_mean_F_val(ax_dict['D'], folders, colors[0], 'F0', summary_paths = summary_paths,  summary_protocol = summary_protocol)

add_legend_to_plot(ax_dict["A"], folders, colors[0], loc = 'lower left', fontsize = 10) 
fig.savefig(fig_path + 'pop_profile_' + summary_protocol + '_' + ''.join(folders) + '.svg', format = 'svg')


 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# POPULATIONS COMPARAISON : responses, responsiveness, OSIs, (can notably be used for inclufact 2-3, or development comparaisons)

summary_paths, folders, colors= get_working_data_defaults(experimenters = ['Cibele'],
                                                        folders_surnames = ['SST_WT_P15', 'SST_WT_P20', 'SST_WT_P24', 'SST_WT'], 
                                                        color_with_grey=True)
summary_protocol = 'Tunings' #'Sensitivities' 

summary_filenames = get_summary_names(folders, summary_paths, summary_protocols = summary_protocol) #template = ['inclufact_2_3'])
titles = {alphabet_upper[i] : '' for i in range(len(folders))}

if False : #compare one pop with inclusion factor = default, 2, 3
    summary_filenames = get_summary_names(folders, summary_paths, summary_protocols = summary_protocol, template = ['inclufact_2_3'])
    colors = colors*3
    folders = folders*3
    titles = {alphabet_upper[i] : summary_filenames[i].split('/')[-1].split('Tunings')[0] for i in range(len(folders))}

if True : #plot young development comparaison
    titles = {alphabet_upper[i] : summary_filenames[i].split('/')[-1].split('cells')[1] for i in range(len(folders))}

mosaic = "".join(alphabet_upper[:len(folders)])
fig, ax_dict = create_figure(mosaic, figsize = (5*len(folders),5), hspace = 0.15, wspace = 0.15)
#plot responses
for folder, summary_filename, color, letter in zip(folders,
                                                summary_filenames, 
                                                colors,
                                                list(ax_dict.keys())) : 

    plot_responses(ax_dict[letter],
                    [summary_filename], 
                    [color],  
                    summary_protocol = summary_protocol,
                    clip_negative = True,
                    ylims = [0,1.05],
                    grid = True)

    ax_dict[letter].set_title(titles[letter], fontsize = 7)
fig.savefig(fig_path + summary_protocol + '_' + ''.join(folders) + '.svg', format = 'svg')
 
if True : #plot responsiveness

    upper_row = "".join([alphabet_upper[i] for i in np.arange(0,len(folders)*2,2)]) #even numbers
    lower_row = "".join([alphabet_upper[i] for i in np.arange(1,len(folders)*2,2)]) #odd numbers
    mosaic = upper_row + '\n' + lower_row

    fig, ax_dict = create_figure(mosaic, figsize = (2*len(folders),5))
    axes = [ax_dict[letter] for letter in alphabet_upper[:len(folders)*2]]
    colors = reshape_colors_with(colors, second_cond_low_alpha=True)
    plot_responsiveness_pies(axes, summary_filenames, colors, summary_protocol = summary_protocol, total_on = 'nROIs_final')
    fig.savefig(fig_path + 'responsiveness_' + summary_protocol + '_' + ''.join(folders) + '.svg', format = 'svg')


if True : #plot OSI and gOSI
    mosaic = "AB"
    fig, ax_dict = create_figure(mosaic, figsize = (2*len(summary_filenames),5)) 

    x = np.arange(0,len(summary_filenames)*2)
    colors = np.concatenate(colors)
    for letter, osi_type in zip(['A','B'], ['OSI','gOSI']):
        plot_OSI_or_gOSI(ax_dict[letter], summary_filenames, colors, osi_type = osi_type, labels = None, x = x, mean_by = 'sessions')
    fig.savefig(fig_path + 'OSIs_' + summary_protocol + '_' + ''.join(folders) + '.svg', format = 'svg')


 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 1 figure COMPARAISON : multiple tuning responses in same ax and gOSI (can notably be use for deconvolved with original population comparaison)

colors = np.concatenate(reshape_colors_with(colors, second_cond_low_alpha = True, second_cond_alpha=0.3))

#deconvolution
if True : 
    summary_prefixe_names = ['Deconvolved_dFoF_','']
    labels =  [prefixe + folder for prefixe in summary_prefixe_names]
    colors = [[(*to_rgb('black'),1),(*to_rgb('grey'),1)], colors]

#KO 
#folders need also to be selected
if False : 
    summary_prefixe_names = ['']
    labels = folders

mosaic = """
    AAB
    AAB
    """

fig, ax_dict = create_figure(mosaic, figsize = (5 + len(folders)*2 ,5))

summary_filenames = get_summary_names(folders, summary_paths, summary_protocols = 'Tunings', summary_prefixe_names = summary_prefixe_names)

plot_responses(ax_dict['A'],
                summary_filenames,
                colors,  
                summary_protocol = 'Tunings',
                clip_negative = True,
                ylims = [0,1.05])

x = np.arange(0,len(summary_filenames)*2)
colors = np.concatenate(colors)
plot_OSI_or_gOSI(ax_dict['B'], summary_filenames, colors, osi_type = 'gOSI', labels = None, x = x, mean_by = 'sessions')
#add_legend_to_plot(ax_dict["A"], labels, colors, loc = 'lower left',bbox_to_anchor = (0,-0.3), fontsize = 12) 
fig.savefig(fig_path + labels[0] + '.svg', format = 'svg')

 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MULTIPLE CORRECTION FACTORS TUNING OR SENSITIVITIE CURVES 

from create_summaries.multiple_corrfacts import *

neuropil_correction_factors = [0, 0.15, 0.3, 0.55, 0.7, 0.85, 1]

fig = plt.figure(figsize = (7,9))
axes = get_multiple_corrfact_axes_and_implement_mosaic(folders, fig)

for i, (folder, summary_path, folder_colors) in enumerate(zip(folders, summary_paths, colors)) : 

    colors, alphas = get_multiple_corfact_colors_and_alphas(colors[0])
    summary_filenames = get_summary_names(folder, summary_path, summary_protocols = 'Tunings', template = ['multiple_corfact'])
    plot_responses(axes[i][0],
                summary_filenames, 
                colors,  
                alphas = alphas,
                summary_protocol = 'Tunings', #'Sensitivities',
                clip_negative = False,
                #sensitivitie_uncertainty=None
                ) 

    axes[i][1].set_title('neuropil correction factors', fontsize = 13)
    plot_corrfact_legend_color_maps(colors = colors[i], fig = fig,  axes = [axes[i][1], axes[i][2]]) 
    fig.savefig(fig_path + 'multiple_correction_factors_' + ''.join(folders) + '.svg', format = 'svg')
 

if True : #plot associated gOSI (one folder at the time)

    mosaic =  """
        A
        """
    fig, ax_dict = create_figure(mosaic, figsize = (2*len(summary_filenames),5))

    colors, alphas = get_multiple_corfact_colors_and_alphas(colors[0])
    x = [0,1,3,4,6,7,9,10,12,13,15,16,18,19]
    labels = np.concatenate([[str(val)]*2 for val in neuropil_correction_factors])
    colors = reshape_colors_with(colors, second_cond_low_alpha = True, second_cond_alpha=0.3)[0]
    colors = np.concatenate([colors for i in range(len(neuropil_correction_factors))])
    plot_OSI_or_gOSI(ax_dict['A'], summary_filenames, colors, osi_type = 'gOSI', labels = labels, x = x, mean_by = 'sessions')
    
    ax_dict['A'].set_xlabel('neuropil correction factors')

    fig.savefig(fig_path + 'multiple_correction_factors_gOSI_' + ''.join(folders) + '.svg', format = 'svg')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ALL STATES VS REST RESPONSES 

summary_protocol ='Sensitivities' # 'Tunings' #'Sensitivities'
ylims = [-0.15,0.7]#[0,1.05] 

#colors = reshape_colors_with(colors, second_cond_low_alpha = True, second_cond_alpha=0.3)

mosaic = """
    AB
    """

fig, ax_dict = create_figure(mosaic, figsize = (10,6), wspace = 0.1)
for summary_prefixe_name, letter in zip(['','Rest_'], ['A','B']) : 
    summary_filenames = get_summary_names(folders, summary_paths, summary_prefixe_names = [summary_prefixe_name], summary_protocols=summary_protocol)

    plot_responses(ax_dict[letter],
                    summary_filenames,
                    colors,  
                    summary_protocol = summary_protocol,
                    clip_negative = True,
                    ylims = ylims,
                    )
    ax_dict[letter].grid(True, linewidth = 1, axis = 'y')

ax_dict['B'].set_title('Rest', fontsize = 12)

#write rest percentage
perc_ep_rest, std_ep_rest = get_perc_rest_from_rest_summary(summary_filenames, summary_protocol=summary_protocol) 
fig.text(x = 0, y = ylims[0] - 0.05, s = '%' + ' of episode rest :', fontsize = 12)
k=2
for perc, std, color in zip(perc_ep_rest, std_ep_rest, np.concatenate(colors)):
    print(perc, std, color)
    fig.text(x = 0.1*k, y =  ylims[0] - 0.05 , s = "%i +/- %i" % (perc, std), color = color, fontsize = 12)
    k+=1
      
fig.savefig(fig_path + 'arousal_' + summary_protocol + '_' + ''.join(folders) + '.svg', format = 'svg')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RESPONSES OF EACH SESSIONS

summary_protocol = 'Sensitivities' #'Tunings'
ylims = [-0.15,0.85] #[0,1.05] 
summary_filenames = get_summary_names(folders, summary_paths, summary_protocols = summary_protocol)
for i, summary_filename in enumerate(summary_filenames) : 
    fig = plot_individual_responses(summary_filename, summary_protocol = summary_protocol, colors = colors[i], ylims = ylims)
    fig.savefig(fig_path + 'sessions_summary_' + summary_filenames[0].split('/')[-1] + '.svg', format = 'svg')
