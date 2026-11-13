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

def p_to_stars(p) :
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'NS'

def draw_significance_brackets(ax, comparisons, bar_x, bar_top, step = 7, fontsize = 10) :
    """
    comparisons : list of (bar1, bar2, p) for the significant differences
    bar_x, bar_top : dicts bar -> x position and top of the bar (mean + sem)
    draws a bracket with the stars above each pair, stacked so that they do not overlap
    """
    drawn = [] # (x1, x2, y) of the brackets already drawn
    for bar1, bar2, p in sorted(comparisons, key = lambda c: abs(bar_x[c[1]] - bar_x[c[0]])) :
        x1, x2 = sorted([bar_x[bar1], bar_x[bar2]])
        y = max(top for bar, top in bar_top.items() if x1 <= bar_x[bar] <= x2) + step/2
        for dx1, dx2, dy in drawn :
            if dx1 <= x2 and x1 <= dx2 : # overlapping bracket: go above it
                y = max(y, dy + step)
        ax.plot([x1, x1, x2, x2], [y - step/4, y, y, y - step/4], color = 'black', lw = 0.8)
        ax.text((x1 + x2) / 2, y - step/6, p_to_stars(p), ha = 'center', va = 'bottom',
                color = 'black', fontsize = fontsize)
        drawn.append((x1, x2, y))
    return max([y for _, _, y in drawn], default = 0)
#%% CORE PLOTS TEMPLATES : works also with populations comparaisons
#%% choose populations

summary_paths, folders, colors= get_working_data_defaults(experimenters = ['Cibele'],
                                                        folders_surnames = ['GCaMP8m', 'GCaMP8s'], 
                                                        color_with_grey=True)


if False:
    add_legend_to_plot(fig, folders, colors, loc = 'outside lower left') 

#%%%%%%%%%%%%%%%% FLUORESCENCE LEVELS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# sessions whose fluorescence is measured: the ones of the Tunings or of the Sensitivities summaries
#F_summary_protocol = 'Tunings'
F_summary_protocol = 'Sensitivities'

# bar = mean of all ROIs pooled ('ROIs') or mean of the session means ('sessions')
# dots = one per session (mean of its ROIs), always drawn
#F_average_by = 'ROIs'
F_average_by = 'sessions'
F_value = 'F0'

mosaic = "A"
fig, ax_dict = create_figure(mosaic, figsize = (2*len(folders),5))
colors = reshape_colors_with(colors)
ax = ax_dict['A']
rng = np.random.default_rng(0) # same horizontal jitter of the dots at each run
for i, (folder, summary_path) in enumerate(zip(folders, summary_paths)) :
    dFoF_parameters = get_dFoF_params(folder, None)
    dFoF_parameters['with_correctedFluo_and_F0'] = True
    nwb_filenames = get_nwb_filenames_from_summary(folder, summary_path, F_summary_protocol)
    Fval_sessions = [get_fluorescence_level([f], dFoF_parameters = dFoF_parameters, F_value = F_value) for f in nwb_filenames] # one value per ROI, per session
    session_means = np.array([np.mean(F) for F in Fval_sessions])
    Fval_ROIs = np.concatenate(Fval_sessions)
    values = Fval_ROIs if F_average_by == 'ROIs' else session_means
    ax.bar(i, np.mean(values), width = 0.7, color = colors[i])
    ax.errorbar(i, np.mean(values), yerr = stats.sem(values), fmt = '.', elinewidth = 1, capthick = 1, capsize = 5, color = 'black', zorder = 3)
    dark_bar = np.mean(colors[i][:3]) < 0.5 # white text and grey dot edges on dark bars, black on light bars
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(session_means)), session_means, s = 14,
               facecolor = 'white', edgecolor = 'grey' if dark_bar else 'black', lw = 0.8, zorder = 4)
    print('%s: %i sessions, %i ROIs, mean %s = %.1f (ROIs) / %.1f (sessions)' % (folder, len(session_means), len(Fval_ROIs), F_value, np.mean(Fval_ROIs), np.mean(session_means)))
    ax.text(i, 0, '%i sessions\n%i ROIs\n' % (len(session_means), len(Fval_ROIs)), ha = 'center', va = 'bottom', fontsize = 9,
            color = 'white' if dark_bar else 'black')
# group names under the x axis, bold, in the group color (e.g. SST WT, PV KO, GCaMP8m)
def F_group_label(folder) :
    surname = get_group_surname(folder)
    return surname if surname.startswith('GCaMP') else '%s %s' % (surname.split('_')[0], 'KO' if 'KO' in surname else 'WT')
ax.set_xticks(range(len(folders)))
ax.set_xticklabels([F_group_label(f) for f in folders])
ax.tick_params(axis = 'x', length = 0, pad = 6)
ax.spines['bottom'].set_position(('outward', 6))
ax.set_xlim(-0.6, len(folders) - 0.4)
for tick_label, c in zip(ax.get_xticklabels(), colors) : # after set_xlim (it re-creates the labels)
    tick_label.set(color = c, fontsize = 12, fontweight = 'bold', family = 'Arial')
ax.set_ylabel('mean %s (bar: %s, dots: sessions)' % (F_value, F_average_by), fontsize = 11)
ax.set_title('sessions of the %s summaries' % F_summary_protocol, fontsize = 10)

ymin = 0
ymax = 800
ax_dict['A'].set_ylim([ymin, ymax])



#%%%%%%%%%%%%%%%% TUNING CURVES  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


summary_protocol = 'Tunings'
#summary_protocol = 'Sensitivities'

ylims = [0,1.05] #[-0.15,0.7]

colors = reshape_colors_with(colors, second_cond_low_alpha = True, second_cond_alpha=0.3)
summary_filenames = get_summary_names(folders, summary_paths, summary_protocols = summary_protocol)

if summary_protocol == 'Tunings' :
    # same responses as plot_responses (significant ROIs, mean per session, normalized to pref.)
    # drawn with the design of PLOT-Tuning-Dataset.py ("ORIENTATION TUNING CURVES WITH NUMBER ANIMALS")
    # and the N's (ROIs, sessions, animals) of each condition outside of the graph
    clip_negative = False
    uncertainty_sy = 'sem'

    # contrast 0.5 in lightgray as there, unless several groups need to be told apart
    tuning_colors = [[c[0], 'lightgray'] for c in colors] if len(folders) == 1 else colors

    fig, ax = pt.figure(right=20)
    x = np.linspace(-30, 180-30, 100)

    k = 0
    for i, summary_filename in enumerate(summary_filenames) :
        for j, key in enumerate(get_protocol_keys(summary_protocol = summary_protocol)) :

            Tunings = np.load(summary_filename + '%s.npy' % key, allow_pickle=True)
            color = tuning_colors[i][j]

            Responses = get_tuning_responses(Tunings, average_by='sessions') #return only significant resp
            Responses = [r/r[1] for r in Responses]
            if clip_negative :
                Responses = np.array([[max(x, 0) for x in r] for r in Responses])

            # Gaussian Fit
            C, func = fit_gaussian(Tunings[0]['shifted_angle'],
                                    np.nanmean(Responses, axis=0))

            pt.scatter(Tunings[0]['shifted_angle'], np.nanmean(Responses, axis=0),
                       sy=get_uncertainty(Responses, uncertainty_sy = uncertainty_sy),
                       color=color, ax=ax, ms=2)
            ax.plot(x, func(x), lw=2, alpha=.5, color=color)

            # N's: responsive / total ROIs, sessions and animals
            # (sessions without responsive ROI do not enter the mean responses)
            sessions = [t for t in Tunings if np.sum(t['significant_ROIs']) > 0]
            n_resp_ROIs = np.sum([np.sum(t['significant_ROIs']) for t in Tunings])
            n_ROIs = np.sum([t['nROIs_final'] for t in Tunings])
            n_animals = len(set(t['subject'] for t in sessions))
            n_animals_total = len(set(t['subject'] for t in Tunings))

            label = os.path.basename(summary_filename).replace('Tunings_', '') + key
            ax.annotate('%s\nn = %i / %i ROIs (resp. / total)\nN = %i / %i sessions, %i / %i animals' % (
                            label, n_resp_ROIs, n_ROIs, len(sessions), len(Tunings),
                            n_animals, n_animals_total),
                        (1.05, 1), xycoords = 'axes fraction',
                        xytext = (0, -24*k), textcoords = 'offset points', va = 'top',
                        color = 'grey' if color == 'lightgray' else color, # readable text
                        fontsize = 5, family = 'Arial')
            k += 1

    angles = Tunings[0]['shifted_angle']
    pt.set_plot(ax, xticks=angles, yticks=np.arange(3)*0.5, ylim=ylims,
                ylabel='norm. $\\delta$ $\\Delta$F/F',  xlabel='angle ($^o$) from pref.',
                xticks_labels=['%i' % a if (a in [0, 90]) else '' for a in angles])

    # clean publication style (as in PLOT-Tuning-Dataset.py)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    spine_thickness = 0.5
    for side in ['left', 'bottom']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color('black')
        ax.spines[side].set_linewidth(spine_thickness)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    label_font = {'family': 'Arial', 'color': 'black', 'weight': 'normal', 'size': 7}
    ax.set_xlabel(ax.get_xlabel(), fontdict=label_font)
    ax.set_ylabel(ax.get_ylabel(), fontdict=label_font)
    ax.tick_params(axis='both', colors='black', labelsize=6, width=spine_thickness, length=3)

else :
    mosaic = "A"
    fig, ax_dict = create_figure(mosaic, figsize = (5,5))
    plot_responses(ax_dict['A'],
                    summary_filenames,
                    colors,
                    summary_protocol = summary_protocol,
                    clip_negative = False,
                    ylims = ylims)

groups_name = '_'.join(os.path.basename(f).replace('%s_' % summary_protocol, '').rstrip('_') for f in summary_filenames)
fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/%s_%s.svg' % (summary_protocol, groups_name)),
            bbox_inches = 'tight', facecolor = 'white')



#%%%%%%%%%%%%%%%% TUNING CURVES - ONE PANEL PER GROUP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# same calculations as the TUNING CURVES cell above (significant ROIs, mean per session,
# normalized to pref., gaussian fit), drawn side by side: one panel per group,
# contrast 1.0 in the group color and contrast 0.5 in lightgray

summary_protocol = 'Tunings'
ylims = [0,1.05]
clip_negative = False
uncertainty_sy = 'sem'
show_Ns = True # N's (ROIs, sessions, animals) of each condition below the figure

colors = reshape_colors_with(colors, second_cond_grey = True)
summary_filenames = get_summary_names(folders, summary_paths, summary_protocols = summary_protocol)

fig, axes = pt.plt.subplots(1, len(summary_filenames), sharey=True,
                            figsize=(1.6*len(summary_filenames), 1.6))
axes = np.atleast_1d(axes)
fig.patch.set_facecolor('white')
x = np.linspace(-30, 180-30, 100)
spine_thickness = 0.8
label_font = {'family': 'Arial', 'color': 'black', 'weight': 'normal', 'size': 8}

all_Ns = []
for i, (summary_filename, ax) in enumerate(zip(summary_filenames, axes)) :
    Ns = []
    for j, key in enumerate(get_protocol_keys(summary_protocol = summary_protocol)) :

        Tunings = np.load(summary_filename + '%s.npy' % key, allow_pickle=True)
        color = colors[i][0] if j == 0 else '#b5b5b5'

        Responses = get_tuning_responses(Tunings, average_by='sessions') #return only significant resp
        Responses = [r/r[1] for r in Responses]
        if clip_negative :
            Responses = np.array([[max(x, 0) for x in r] for r in Responses])

        # Gaussian Fit
        C, func = fit_gaussian(Tunings[0]['shifted_angle'],
                                np.nanmean(Responses, axis=0))

        ax.plot(x, func(x), lw=2, alpha=.5, color=color)
        # small markers so that the sem bars (~0.02-0.04) stay visible, drawn on top of the fit
        ax.errorbar(Tunings[0]['shifted_angle'], np.nanmean(Responses, axis=0),
                    yerr=get_uncertainty(Responses, uncertainty_sy = uncertainty_sy),
                    fmt='o', ms=2.5, color=color, ecolor=color, elinewidth=0.4, capsize=1,
                    capthick=0.4, zorder=3)

        # N's: responsive / total ROIs, sessions and animals
        sessions = [t for t in Tunings if np.sum(t['significant_ROIs']) > 0]
        Ns.append('%s: n=%i/%i ROIs, N=%i/%i sess., %i/%i anim.' % (
                    key.replace('_', ''),
                    np.sum([np.sum(t['significant_ROIs']) for t in Tunings]),
                    np.sum([t['nROIs_final'] for t in Tunings]),
                    len(sessions), len(Tunings),
                    len(set(t['subject'] for t in sessions)),
                    len(set(t['subject'] for t in Tunings))))

    label = os.path.basename(summary_filename).replace('Tunings_', '').rstrip('_')
    print(label, '\n   ' + '\n   '.join(Ns))
    all_Ns.append((label + ' -- ' + ' | '.join(Ns), colors[i][0]))

    # x axis: ticks at all angles, labels only at 0 and 90
    angles = Tunings[0]['shifted_angle']
    ax.set_xticks(angles)
    ax.set_xticklabels(['%i' % a if (a in [0, 90]) else '' for a in angles])
    ax.set_xlim([angles[0]-20, angles[-1]+20])
    ax.set_ylim(ylims)
    ax.set_facecolor('white')
    for side in ['top', 'right'] :
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(spine_thickness)
    ax.spines['bottom'].set_position(('outward', 6))
    ax.tick_params(axis='both', colors='black', labelsize=7, width=spine_thickness, length=3)
    if i == 0 :
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(spine_thickness)
        ax.spines['left'].set_position(('outward', 6))
        ax.set_yticks(np.arange(3)*0.5)
        ax.set_ylabel('norm. ΔF/F', fontdict=label_font)
    else :
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', left=False)

fig.subplots_adjust(wspace=0.15)
# x label centered below all the panels
x_center = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
fig.canvas.draw()
y_ticklabels = fig.transFigure.inverted().transform(
                    axes[0].xaxis.get_tightbbox(fig.canvas.get_renderer()))[0, 1] # bottom of the tick labels
fig.text(x_center, y_ticklabels - 0.03, 'angle (°) from pref.', ha='center', va='top', fontdict=label_font)

# N's of each group and condition below the figure, in the group color
if show_Ns :
    for k, (text, color) in enumerate(all_Ns) :
        fig.text(axes[0].get_position().x0, y_ticklabels - 0.22 - 0.06*k, text, ha='left', va='top',
                 fontsize=5, family='Arial', color=color)

# dotted grey separators between the panels
for ax_left, ax_right in zip(axes[:-1], axes[1:]) :
    x_sep = (ax_left.get_position().x1 + ax_right.get_position().x0) / 2
    fig.add_artist(pt.plt.Line2D([x_sep, x_sep], [ax_left.get_position().y0, ax_left.get_position().y1],
                                 transform=fig.transFigure, color='grey', lw=0.8, ls=':'))

groups_name = '_'.join(os.path.basename(f).replace('%s_' % summary_protocol, '').rstrip('_') for f in summary_filenames)
fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/%s_panels_%s.svg' % (summary_protocol, groups_name)),
            bbox_inches = 'tight', facecolor = 'white')



#%%%%%%%%%%%%%%%% TUNING CURVES - ONE ROW PER CONTRAST, GROUPS OVERLAID %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# same calculations as the TUNING CURVES cell above (significant ROIs, mean per session,
# normalized to pref., gaussian fit)
# rows   : contrast 1.0 (top) and contrast 0.5 (bottom)
# columns: one panel per set of groups, the groups of a panel overlaid, each in its own color
#          (e.g. SST WT vs SST KO | PV WT vs PV KO)

summary_protocol = 'Tunings'
ylims = [0,1.05]
clip_negative = False
uncertainty_sy = 'sem'
show_Ns = True # N's (ROIs, sessions, animals) of each group and contrast below the figure
# which groups go together in a panel: indices in folders (order of "choose populations"),
# e.g. [[0, 1], [2, 3]] ; None -> one panel per cell type (SST-cells, PV-cells, ...)
panels = None

colors = reshape_colors_with(colors, second_cond_grey = True)
summary_filenames = get_summary_names(folders, summary_paths, summary_protocols = summary_protocol)
group_names = [os.path.basename(f).replace('%s_' % summary_protocol, '').rstrip('_') for f in summary_filenames]
if panels is None :
    cell_types = list(dict.fromkeys(name.split('_')[0] for name in group_names)) # order of folders
    panels = [[i for i, name in enumerate(group_names) if name.split('_')[0] == ct] for ct in cell_types]
protocol_keys = get_protocol_keys(summary_protocol = summary_protocol) # contrast 1.0, contrast 0.5

fig, axes = pt.plt.subplots(len(protocol_keys), len(panels), sharey=True,
                            figsize=(1.6*len(panels), 2.15*len(protocol_keys)), squeeze=False) # same panel size as ONE PANEL PER GROUP
fig.patch.set_facecolor('white')
x = np.linspace(-30, 180-30, 100)
spine_thickness = 0.8
label_font = {'family': 'Arial', 'color': 'black', 'weight': 'normal', 'size': 8}

all_Ns = []
for row, key in enumerate(protocol_keys) :
    for col, panel in enumerate(panels) :
        ax = axes[row, col]
        for i in panel :

            Tunings = np.load(summary_filenames[i] + '%s.npy' % key, allow_pickle=True)
            color = colors[i][0]

            Responses = get_tuning_responses(Tunings, average_by='sessions') #return only significant resp
            Responses = [r/r[1] for r in Responses]
            if clip_negative :
                Responses = np.array([[max(x, 0) for x in r] for r in Responses])

            # Gaussian Fit
            C, func = fit_gaussian(Tunings[0]['shifted_angle'],
                                    np.nanmean(Responses, axis=0))

            ax.plot(x, func(x), lw=2, alpha=.5, color=color)
            ax.errorbar(Tunings[0]['shifted_angle'], np.nanmean(Responses, axis=0),
                        yerr=get_uncertainty(Responses, uncertainty_sy = uncertainty_sy),
                        fmt='o', ms=2.5, color=color, ecolor=color, elinewidth=0.4, capsize=1,
                        capthick=0.4, zorder=3)

            # N's: responsive / total ROIs, sessions and animals
            sessions = [t for t in Tunings if np.sum(t['significant_ROIs']) > 0]
            N = '%s%s: n=%i/%i ROIs, N=%i/%i sess., %i/%i anim.' % (
                    group_names[i], key,
                    np.sum([np.sum(t['significant_ROIs']) for t in Tunings]),
                    np.sum([t['nROIs_final'] for t in Tunings]),
                    len(sessions), len(Tunings),
                    len(set(t['subject'] for t in sessions)),
                    len(set(t['subject'] for t in Tunings)))
            print(N)
            all_Ns.append((N, color))

        # x axis: ticks at all angles, labels only at 0 and 90
        angles = Tunings[0]['shifted_angle']
        ax.set_xticks(angles)
        ax.set_xticklabels(['%i' % a if (a in [0, 90]) else '' for a in angles])
        ax.set_xlim([angles[0]-20, angles[-1]+20])
        ax.set_ylim(ylims)
        ax.set_facecolor('white')
        for side in ['top', 'right'] :
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(spine_thickness)
        ax.spines['bottom'].set_position(('outward', 6))
        ax.tick_params(axis='both', colors='black', labelsize=7, width=spine_thickness, length=3)
        if col == 0 :
            ax.spines['left'].set_color('black')
            ax.spines['left'].set_linewidth(spine_thickness)
            ax.spines['left'].set_position(('outward', 6))
            ax.set_yticks(np.arange(3)*0.5)
            ax.set_ylabel('norm. $\\delta$ ΔF/F', fontdict=label_font)
            # contrast of the row, bold, on the left
            ax.annotate('Contrast %s' % key.split('-')[-1], (-0.8, 0.5), xycoords='axes fraction',
                        rotation=90, ha='center', va='center', fontsize=10, fontweight='bold', family='Arial')
        else :
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', left=False)

fig.subplots_adjust(wspace=0.15, hspace=0.7)

# as in the ONE PANEL PER GROUP cell: x label centered below all the panels of each row,
# and dotted grey separators between the panels
fig.canvas.draw()
x_center = (axes[0, 0].get_position().x0 + axes[0, -1].get_position().x1) / 2
for row_axes in axes :
    y_ticklabels = fig.transFigure.inverted().transform(
                        row_axes[0].xaxis.get_tightbbox(fig.canvas.get_renderer()))[0, 1] # bottom of the tick labels
    fig.text(x_center, y_ticklabels - 0.01, 'angle (°) from pref.', ha='center', va='top', fontdict=label_font)
    for ax_left, ax_right in zip(row_axes[:-1], row_axes[1:]) :
        x_sep = (ax_left.get_position().x1 + ax_right.get_position().x0) / 2
        fig.add_artist(pt.plt.Line2D([x_sep, x_sep], [ax_left.get_position().y0, ax_left.get_position().y1],
                                     transform=fig.transFigure, color='grey', lw=0.8, ls=':'))

# N's of each group and contrast below the figure, in the group color
if show_Ns :
    for k, (text, color) in enumerate(all_Ns) :
        fig.text(axes[0, 0].get_position().x0, axes[-1, 0].get_position().y0 - 0.11 - 0.022*k, text, ha='left', va='top',
                 fontsize=5, family='Arial', color=color)

fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/%s_contrast-rows_%s.svg' % (summary_protocol, '_'.join(group_names))),
            bbox_inches = 'tight', facecolor = 'white')






#%%%%%%%%%%%%%%%% OSIs and gOSIs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# one pair of violins per group (contrast 1.0 in the group color, contrast 0.5 in grey),
# the group underlined with its number of animals below
# + Kruskal-Wallis and post-hoc Mann-Whitney (Bonferroni) on the plotted values
from create_summaries.gOSI import get_OSIs_or_gOSIs
from scipy.stats import kruskal, mannwhitneyu
from matplotlib.colors import to_rgb
import matplotlib.transforms as transforms
import itertools

osi_type = 'gOSI'     # 'gOSI' or 'OSI' (= (pref-orth)/(pref+orth), the selectivity index)
mean_by = 'sessions'  # 'sessions' (one mean value per session) or 'cells'
clip_negative = True
group_labels = None   # e.g. ['SST-cre', 'PV-cre', 'C57Bl/6\nPyramidal'], None -> from the folder names
stars_on_plot = 'all'     # 'all' significant pairs or 'within' (contrast 1.0 vs 0.5 of each group only)
group_spacing = 2.2       # distance between the pairs of violins (1 = inside a pair)

summary_filenames = get_summary_names(folders, summary_paths, summary_protocols = 'Tunings')
keys = get_protocol_keys(summary_protocol = 'Tunings') # [contrast-1.0, contrast-0.5]
group_colors = [c[0] if len(c) == 2 else c for c in colors] # group color, without the grey of the control cond
if group_labels is None :
    # cell type on the first line, genotype on the second
    group_labels = [get_group_surname(f).replace('_', '\n', 1).replace('_', ' ') for f in folders]

values, labels, n_animals = [], [], []
for summary_filename in summary_filenames :
    for key in keys :
        Tunings = np.load(summary_filename + key + '.npy', allow_pickle = True)
        OSIs = get_OSIs_or_gOSIs(Tunings, osi_type = osi_type, clip_negative = clip_negative) # only significant ROIs
        if mean_by == 'sessions' :
            # sessions without responsive ROI are left out
            OSIs = [np.mean(session_OSIs) for session_OSIs in OSIs if len(session_OSIs) > 0]
        elif mean_by == 'cells' :
            OSIs = np.concatenate(OSIs)
        values.append(np.array(OSIs))
        labels.append(os.path.basename(summary_filename).replace('Tunings_', '') + key)
    # animals with at least one responsive ROI (contrast 1.0)
    Tunings = np.load(summary_filename + keys[0] + '.npy', allow_pickle = True)
    n_animals.append(len(set(t['subject'] for t in Tunings if np.sum(t['significant_ROIs']) > 0)))

# positions : pairs of violins separated by a gap
x_positions = np.concatenate([[group_spacing*i, group_spacing*i + 1] for i in range(len(folders))])

fig, ax = pt.figure(ax_scale = (0.45*len(folders), 1.2))
for x, v, k in zip(x_positions, values, itertools.cycle(range(len(keys)))) :
    group_color = group_colors[int(x // group_spacing)][:3]
    if k == 0 : # contrast 1.0 : light group color, group color for mean and quartiles
        face_color = 0.35*np.array(to_rgb(group_color)) + 0.65*np.ones(3)
        line_color = group_color
    else :      # contrast 0.5 : grey
        face_color, line_color = 'gainsboro', 'dimgrey'
    parts = ax.violinplot(v, positions = [x], widths = 0.8, showextrema = False, points = 100)
    for pc in parts['bodies'] :
        pc.set_facecolor(face_color)
        pc.set_edgecolor('none')
        pc.set_alpha(1.0)
    ax.hlines(np.quantile(v, [0.25, 0.5, 0.75]), x - 0.18, x + 0.18, color = line_color, lw = 0.8)
    ax.plot([x], [np.mean(v)], 'o', ms = 2.5, color = line_color)

pt.set_plot(ax, ['left'], yticks = np.arange(3)*0.5, ylabel = osi_type)
ax.set_xlim([x_positions[0] - 0.6, x_positions[-1] + 0.6])

# groups underlined in their color, number of animals below
trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
for i, (label, n, color) in enumerate(zip(group_labels, n_animals, group_colors)) :
    x0, x1 = group_spacing*i - 0.45, group_spacing*i + 1.45
    ax.plot([x0, x1], [-0.04, -0.04], color = color, lw = 1, transform = trans, clip_on = False)
    ax.text((x0 + x1) / 2, -0.1, '%i %s' % (n, label), ha = 'center', va = 'top', transform = trans,
            color = color, fontsize = 6, fontweight = 'bold', family = 'Arial')


# ====================================================
# STATISTICS
# ====================================================

print("="*70)
print("%s - %s-LEVEL STATISTICS" % (osi_type, mean_by.upper()))
print("="*70)

print("\nDataset info (%s):" % mean_by)
for label, v in zip(labels, values) :
    print("  %s: N = %i %s, %s = %.2f +/- %.2f (mean +/- sem)" % (label, len(v), mean_by, osi_type, np.mean(v), sem(v)))

H, p = kruskal(*values)
print("\nKruskal-Wallis H-test (Global): H = %.4f, p = %.4g" % (H, p))

sig_pairs = []
if p < 0.05 :
    print("\nPost-hoc Pairwise Comparisons (Mann-Whitney U, Bonferroni Corrected):")
    # only pairs that differ by one factor : same group at both contrasts, or same contrast in two groups
    pairs = [(i, j) for i, j in itertools.combinations(range(len(values)), 2)
                if (i // len(keys) == j // len(keys)) or (i % len(keys) == j % len(keys))]
    for i, j in pairs :
        _, p_raw = mannwhitneyu(values[i], values[j], alternative = "two-sided")
        p_adjusted = min(p_raw * len(pairs), 1.0)
        print("  %s vs %s: p = %.4g [%s]" % (labels[i], labels[j], p_adjusted, "*" if p_adjusted < 0.05 else "NS"))
        if p_adjusted < 0.05 :
            sig_pairs.append((i, j))
else :
    print("\nResult is not significant. No post-hoc tests performed.")

print("="*70)


# --- AESTHETIC ADJUSTMENTS (as PLOT-Tuning-Dataset.py) ---
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
spine_thickness = 0.5
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(spine_thickness)

label_font = {'family': 'Arial', 'color': 'black', 'weight': 'normal', 'size': 7}
ax.set_ylabel(ax.get_ylabel(), fontdict = label_font)
ax.tick_params(axis = 'y', colors = 'black', labelsize = 6, width = spine_thickness, length = 3)

# --- SIGNIFICANCE MARKERS (all the tests are printed above) ---
def add_star(i, j, y) :
    ax.plot([x_positions[i], x_positions[j]], [y, y], color = 'black', lw = spine_thickness, clip_on = False)
    ax.text((x_positions[i] + x_positions[j]) / 2.0, y - 0.04, '*', ha = 'center', va = 'bottom',
            fontsize = 7, color = 'black', clip_on = False)

within_pairs = [(i, j) for i, j in sig_pairs if i // len(keys) == j // len(keys)]
between_pairs = [(i, j) for i, j in sig_pairs if i // len(keys) != j // len(keys)]

# contrast 1.0 vs 0.5 : one bar just above each pair
for i, j in within_pairs :
    add_star(i, j, max(np.max(values[i]), np.max(values[j])) + 0.05)

# between groups : stacked bars above all the violins, closest pairs first
if stars_on_plot == 'all' :
    y = np.max(np.concatenate(values)) + (0.13 if within_pairs else 0.05)
    for i, j in sorted(between_pairs, key = lambda pair : pair[1] - pair[0]) :
        add_star(i, j, y)
        y += 0.09

fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/%s.svg' % osi_type), bbox_inches = 'tight')



#%%%%%%%%%%%%%%%% OSIs and gOSIs - ONE ROW PER CONTRAST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# same values as the OSIs and gOSIs cell above (only significant ROIs, mean per session),
# laid out as the TUNING CURVES - ONE ROW PER CONTRAST cell:
# rows : contrast 1.0 (top) and contrast 0.5 (bottom)
# one violin per group in its own color, the groups of a same cell type side by side (e.g. SST WT, SST KO | PV WT, PV KO)
# + per contrast: Kruskal-Wallis and post-hoc Mann-Whitney (Bonferroni)
from create_summaries.gOSI import get_OSIs_or_gOSIs
from scipy.stats import kruskal, mannwhitneyu
from matplotlib.colors import to_rgb
import itertools

osi_type = 'gOSI'     # 'gOSI' or 'OSI' (= (pref-orth)/(pref+orth), the selectivity index)
mean_by = 'sessions'  # 'sessions' (one mean value per session) or 'cells'
clip_negative = True
ylabel = None        # None -> osi_type ('OSI' or 'gOSI'), or any text e.g. 'Select. Index'
# which groups go together: indices in folders (order of "choose populations"),
# e.g. [[0, 1], [2, 3]] ; None -> one set per cell type (SST-cells, PV-cells, ...)
panels = None
compare = 'panel'     # stars for 'panel' (groups of a same set only, e.g. WT vs KO) or 'all' pairs of groups
panel_gap = 0.6       # extra space between two sets of groups

summary_filenames = get_summary_names(folders, summary_paths, summary_protocols = 'Tunings')
keys = get_protocol_keys(summary_protocol = 'Tunings') # [contrast-1.0, contrast-0.5]
group_colors = [c[0] if len(c) == 2 else c for c in colors] # group color, without the grey of the control cond
group_names = [os.path.basename(f).replace('Tunings_', '').rstrip('_') for f in summary_filenames]
if panels is None :
    cell_types = list(dict.fromkeys(name.split('_')[0] for name in group_names)) # order of folders
    panels = [[i for i, name in enumerate(group_names) if name.split('_')[0] == ct] for ct in cell_types]
order = [i for panel in panels for i in panel]
# x position of each group: one per group, a gap between the sets
x_positions = {}
x = 0
for panel in panels :
    for i in panel :
        x_positions[i] = x
        x += 1
    x += panel_gap

# axes size in inches as in the OSIs and gOSIs cell above (0.29 in per violin step, 0.66 in high)
x_range = max(x_positions.values()) + 1.2
ax_width, ax_height, row_gap = 0.29*x_range, 0.66, 0.45
left, right, bottom, top = 0.75, 0.1, 0.15, 0.25
fig_width = left + ax_width + right
fig_height = bottom + len(keys)*ax_height + (len(keys)-1)*row_gap + top
fig, axes = pt.plt.subplots(len(keys), 1, figsize = (fig_width, fig_height), squeeze = False)
fig.subplots_adjust(left = left/fig_width, right = 1 - right/fig_width,
                    bottom = bottom/fig_height, top = 1 - top/fig_height, hspace = row_gap/ax_height)
axes = axes[:, 0]
fig.patch.set_facecolor('white')
spine_thickness = 0.8
label_font = {'family': 'Arial', 'color': 'black', 'weight': 'normal', 'size': 8}

print("="*70)
print("%s - %s-LEVEL STATISTICS, PER CONTRAST" % (osi_type, mean_by.upper()))
for ax, key in zip(axes, keys) :

    values = {}
    for i in order :
        Tunings = np.load(summary_filenames[i] + key + '.npy', allow_pickle = True)
        OSIs = get_OSIs_or_gOSIs(Tunings, osi_type = osi_type, clip_negative = clip_negative) # only significant ROIs
        if mean_by == 'sessions' :
            # sessions without responsive ROI are left out
            OSIs = [np.mean(session_OSIs) for session_OSIs in OSIs if len(session_OSIs) > 0]
        elif mean_by == 'cells' :
            OSIs = np.concatenate(OSIs)
        values[i] = np.array(OSIs)

        # violin: light group color, group color for the mean, black quartiles
        x = x_positions[i]
        face_color = 0.35*np.array(to_rgb(group_colors[i][:3])) + 0.65*np.ones(3)
        parts = ax.violinplot(values[i], positions = [x], widths = 0.8, showextrema = False, points = 100)
        for pc in parts['bodies'] :
            pc.set_facecolor(face_color)
            pc.set_edgecolor('none')
            pc.set_alpha(1.0)
        ax.hlines(np.quantile(values[i], [0.25, 0.5, 0.75]), x - 0.18, x + 0.18, color = 'black', lw = 0.8)
        ax.plot([x], [np.mean(values[i])], 'o', ms = 2.5, color = group_colors[i])

    # statistics at this contrast
    print("\n--- %s ---" % key.replace('_', ''))
    for i in order :
        print("  %s: N = %i %s, %s = %.2f +/- %.2f (mean +/- sem)" % (
              group_names[i], len(values[i]), mean_by, osi_type, np.mean(values[i]), sem(values[i])))
    if len(order) > 2 :
        H, p = kruskal(*[values[i] for i in order])
        print("  Kruskal-Wallis H = %.4f, p = %.4g" % (H, p))
    if compare == 'panel' :
        pairs = [pair for panel in panels for pair in itertools.combinations(panel, 2)]
    else :
        pairs = list(itertools.combinations(order, 2))
    sig_pairs = []
    for i, j in pairs :
        _, p_raw = mannwhitneyu(values[i], values[j], alternative = "two-sided")
        p_adjusted = min(p_raw * len(pairs), 1.0)
        print("  %s vs %s: Mann-Whitney p = %.4g (Bonferroni) [%s]" % (group_names[i], group_names[j], p_adjusted,
              p_to_stars(p_adjusted)))
        if p_adjusted < 0.05 :
            sig_pairs.append((i, j, p_adjusted))

    # stars above the violins, closest pairs first
    y = max(np.max(v) for v in values.values()) + 0.05
    for i, j, p in sorted(sig_pairs, key = lambda pair : abs(x_positions[pair[1]] - x_positions[pair[0]])) :
        ax.plot([x_positions[i], x_positions[j]], [y, y], color = 'black', lw = spine_thickness, clip_on = False)
        ax.text((x_positions[i] + x_positions[j]) / 2, y - 0.03, p_to_stars(p), ha = 'center', va = 'bottom',
                fontsize = 7, color = 'black', clip_on = False)
        y += 0.1

    # style as the TUNING CURVES - ONE ROW PER CONTRAST cell
    ax.set_xlim([-0.6, max(x_positions.values()) + 0.6])
    ax.set_ylim([0, 1.05])
    ax.set_yticks(np.arange(3)*0.5)
    ax.set_xticks([])
    ax.set_facecolor('white')
    for side in ['top', 'right', 'bottom'] :
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(spine_thickness)
    ax.spines['left'].set_position(('outward', 6))
    ax.tick_params(axis = 'y', colors = 'black', labelsize = 7, width = spine_thickness, length = 3)
    ax.set_ylabel(osi_type if ylabel is None else ylabel, fontdict = label_font)
    ax.annotate('Contrast %s' % key.split('-')[-1], (0, 0.5), xycoords = 'axes fraction',
                xytext = (-42, 0), textcoords = 'offset points', rotation = 90, ha = 'center', va = 'center', fontsize = 10, fontweight = 'bold', family = 'Arial')
print("="*70)

fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/%s_contrast-rows_%s.svg' % (osi_type, '_'.join(group_names))),
            bbox_inches = 'tight', facecolor = 'white')










#%%%%%%%%%%%%%%%%  RESPONSIVENESS TUNING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mosaic = "".join(alphabet_upper[:len(folders)*2])
fig, ax_dict = create_figure(mosaic, figsize = (2*len(folders),5))
#plot resp to visual pies
summary_filenames = get_summary_names(folders, summary_paths, summary_protocols = 'Tunings')
colors = reshape_colors_with(colors, second_cond_low_alpha=True)
plot_responsiveness_pies(list(ax_dict.values()), summary_filenames, colors, summary_protocol = 'Tunings', total_on = 'nROIs_final')

# design: '% resp' labels smaller and centered above each pie, so that they do not overlap
pie_axes = list(ax_dict.values())
for ax in pie_axes :
    for label in ax.texts :
        if label.get_text().endswith('resp') :
            label.set_text(label.get_text().replace(' \n resp', '\nresp'))
            label.xy = label.xyann = (0, 1.1)
            label.set(fontsize = 9, ha = 'center', va = 'bottom', linespacing = 1.0)

# group legend above each pair of pies: colored line, then 'N cell type' / 'genotype' in the group color (N = animals)
fig.canvas.draw()
n_keys = len(get_protocol_keys(summary_protocol = 'Tunings'))
for i, summary_filename in enumerate(summary_filenames) :
    group_label = os.path.basename(summary_filename).replace('Tunings_', '').rstrip('_')
    n_animals = len(set(T['subject'] for T in np.load(summary_filename + '%s.npy' % get_protocol_keys(summary_protocol = 'Tunings')[0],
                                                     allow_pickle = True)))
    cell_type = group_label.split('_')[0].replace('-cells', '')
    genotype = group_label.split('_')[1].replace('cond-', '').replace('-', ' ')
    group_axes = pie_axes[i*n_keys:(i+1)*n_keys]
    # in figure fraction: from the left of the first pie to the right of the last one, above the '% resp' labels
    x0, x1 = group_axes[0].get_position().x0, group_axes[-1].get_position().x1
    labels_top = max(t.get_window_extent(fig.canvas.get_renderer()).y1 for ax in group_axes for t in ax.texts)
    y_text = fig.transFigure.inverted().transform((0, labels_top))[1] + 8/72/fig.get_figheight()
    label = fig.text((x0 + x1)/2, y_text, '%i %s\n%s' % (n_animals, cell_type, genotype), color = colors[i][0],
                     ha = 'center', va = 'bottom', fontsize = 9, fontweight = 'bold', family = 'Arial', linespacing = 1.0)
    text_top = fig.transFigure.inverted().transform((0, label.get_window_extent(fig.canvas.get_renderer()).y1))[1]
    # thin rectangle rather than a line: a flat line has no height and is cropped by bbox_inches = 'tight'
    fig.add_artist(mpatches.Rectangle((x0 + 0.1*(x1 - x0), text_top + 2/72/fig.get_figheight()), 0.8*(x1 - x0),
                                      1.5/72/fig.get_figheight(), color = colors[i][0], lw = 0, transform = fig.transFigure))



#%%%%%%%%%%%%%%%%  RESPONSIVENESS TUNING - BARS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# same significance as the pies above (significant_ROIs, in % of nROIs_final), as bars:
# per group, contrast 1.0 in the group color and contrast 0.5 in lightgray (mean +/- sem over sessions)

total_on = 'nROIs_final'

summary_filenames = get_summary_names(folders, summary_paths, summary_protocols = 'Tunings')
colors = reshape_colors_with(colors, second_cond_grey=True) # contrast 0.5 in lightgray
keys = get_protocol_keys(summary_protocol = 'Tunings')
bar_width = 0.8 / len(keys)

fig, ax = pt.plt.subplots(figsize = (1.6*len(summary_filenames) + 0.5, 3))
fig.patch.set_facecolor('white')

group_labels = []
perc_values, perc_datafiles = {}, {} # per (group, contrast): % resp. of each session, for the statistics
bar_x, bar_top = {}, {} # per (group, contrast): position and top of the bar, for the stars
for i, summary_filename in enumerate(summary_filenames) :
    group_labels.append(os.path.basename(summary_filename).replace('Tunings_', '').rstrip('_'))
    for j, key in enumerate(keys) :

        summary = np.load(summary_filename + '%s.npy' % key, allow_pickle=True)
        perc = np.array([100 * np.sum(T['significant_ROIs']) / T[total_on] for T in summary]) # per session
        perc_values[i, j] = perc
        perc_datafiles[i, j] = [T['datafile'] for T in summary]

        x = i + (j - (len(keys) - 1) / 2) * bar_width
        bar_x[i, j], bar_top[i, j] = x, perc.mean() + sem(perc)
        ax.bar(x, perc.mean(), width = bar_width * 0.85, color = colors[i][j])
        ax.errorbar(x, perc.mean(), yerr = sem(perc), fmt = 'none', ecolor = 'black', elinewidth = 3, capsize = 0)
        print('%s%s: %.1f +/- %.1f %% resp. (n = %i ROIs, N = %i sessions)' % (
                group_labels[-1], key, perc.mean(), sem(perc), np.sum([T[total_on] for T in summary]), len(summary)))

ax.set_xticks(np.arange(len(summary_filenames)))
ax.set_xticklabels([' '.join(g.split('_')[:2]) for g in group_labels]) # short names, e.g. 'SST-cells WT'
ax.set_ylabel('% resp.', fontsize = 11, color = 'black')
ax.set_yticks([0, 30, 60, 90])
ax.set_facecolor('white')
for side in ['top', 'right'] :
    ax.spines[side].set_visible(False)
for side in ['left', 'bottom'] :
    ax.spines[side].set_color('black')
    ax.spines[side].set_linewidth(1)
    ax.spines[side].set_position(('outward', 6))
ax.tick_params(axis = 'both', colors = 'black', labelsize = 10)
ax.legend(handles = [pt.plt.Rectangle((0, 0), 1, 1, color = 'black'),
                     pt.plt.Rectangle((0, 0), 1, 1, color = 'lightgray')],
          labels = [k.replace('_', '') for k in keys],
          loc = 'upper left', bbox_to_anchor = (1.01, 1), frameon = False, fontsize = 8, labelcolor = 'black')

# ---------- statistics (on the % resp. per session) ----------
from scipy.stats import kruskal, mannwhitneyu, wilcoxon
import itertools

print("\n" + "="*70)
print("RESPONSIVENESS TUNING - STATISTICS (%% resp. per session, on %s)" % total_on)
print("="*70)
comparisons = [] # significant differences, drawn as stars on the figure

# 1) between groups, at each contrast: Kruskal-Wallis, then Mann-Whitney U (Bonferroni)
for j, key in enumerate(keys) :
    print("\n--- %s : between groups ---" % key.replace('_', ''))
    values = [perc_values[i, j] for i in range(len(summary_filenames))]
    if len(values) < 2 :
        print("  only one group: no comparison")
        continue
    H, p = kruskal(*values)
    print("  Kruskal-Wallis H-test: H = %.4f, p = %.4g" % (H, p))
    if p < 0.05 :
        print("  Post-hoc Pairwise Comparisons (Mann-Whitney U, Bonferroni Corrected):")
        pairs = list(itertools.combinations(range(len(values)), 2))
        for i1, i2 in pairs :
            _, p_raw = mannwhitneyu(values[i1], values[i2], alternative = "two-sided")
            p_adjusted = min(p_raw * len(pairs), 1.0)
            print("    %s vs %s: p = %.4g [%s]" % (group_labels[i1], group_labels[i2], p_adjusted,
                                                  p_to_stars(p_adjusted)))
            if p_adjusted < 0.05 :
                comparisons.append(((i1, j), (i2, j), p_adjusted))
    else :
        print("  Result is not significant. No post-hoc tests performed.")

# 2) within each group, contrast 1.0 vs 0.5: Wilcoxon signed-rank (same sessions, paired by datafile)
print("\n--- %s vs %s : within each group (Wilcoxon signed-rank, paired sessions) ---" % tuple(k.replace('_', '') for k in keys))
for i in range(len(summary_filenames)) :
    common = [f for f in perc_datafiles[i, 0] if f in perc_datafiles[i, 1]]
    v0 = np.array([perc_values[i, 0][perc_datafiles[i, 0].index(f)] for f in common])
    v1 = np.array([perc_values[i, 1][perc_datafiles[i, 1].index(f)] for f in common])
    if np.all(v0 == v1) :
        print("  %s: identical values in both contrasts, no test" % group_labels[i])
        continue
    W, p = wilcoxon(v0, v1)
    print("  %s: N = %i sessions, W = %.1f, p = %.4g [%s]" % (group_labels[i], len(common), W, p,
                                                             p_to_stars(p)))
    if p < 0.05 :
        comparisons.append(((i, 0), (i, 1), p))
print("="*70)

# stars on top of the significant differences
y_max = draw_significance_brackets(ax, comparisons, bar_x, bar_top, step = 12)
ax.set_ylim([0, max(100, y_max + 8)])
ax.spines['left'].set_bounds(0, 90)

# group legend on the x axis: colored line, then 'N cell type' / 'genotype' in the group color (N = animals)
ax.set_xticklabels([])
ax.tick_params(axis = 'x', length = 0)
ax.spines['bottom'].set_visible(False) # no x axis line: the group lines below replace it
import matplotlib.transforms as transforms
legend_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
# below the x axis (moved 6 points outward), in points from the bottom of the axes
line_transform = legend_transform + transforms.ScaledTranslation(0, -12/72, fig.dpi_scale_trans)
for i, summary_filename in enumerate(summary_filenames) :
    n_animals = len(set(T['subject'] for T in np.load(summary_filename + '%s.npy' % keys[0], allow_pickle=True)))
    cell_type = group_labels[i].split('_')[0].replace('-cells', '')
    genotype = group_labels[i].split('_')[1].replace('cond-', '').replace('-', ' ')
    # thin rectangle rather than a line: a flat line has no height and is cropped by bbox_inches = 'tight'
    ax.add_patch(mpatches.Rectangle((i - 0.45, 0), 0.9, 0.012, color = colors[i][0], lw = 0, clip_on = False,
                                    transform = line_transform))
    ax.annotate('%i %s\n%s' % (n_animals, cell_type, genotype), (i, 0), xycoords = legend_transform,
                xytext = (0, -15), textcoords = 'offset points', color = colors[i][0], ha = 'center', va = 'top',
                fontsize = 9, fontweight = 'bold', family = 'Arial', linespacing = 1.0, annotation_clip = False)

groups_name = '_'.join(group_labels)
fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/Responsiveness_bars_Tunings_%s.svg' % groups_name),
            bbox_inches = 'tight', facecolor = 'white')



#%%%%%%%%%%%%%%%%  RESPONSIVENESS BARS - SENSITIVITIES, PER CONTRAST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % of responsive ROIs at each contrast, one bar per group (mean +/- sem over sessions)
# same significance as get_responsiveness_to_visual_stim (positive or negative, in % of nROIs_final),
# or only the positive responses with responses = 'pos'

angle_key = '_angle-90.0' # orientation of the Sensitivities summaries ('_angle-90.0' also exists)
responses = 'pos+neg' # 'pos+neg' (as the pies) or 'pos'
quantity = 'dFoF' # 'dFoF' (Sensitivities_<group>) or 'Deconvolved' (Sensitivities_Deconvolved_<group>)
total_on = 'nROIs_final'

prefix = 'Deconvolved_' if quantity == 'Deconvolved' else ''
summary_filenames = get_summary_names([prefix + f for f in folders], summary_paths, summary_protocols = 'Sensitivities')
colors = reshape_colors_with(colors, second_cond_grey=True)

# contrasts of the protocol: from the sessions that have trials at this angle (some have none)
contrasts = max((S['contrast'] for f in summary_filenames for S in np.load(f + '%s.npy' % angle_key, allow_pickle=True)), key = len)
bar_width = 0.8 / len(summary_filenames)

fig, ax = pt.plt.subplots(figsize = (1.1*len(contrasts), 3))
fig.patch.set_facecolor('white')

perc_values, group_labels = [], [] # per group: % resp. (sessions, contrasts), for the statistics
bar_x, bar_top = {}, {} # per (group, contrast): position and top of the bar, for the stars
for i, summary_filename in enumerate(summary_filenames) :

    summary = np.load(summary_filename + '%s.npy' % angle_key, allow_pickle=True)
    # sessions without this orientation (e.g. GCaMP8m tests: 90 deg gratings only) have no contrast: skipped
    skipped = [os.path.basename(S['datafile']) for S in summary if len(S['contrast']) != len(contrasts)]
    if len(skipped) :
        print('%s%s: %i/%i sessions skipped (no trials at this angle): %s' % (os.path.basename(summary_filename), angle_key,
                                                                         len(skipped), len(summary), ', '.join(skipped)))
    summary = [S for S in summary if len(S['contrast']) == len(contrasts)]
    if len(summary) == 0 :
        raise ValueError('%s%s: no session with trials at this angle, try angle_key = "_angle-90.0"' % (os.path.basename(summary_filename), angle_key))
    # % of responsive ROIs per session and contrast: (sessions, contrasts)
    if responses == 'pos' :
        perc = np.array([100 * np.sum(S['significant_pos'], axis = 0) / S[total_on] for S in summary])
    else :
        perc = np.array([100 * np.sum(S['significant_pos'] + S['significant_neg'], axis = 0) / S[total_on]
                         for S in summary])

    perc_values.append(perc)
    group_labels.append(os.path.basename(summary_filename).replace('Sensitivities_' + prefix, '').rstrip('_'))

    x = np.arange(len(contrasts)) + (i - (len(summary_filenames) - 1) / 2) * bar_width
    for c in range(len(contrasts)) :
        bar_x[i, c], bar_top[i, c] = x[c], perc[:, c].mean() + sem(perc[:, c])
    ax.bar(x, perc.mean(axis = 0), width = bar_width * 0.85, color = colors[i][0],
           label = '%s (n = %i ROIs, N = %i sessions)' % (
                os.path.basename(summary_filename).replace('Sensitivities_' + prefix, '').rstrip('_'),
                np.sum([S[total_on] for S in summary]), len(summary)))
    ax.errorbar(x, perc.mean(axis = 0), yerr = sem(perc, axis = 0),
                fmt = 'none', ecolor = 'black', elinewidth = 3, capsize = 0)

ax.set_xticks(np.arange(len(contrasts)))
ax.set_xticklabels(['%.2f' % c for c in contrasts])
ax.set_xlabel('contrast', fontsize = 11, color = 'black')
ax.set_ylabel('% pos. resp.' if responses == 'pos' else '% resp. (pos.+neg.)', fontsize = 11, color = 'black')
ax.set_title(quantity, fontsize = 11, color = 'black')
ax.set_yticks([0, 30, 60, 90])
ax.set_facecolor('white')
for side in ['top', 'right'] :
    ax.spines[side].set_visible(False)
for side in ['left', 'bottom'] :
    ax.spines[side].set_color('black')
    ax.spines[side].set_linewidth(1)
    ax.spines[side].set_position(('outward', 6))
ax.tick_params(axis = 'both', colors = 'black', labelsize = 10)
ax.legend(loc = 'upper left', bbox_to_anchor = (1.01, 1), frameon = False, fontsize = 8, labelcolor = 'black')

# ---------- statistics (on the % resp. per session) ----------
from scipy.stats import kruskal, mannwhitneyu, friedmanchisquare
import itertools

print("\n" + "="*70)
print("RESPONSIVENESS SENSITIVITIES%s %s - STATISTICS (%% %s resp. per session, on %s)" % (angle_key, quantity, responses, total_on))
print("="*70)
comparisons = [] # significant differences between groups, drawn as stars on the figure

# 1) between groups, at each contrast: Kruskal-Wallis, then Mann-Whitney U (Bonferroni within the contrast)
print("\n--- between groups, at each contrast (Kruskal-Wallis + Mann-Whitney U, Bonferroni) ---")
pairs = list(itertools.combinations(range(len(perc_values)), 2))
for c, contrast in enumerate(contrasts) :
    values = [perc[:, c] for perc in perc_values]
    print("\n  contrast %.2f : " % contrast + ", ".join("%s = %.1f +/- %.1f" % (label, np.mean(v), sem(v))
                                                    for label, v in zip(group_labels, values)))
    if len(values) < 2 :
        print("    only one group: no comparison")
        continue
    H, p = kruskal(*values)
    print("    Kruskal-Wallis H-test: H = %.4f, p = %.4g" % (H, p))
    if p < 0.05 :
        for i1, i2 in pairs :
            _, p_raw = mannwhitneyu(values[i1], values[i2], alternative = "two-sided")
            p_adjusted = min(p_raw * len(pairs), 1.0)
            print("      %s vs %s: p = %.4g [%s]" % (group_labels[i1], group_labels[i2], p_adjusted,
                                                    p_to_stars(p_adjusted)))
            if p_adjusted < 0.05 :
                comparisons.append(((i1, c), (i2, c), p_adjusted))
    else :
        print("    Result is not significant. No post-hoc tests performed.")

# 2) within each group, effect of contrast: Friedman test (same sessions at all contrasts)
print("\n--- within each group, effect of contrast (Friedman test, paired sessions) ---")
for label, perc in zip(group_labels, perc_values) :
    chi2, p = friedmanchisquare(*[perc[:, c] for c in range(len(contrasts))])
    print("  %s: N = %i sessions, chi2 = %.2f, p = %.4g [%s]" % (label, len(perc), chi2, p,
                                                                p_to_stars(p)))
print("="*70)

# stars on top of the significant differences between groups (at each contrast)
y_max = draw_significance_brackets(ax, comparisons, bar_x, bar_top, step = 10, fontsize = 8)
ax.set_ylim([0, max(100, y_max + 8)])
ax.spines['left'].set_bounds(0, 90)

groups_name = '_'.join(os.path.basename(f).replace('Sensitivities_' + prefix, '').rstrip('_') for f in summary_filenames)
fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/Responsiveness_bars_Sensitivities%s_%s_%s_%s.svg' % (angle_key, quantity, responses, groups_name)),
            bbox_inches = 'tight', facecolor = 'white')







# %%
