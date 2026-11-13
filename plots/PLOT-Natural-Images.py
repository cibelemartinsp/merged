# %% [markdown]
# # Natural Images -- Pyramidal cells
#
# reads the summaries built by analysis/NaturalImages-Dataset.py:
#   ~/CURATED/Cibele/summary/NaturalImages_<quantity>_<condition>.npy
#
# each entry = one session, with:
#   Responses        (nROIs, nContrasts, nImages)  post-pre response
#   significant_pos  (nROIs, nContrasts, nImages)  uncorrected p<0.05
#   pval_pos         (nROIs, nContrasts, nImages)
#   Traces           (nROIs, nContrasts, nImages, nTimes)  trial-averaged, binned
#   t, contrast, Image-ID, subject, nROIs_original, nROIs_final

# %%
import sys, os, itertools
# merged/ root (parent of plots/): physion and general_tools are taken from there
try:
    merged_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError: # interactive window
    merged_root = os.path.join(os.path.expanduser('~'), 'code', 'merged')
sys.path.insert(0, os.path.join(merged_root, 'physion', 'src'))
sys.path.insert(0, merged_root)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, mannwhitneyu, kruskal, wilcoxon, pearsonr

from general_tools.cell_type_color_codes import get_color

import physion.utils.plot_tools as pt
pt.set_style()

summary_folder = \
    os.path.expanduser('~/CURATED/Cibele/summary')

# ---------------- SETTINGS ----------------
# signals analysed: every cell shows its figures for each of them, one below the other
#   'Deconvolved', 'dFoF' or 'Zscore_correctedFluo' (summaries of create_summaries/NaturalImages-Dataset.py)
#   response window (post - pre): Deconvolved [0, 1] - [-1, 0] s, dFoF and Zscore [1, 2] - [-1, 0] s
QUANTITIES = ['Deconvolved', 'Zscore_correctedFluo']
QUANTITY = QUANTITIES[0] # signal of the protocol figure
quantity_labels = {'Deconvolved':'Deconvolved', 'dFoF':'dF/F', 'Zscore_correctedFluo':'Z-score (corr. fluo)'}

DATASETS = [
    {"name": "PYR-PV-SynGCaMP_WT_P15-P19_V1"},
    {"name": "PYR-PV-SynGCaMP_WT_P20-P23_V1"},
    {"name": "PYR-PV-SynGCaMP_WT_P24-P27_V1"},
    {"name": "PYR-SynGCaMP_WT_Adult_V1"},
]

# responsive ROI = significant (positive) for >=1 of the 20 image x contrast
# conditions, after Bonferroni correction over those 20 tests
BONFERRONI = True
ALPHA = 0.05
ONLY_RESPONSIVE = True     # restrict amplitude / selectivity analyses to responsive ROIs

SAVE = False
output_dir = os.path.expanduser('~/Desktop/Final-Figures-PhD')


# ---------------- LOAD ----------------
def short(name):
    return name.replace('PYR-PV-SynGCaMP_WT_', '').replace('PYR-SynGCaMP_WT_', '')\
               .replace('_V1', '')

def responsive_mask(S):
    """ (nROIs,) bool: responsive to at least one image x contrast """
    n_tests = S['pval_pos'][0].size
    alpha = ALPHA/n_tests if BONFERRONI else ALPHA
    return np.nanmin(S['pval_pos'].reshape(len(S['pval_pos']), -1), axis=1) < alpha

def load_data(quantity):
    DATA = {}
    for d in DATASETS:
        fn = os.path.join(summary_folder, 'NaturalImages_%s_%s.npy' % (quantity, d['name']))
        if not os.path.isfile(fn):
            print('  [!!] missing:', fn)
            continue
        S = list(np.load(fn, allow_pickle=True))
        # keep only the sessions of the 10 images x 2 contrasts protocol
        # (e.g. PyrF_12 2026_07_20-11-36-12 was recorded with 2 images x 8 contrasts: its arrays cannot be stacked with the others)
        other_protocol = [s for s in S if np.shape(s['Responses'])[1:] != (2, 10)]
        for s in other_protocol:
            print('  [!!] %s: session %s (%s) skipped, other protocol: %i images x %i contrasts' % (
                    short(d['name']), os.path.basename(s['datafile']).replace('.nwb', ''), s['subject'],
                    len(s['Image-ID']), len(s['contrast'])))
        S = [s for s in S if np.shape(s['Responses'])[1:] == (2, 10)]
        DATA[d['name']] = dict(sessions=S, color=get_color(d['name']))
        print('%s: N=%i sessions, %i animals, n=%i ROIs' % (
                d['name'], len(S), len(set(s['subject'] for s in S)),
                sum(len(s['Responses']) for s in S)))
    return DATA

DATA_ALL = {}
for q in QUANTITIES:
    print('\n--- %s ---' % q)
    DATA_ALL[q] = load_data(q)
    if not DATA_ALL[q]:
        print('  [!!] no NaturalImages_%s_ summary -> run: python create_summaries/NaturalImages-Dataset.py %s' % (q, q))
        del DATA_ALL[q]
QUANTITIES = list(DATA_ALL.keys())

# DATA and names: the signal being plotted (each analysis cell loops over QUANTITIES)
DATA = DATA_ALL[QUANTITY]
names = list(DATA.keys())
contrasts = DATA[names[0]]['sessions'][0]['contrast']
images = DATA[names[0]]['sessions'][0]['Image-ID']

def counts(n, rois='all'):
    """ N animals, N sessions, n ROIs of group n
        rois='all'  -> every session and ROI
        rois='used' -> what the amplitude/selectivity panels use: responsive ROIs
                       (if ONLY_RESPONSIVE), sessions with at least one of them """
    S = DATA[n]['sessions']
    if rois == 'used' and ONLY_RESPONSIVE:
        S = [s for s in S if responsive_mask(s).any()]
        nROIs = sum(responsive_mask(s).sum() for s in S)
    else:
        nROIs = sum(len(s['Responses']) for s in S)
    return dict(animals=len(set(s['subject'] for s in S)), sessions=len(S), rois=nROIs)

def n_label(n, rois='all', sep='\n'):
    c = counts(n, rois)
    return sep.join([short(n), 'N=%i animals' % c['animals'],
                     'N=%i sessions' % c['sessions'], 'n=%i ROIs' % c['rois']])

def set_group_xticks(ax, rois='all'):
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n_label(n, rois) for n in names], fontsize=7)

def savefig(fig, label):
    # signal of the figure as title, and in the file name
    fig.suptitle(quantity_labels.get(QUANTITY, QUANTITY), fontsize=12, fontweight='bold', y=1.03)
    if SAVE:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'NaturalImages_%s_%s.svg' % (QUANTITY, label)),
                    bbox_inches='tight')

def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

def add_significance(ax, pairs):
    """ pairs: list of (x1, x2, p) -> bracket + stars above the data, for p<0.05 only """
    pairs = [(x1, x2, p) for x1, x2, p in pairs if p < 0.05]
    if not pairs:
        return
    y0, y1 = ax.get_ylim()
    step = 0.08*(y1 - y0)
    for k, (x1, x2, p) in enumerate(sorted(pairs, key=lambda q: abs(q[1] - q[0]))):
        y = y1 + k*step
        ax.plot([x1, x1, x2, x2], [y - 0.3*step, y, y, y - 0.3*step], color='k', lw=0.8)
        ax.text((x1 + x2)/2, y - 0.25*step, stars(p), ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax.set_ylim(y0, y1 + (len(pairs) + 0.5)*step)

def compare_groups(values, label, ax=None, x_offset=0):
    """ values: {name: array over sessions} -> Kruskal-Wallis + pairwise Mann-Whitney
        if ax is given, significant pairs are marked with stars on that panel
        (x_offset: shift of the brackets, e.g. +0.2 when the group sits right of its tick) """
    groups = [np.asarray(values[n]) for n in names if len(values[n]) > 1]
    print('\n--- %s ---' % label)
    for n in names:
        v = np.asarray(values[n])
        print('   %-10s %.3f ± %.3f  (N=%i)' % (short(n), np.mean(v), sem(v), len(v)))
    if len(groups) > 2:
        print('   Kruskal-Wallis p = %.4f' % kruskal(*groups).pvalue)
    pairs = []
    for (i1, n1), (i2, n2) in itertools.combinations(enumerate(names), 2):
        if len(values[n1]) > 1 and len(values[n2]) > 1:
            p = mannwhitneyu(values[n1], values[n2]).pvalue
            print('   %s vs %s: Mann-Whitney p = %.4f%s' % (short(n1), short(n2), p,
                                                         '  ' + stars(p) if p < 0.05 else ''))
            pairs.append((i1 + x_offset, i2 + x_offset, p))
    if ax is not None:
        add_significance(ax, pairs)


# %%
#######################################################################################
### PROTOCOL SCHEMATIC ###
#######################################################################################
#   parameters and trial order read from one NWB file of the analysed sessions
#   images: Image-ID i = i-th file of the sorted folder (0-based, string sort as natural_image.py:
#   1.png, 10.png, 11.png, 2.png, ...), processed as on the rig: greyscale (skimage rgb2gray)
#   + histogram equalization, then bg + contrast*(image - 0.5)
#   if NI_FOLDER is None or too small, a placeholder texture (1/f noise, NOT the real images) is drawn, labelled as such.

NI_FOLDER = os.path.join(os.path.expanduser('~'), 'DATA', 'NI_bank11')
PROTOCOL_NWB = DATA[names[-1]]['sessions'][0]['datafile'] # first session of the last group
N_TRIALS_SHOWN = 6 # trials drawn on the timeline

from functools import lru_cache
from PIL import Image
from skimage import color, io
from physion.analysis.read_NWB import Data
from physion.visual_stim.natural_images.preprocess import img_after_hist_normalization

nwb = Data(PROTOCOL_NWB, verbose=False)
P = nwb.metadata
stim_table = {k: np.array(nwb.nwbfile.stimulus[k].data[:]).flatten()
              for k in ['Image-ID', 'contrast', 'time_start_realigned', 'time_stop_realigned']}
image_ids = np.unique(stim_table['Image-ID']).astype(int)
stim_contrasts = np.unique(stim_table['contrast'])
bg = float(P['presentation-blank-screen-color'])

@lru_cache(maxsize=None)
def source_image(image_id, size=(160, 120)):
    """ grey-level image in [0,1] (real image if NI_FOLDER has it, else a placeholder) """
    if NI_FOLDER is not None and os.path.isdir(NI_FOLDER):
        files = np.sort(os.listdir(NI_FOLDER))
        if int(image_id) < len(files):
            with np.errstate(invalid='ignore'): # spurious matmul warning of rgb2gray on macOS (no NaN)
                img = img_after_hist_normalization(color.rgb2gray(io.imread(os.path.join(NI_FOLDER, files[int(image_id)])))) # as on the rig
            return np.array(Image.fromarray(img).resize(size, Image.BILINEAR)), True
    rng = np.random.default_rng(int(image_id))
    fx, fy = np.meshgrid(np.fft.fftfreq(size[0]), np.fft.fftfreq(size[1]))
    spectrum = (rng.normal(size=fx.shape) + 1j*rng.normal(size=fx.shape)) / np.maximum(np.hypot(fx, fy), 1./size[0])
    img = np.real(np.fft.ifft2(spectrum))
    return np.clip((img - img.mean()) / (4*img.std()) + 0.5, 0, 1), False

def shown_image(image_id, contrast):
    # as physion's natural_image stimulus: bg + contrast*(image - 0.5)
    img, real = source_image(image_id)
    return bg + contrast*(img - 0.5), real

fig = plt.figure(figsize=(12, 6.))
gs = fig.add_gridspec(3, len(image_ids), height_ratios=[1, 1, 1.6], hspace=0.3, wspace=0.08,
                      left=0.07, right=0.98, top=0.88, bottom=0.2)

# A: the stimulus set, one row per contrast (highest on top)
real_images = True
for row, c in enumerate(stim_contrasts[::-1]):
    for col, image_id in enumerate(image_ids):
        ax = fig.add_subplot(gs[row, col])
        img, real = shown_image(image_id, c)
        real_images &= real
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title('image %i' % image_id if not real else 'image %i\n(%s)' % (image_id,
                         np.sort(os.listdir(NI_FOLDER))[image_id]), fontsize=8)
        if col == 0:
            ax.set_ylabel('contrast\n%.1f' % c, fontsize=9)
fig.text(0.07, 0.95, 'A. Stimulus set: %i natural images x %i contrasts%s' % (
         len(image_ids), len(stim_contrasts), '' if real_images else
         '   [placeholder textures: set NI_FOLDER to show the real images]'),
         fontsize=11, fontweight='bold')

# B: timeline of the first trials (real order of this session)
ax = fig.add_subplot(gs[2, :])
t_on, t_off = stim_table['time_start_realigned'], stim_table['time_stop_realigned']
t_end = t_off[N_TRIALS_SHOWN - 1] + 2.
ax.axvspan(0, t_end, ymin=0.3, ymax=0.7, color=str(bg), lw=0)
for k in range(N_TRIALS_SHOWN):
    img, _ = shown_image(stim_table['Image-ID'][k], stim_table['contrast'][k])
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, extent=[t_on[k], t_off[k], 0.3, 0.7], aspect='auto', zorder=2)
    ax.text((t_on[k] + t_off[k])/2, 0.76, 'img %i\nc=%.1f' % (stim_table['Image-ID'][k], stim_table['contrast'][k]),
            ha='center', va='bottom', fontsize=8)
# analysis windows of the first trial
post_start = 0. if QUANTITY == 'Deconvolved' else 1.
ax.plot([t_on[0] - 1, t_on[0]], [0.2, 0.2], color='0.5', lw=4, solid_capstyle='butt')
ax.plot([t_on[0] + post_start, t_on[0] + post_start + 1], [0.2, 0.2], color='#0a5c8a', lw=4, solid_capstyle='butt')
ax.text(t_on[0] - 0.5, 0.13, 'pre', ha='center', va='top', fontsize=7, color='0.4')
ax.text(t_on[0] + post_start + 0.5, 0.13, 'post', ha='center', va='top', fontsize=7, color='#0a5c8a')
ax.annotate('', (t_on[1], 0.25), xytext=(t_on[0], 0.25), arrowprops=dict(arrowstyle='<->', lw=0.8))
ax.text((t_on[0] + t_on[1])/2, 0.27, '%.0f s' % (t_on[1] - t_on[0]), ha='center', va='bottom', fontsize=7)
ax.text(t_on[0]/2, 0.5, 'grey\n%.0f s' % t_on[0], ha='center', va='center', fontsize=7)
ax.set_xlim(0, t_end); ax.set_ylim(0, 1.05)
ax.set_yticks([])
for side in ['left', 'right', 'top']:
    ax.spines[side].set_visible(False)
ax.set_xlabel('time from protocol start (s)')
ax.set_title('B. Timeline: first %i of %i trials (randomized order, %s)' % (N_TRIALS_SHOWN, len(t_on),
             os.path.basename(PROTOCOL_NWB).replace('.nwb', '')), loc='left', fontsize=11, fontweight='bold')

# C: protocol parameters
fig.text(0.07, 0.07,
         '%s  |  %i images x %i contrasts (%s) x %i repeats = %i trials, %s order (%s shuffling, seed %s)\n'
         'image %s s, grey screen (level %.1f) %s s between images, %s s before / %s s after  |  radius %s deg, screen %s  |  total %.0f min\n'
         'response = mean(post %s s) - mean(pre [-1, 0] s) (%s), t-test over the %i repeats' % (
             P['protocol'], len(image_ids), len(stim_contrasts), ', '.join('%.1f' % c for c in stim_contrasts),
             int(P['N-repeat']), len(t_on), P['Presentation'], P['shuffling'], P['shuffling-seed'],
             P['presentation-duration'], bg, P['presentation-interstim-period'],
             P['presentation-prestim-period'], P['presentation-poststim-period'],
             P['radius'], P['Screen'], (t_off[-1] + float(P['presentation-poststim-period']))/60,
             '[0, 1]' if QUANTITY == 'Deconvolved' else '[1, 2]', QUANTITY, int(P['N-repeat'])),
         fontsize=8.5, va='center')
savefig(fig, 'protocol')


# %%
#######################################################################################
### 0) DATASET SUMMARY: N animals, N sessions, n ROIs ###
#######################################################################################

for QUANTITY in QUANTITIES:
    DATA, names = DATA_ALL[QUANTITY], list(DATA_ALL[QUANTITY].keys())
    print('\n==================== %s ====================' % QUANTITY)
    rows = []
    for n in names:
        c, u = counts(n, 'all'), counts(n, 'used')
        rows.append([short(n), c['animals'], c['sessions'], c['rois'],
                     '%i (%.0f%%)' % (u['rois'], 100*u['rois']/max(c['rois'], 1))])
    columns = ['group', 'N animals', 'N sessions', 'n ROIs', 'n responsive ROIs']

    fig, ax = plt.subplots(figsize=(6.5, 0.4 + 0.3*len(rows)))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (r, col), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight='bold')
        elif col == 0:
            cell.set_facecolor(DATA[names[r - 1]]['color'])
    fig.tight_layout()
    savefig(fig, 'dataset_summary')

    print('\n%-10s %10s %11s %8s %18s' % tuple(columns))
    for r in rows:
        print('%-10s %10i %11i %8i %18s' % tuple(r))


# %%
#######################################################################################
### 1) RESPONSIVENESS ###
#######################################################################################

for QUANTITY in QUANTITIES:
    DATA, names = DATA_ALL[QUANTITY], list(DATA_ALL[QUANTITY].keys())
    print('\n==================== %s ====================' % QUANTITY)
    fig, AX = plt.subplots(1, 2, figsize=(7, 3))

    frac_any, contrast_pairs = {}, []
    for i, n in enumerate(names):
        S, col = DATA[n]['sessions'], DATA[n]['color']

        # % responsive ROIs per session (any image, any contrast)
        frac_any[n] = np.array([100*responsive_mask(s).mean() for s in S])
        AX[0].bar(i, frac_any[n].mean(), yerr=sem(frac_any[n]), color=col, alpha=0.5, capsize=3)
        AX[0].scatter(i + np.random.uniform(-.15, .15, len(S)), frac_any[n], s=10, color=col)

        # % responsive ROIs per contrast (any image)
        frac_c = []
        for ic, c in enumerate(contrasts):
            n_tests = S[0]['pval_pos'][0].size if BONFERRONI else 1
            frac = [100*(np.nanmin(s['pval_pos'][:, ic, :], axis=1) < ALPHA/n_tests).mean()
                    for s in S]
            AX[1].bar(i + 0.4*ic - 0.2, np.mean(frac), yerr=sem(frac), width=0.35,
                      color=col, alpha=0.4 + 0.5*ic, capsize=2)
            frac_c.append(frac)
        if len(S) > 1:
            # within group, contrast 0.5 vs 1.0 (paired sessions)
            contrast_pairs.append((i - 0.2, i + 0.2, wilcoxon(frac_c[0], frac_c[1]).pvalue))

    AX[0].set_ylabel('% responsive ROIs')
    AX[0].set_title('any image, any contrast')
    AX[1].set_title('per contrast (light: %.1f, dark: %.1f)' % tuple(contrasts))
    for ax in AX:
        set_group_xticks(ax, 'all')
    fig.tight_layout()
    compare_groups(frac_any, '% responsive ROIs (per session)', ax=AX[0])
    add_significance(AX[1], contrast_pairs)
    savefig(fig, 'responsiveness')


# %%
#######################################################################################
### 2) EVOKED RESPONSES: TIME COURSES PER CONTRAST ###
#######################################################################################

for QUANTITY in QUANTITIES:
    DATA, names = DATA_ALL[QUANTITY], list(DATA_ALL[QUANTITY].keys())
    print('\n==================== %s ====================' % QUANTITY)
    fig, AX = plt.subplots(1, len(names), figsize=(3*len(names), 2.8), sharey=True)
    AX = np.atleast_1d(AX)
    for ax, n in zip(AX, names):
        S, col = DATA[n]['sessions'], DATA[n]['color']
        t = S[0]['t']
        for ic, c in enumerate(contrasts):
            # one trace per session: mean over (responsive) ROIs and images
            traces = np.array([s['Traces'][responsive_mask(s) if ONLY_RESPONSIVE else slice(None),
                                           ic].mean(axis=(0, 1))
                               for s in S if (responsive_mask(s).any() or not ONLY_RESPONSIVE)])
            m, e = traces.mean(axis=0), sem(traces, axis=0)
            ax.plot(t, m, color=col, ls=['--', '-'][ic], label='contrast %.1f' % c)
            ax.fill_between(t, m - e, m + e, color=col, alpha=0.2, lw=0)
        ax.axvspan(0, 2, facecolor='0.92', edgecolor='none', zorder=-1)
        ax.set_title(n_label(n, 'used'), fontsize=8)
        ax.set_xlabel('time from stim. (s)')
    AX[0].set_ylabel(quantity_labels.get(QUANTITY, QUANTITY))
    AX[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    savefig(fig, 'timecourses')


# %%
#######################################################################################
### 3) RESPONSE AMPLITUDE & CONTRAST MODULATION ###
#######################################################################################
#   amplitude per session = mean over (responsive) ROIs and the 10 images
#   contrast modulation index per ROI: CMI = (R1.0 - R0.5) / (|R1.0| + |R0.5|)

for QUANTITY in QUANTITIES:
    DATA, names = DATA_ALL[QUANTITY], list(DATA_ALL[QUANTITY].keys())
    print('\n==================== %s ====================' % QUANTITY)
    def session_rois(s):
        return responsive_mask(s) if ONLY_RESPONSIVE else np.ones(len(s['Responses']), bool)

    fig, AX = plt.subplots(1, 2, figsize=(7, 3))

    CMI, contrast_pairs = {}, []
    print('\n--- contrast effect within each group (Wilcoxon, paired sessions) ---')
    for i, n in enumerate(names):
        S, col = DATA[n]['sessions'], DATA[n]['color']
        amp = np.array([s['Responses'][session_rois(s)].mean(axis=(0, 2))
                        for s in S if session_rois(s).any()])          # (nSessions, nContrasts)
        for j in range(len(amp)):
            AX[0].plot(i + np.array([-0.2, 0.2]), amp[j], color=col, alpha=0.4, lw=0.8)
        AX[0].errorbar(i + np.array([-0.2, 0.2]), amp.mean(axis=0), yerr=sem(amp, axis=0),
                       color=col, marker='o', lw=2, capsize=3)
        if len(amp) > 1:
            print('   %-10s R%.1f = %.3f, R%.1f = %.3f   p = %.4f' % (
                short(n), contrasts[0], amp[:, 0].mean(), contrasts[1], amp[:, 1].mean(),
                wilcoxon(amp[:, 0], amp[:, 1]).pvalue))
            contrast_pairs.append((i - 0.2, i + 0.2, wilcoxon(amp[:, 0], amp[:, 1]).pvalue))

        cmi_sessions = []
        for s in S:
            R = s['Responses'][session_rois(s)].mean(axis=2)            # (nROIs, nContrasts)
            if len(R):
                cmi = (R[:, 1] - R[:, 0]) / (np.abs(R[:, 1]) + np.abs(R[:, 0]) + 1e-12)
                cmi_sessions.append(np.mean(cmi))
        CMI[n] = np.array(cmi_sessions)
        AX[1].bar(i, CMI[n].mean(), yerr=sem(CMI[n]), color=col, alpha=0.5, capsize=3)
        AX[1].scatter(i + np.random.uniform(-.15, .15, len(CMI[n])), CMI[n], s=10, color=col)

    AX[0].set_ylabel('response (%s)' % quantity_labels.get(QUANTITY, QUANTITY))
    AX[0].set_title('contrast %.1f (left) vs %.1f (right)' % tuple(contrasts))
    AX[1].axhline(0, color='k', lw=0.5)
    AX[1].set_ylabel('contrast modulation index')
    for ax in AX:
        set_group_xticks(ax, 'used')
    fig.tight_layout()
    add_significance(AX[0], contrast_pairs)
    compare_groups(CMI, 'contrast modulation index (per session)', ax=AX[1])
    savefig(fig, 'amplitude_CMI')


# %%
#######################################################################################
### 4) IMAGE SELECTIVITY ###
#######################################################################################
#   lifetime sparseness per ROI (Vinje & Gallant 2000), across the 10 images:
#       S = [1 - (sum r/N)^2 / sum(r^2/N)] / (1 - 1/N),   r clipped at 0
#   0 = same response to all images, 1 = responds to a single image
#
#   tuning stability across contrasts: Pearson r between the 10-image response
#   profiles at contrast 0.5 and 1.0, per ROI

for QUANTITY in QUANTITIES:
    DATA, names = DATA_ALL[QUANTITY], list(DATA_ALL[QUANTITY].keys())
    print('\n==================== %s ====================' % QUANTITY)
    def lifetime_sparseness(R):
        R = np.clip(R, 0, None)
        N = R.shape[-1]
        num = (R.sum(axis=-1)/N)**2
        den = (R**2).sum(axis=-1)/N
        with np.errstate(invalid='ignore', divide='ignore'):
            return (1 - num/den) / (1 - 1./N)

    fig, AX = plt.subplots(1, 3, figsize=(12, 3))

    SPARSE, STABILITY = {}, {}
    for i, n in enumerate(names):
        S, col = DATA[n]['sessions'], DATA[n]['color']

        ls_rois = np.concatenate([lifetime_sparseness(s['Responses'][session_rois(s), -1])
                                  for s in S])
        ls_rois = ls_rois[np.isfinite(ls_rois)]
        AX[0].hist(ls_rois, bins=np.linspace(0, 1, 21), density=True, histtype='step',
                   color=col, lw=1.5, label=n_label(n, 'used', sep=', '))

        SPARSE[n] = np.array([np.nanmean(lifetime_sparseness(s['Responses'][session_rois(s), -1]))
                              for s in S if session_rois(s).any()])
        AX[1].bar(i, SPARSE[n].mean(), yerr=sem(SPARSE[n]), color=col, alpha=0.5, capsize=3)
        AX[1].scatter(i + np.random.uniform(-.15, .15, len(SPARSE[n])), SPARSE[n], s=10, color=col)

        stab = []
        for s in S:
            R = s['Responses'][session_rois(s)]
            r = [pearsonr(x[0], x[1])[0] for x in R if np.std(x[0]) > 0 and np.std(x[1]) > 0]
            if len(r):
                stab.append(np.nanmean(r))
        STABILITY[n] = np.array(stab)
        AX[2].bar(i, STABILITY[n].mean(), yerr=sem(STABILITY[n]), color=col, alpha=0.5, capsize=3)
        AX[2].scatter(i + np.random.uniform(-.15, .15, len(STABILITY[n])), STABILITY[n],
                      s=10, color=col)

    AX[0].set_xlabel('lifetime sparseness (contrast %.1f)' % contrasts[-1])
    AX[0].set_ylabel('density (ROIs)')
    AX[0].legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(0, -0.25))
    AX[1].set_ylabel('lifetime sparseness\n(session mean)')
    AX[2].set_ylabel('image-tuning correlation\ncontrast %.1f vs %.1f' % tuple(contrasts))
    for ax in AX[1:]:
        set_group_xticks(ax, 'used')
    fig.tight_layout()
    compare_groups(SPARSE, 'lifetime sparseness (per session)', ax=AX[1])
    compare_groups(STABILITY, 'image-tuning stability across contrasts (per session)', ax=AX[2])
    savefig(fig, 'selectivity')


# %%
#######################################################################################
### 5) POPULATION RESPONSE MATRIX (ROIs x images), sorted by preferred image ###
#######################################################################################

for QUANTITY in QUANTITIES:
    DATA, names = DATA_ALL[QUANTITY], list(DATA_ALL[QUANTITY].keys())
    print('\n==================== %s ====================' % QUANTITY)
    fig, AX = plt.subplots(1, len(names), figsize=(2.6*len(names), 4))
    AX = np.atleast_1d(AX)
    for ax, n in zip(AX, names):
        R = np.concatenate([s['Responses'][session_rois(s), -1] for s in DATA[n]['sessions']])
        # normalize each ROI to its max, sort by preferred image
        Rn = R / np.clip(np.abs(R).max(axis=1, keepdims=True), 1e-12, None)
        order = np.lexsort((-Rn.max(axis=1), Rn.argmax(axis=1)))
        ax.imshow(Rn[order], aspect='auto', cmap='magma', vmin=0, vmax=1, interpolation='none')
        ax.set_xticks(range(len(images)))
        ax.set_xticklabels(images.astype(int), fontsize=7)
        ax.set_xlabel('image ID')
        ax.set_title(n_label(n, 'used'), fontsize=8)
    AX[0].set_ylabel('ROIs (sorted by preferred image)')
    fig.tight_layout()
    savefig(fig, 'population_matrix')

    # preferred-image distribution (are some images driving most cells?)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for i, n in enumerate(names):
        R = np.concatenate([s['Responses'][session_rois(s), -1] for s in DATA[n]['sessions']])
        frac = 100*np.bincount(R.argmax(axis=1), minlength=len(images))/len(R)
        ax.plot(images, frac, 'o-', color=DATA[n]['color'], label=n_label(n, 'used', sep=', '))
    ax.axhline(100/len(images), color='k', lw=0.5, ls='--')
    ax.set_xlabel('image ID')
    ax.set_ylabel('% ROIs preferring image')
    ax.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(0, -0.2))
    fig.tight_layout()
    savefig(fig, 'preferred_image')

    plt.show()


# %%
#######################################################################################
### SHARED TOOLS FOR CELLS 6-8 ###
#######################################################################################
#   cells 6 (decoding) and 7 (reliability) need single-trial responses:
#       s['Responses_trials']  (nROIs, nContrasts, nImages, nRepeats)
#   -> summaries built with the current analysis/NaturalImages-Dataset.py

def sessions_with_trials(n):
    S = [s for s in DATA[n]['sessions'] if 'Responses_trials' in s]
    if len(S) < len(DATA[n]['sessions']):
        print('  [!!] %s: %i/%i sessions without single trials -> re-run NaturalImages-Dataset.py'
              % (short(n), len(DATA[n]['sessions']) - len(S), len(DATA[n]['sessions'])))
    return S

def paired_contrast_panel(ax, values, label):
    """ values: {name: (nSessions, nContrasts)}
        one line per session (contrast 0.5 left, 1.0 right), group mean ± sem,
        stars: contrast effect within group (Wilcoxon, paired sessions)
               + group differences at the highest contrast (Mann-Whitney) """
    pairs = []
    print('\n--- %s: contrast effect within each group (Wilcoxon, paired sessions) ---' % label)
    for i, n in enumerate(names):
        v, col = values[n], DATA[n]['color']
        if len(v) == 0:
            continue
        for j in range(len(v)):
            ax.plot(i + np.array([-0.2, 0.2]), v[j], color=col, alpha=0.4, lw=0.8)
        ax.errorbar(i + np.array([-0.2, 0.2]), np.nanmean(v, axis=0), yerr=sem(v, axis=0),
                    color=col, marker='o', lw=2, capsize=3)
        if len(v) > 1:
            p = wilcoxon(v[:, 0], v[:, 1]).pvalue
            print('   %-10s %.1f: %.3f, %.1f: %.3f   p = %.2g%s' % (
                short(n), contrasts[0], v[:, 0].mean(), contrasts[1], v[:, 1].mean(), p,
                '  ' + stars(p) if p < 0.05 else ''))
            pairs.append((i - 0.2, i + 0.2, p))
    add_significance(ax, pairs)
    compare_groups({n: values[n][:, -1] if len(values[n]) else [] for n in names},
                   '%s at contrast %.1f (per session)' % (label, contrasts[-1]),
                   ax=ax, x_offset=0.2)


# %%
#######################################################################################
### 6) DECODING IMAGE IDENTITY FROM THE POPULATION RESPONSE ###
#######################################################################################
#   which of the 10 images was shown, from the single-trial population vector ?
#   (Y. Zerlaut, "Mouse Visual Processing 2.0", III.3)
#
#   per session and contrast, leave-one-repeat-out:
#       train = mean response to each image over the 9 other repeats ("template")
#       test  = held-out trial -> predicted image = template with the highest
#               Pearson correlation across ROIs (pattern-based, ignores overall gain)
#   chance = 1/10 = 10%
#
#   decoding accuracy grows with the number of ROIs, and sessions have different
#   numbers of ROIs -> each session is decoded with the same number of ROIs
#   (N_DECODE_ROIS, random subsets, averaged over N_SUBSAMPLES draws);
#   sessions with fewer ROIs are left out. All ROIs are used (not only responsive),
#   as non-responsive cells can still carry information.

for QUANTITY in QUANTITIES:
    DATA, names = DATA_ALL[QUANTITY], list(DATA_ALL[QUANTITY].keys())
    print('\n==================== %s ====================' % QUANTITY)
    N_DECODE_ROIS = 20
    N_SUBSAMPLES = 20
    ROI_NUMBERS = [2, 5, 10, 20, 40, 80]     # accuracy vs population size (panel 2)
    DECODE_SEED = 0

    def zscore_columns(X):
        with np.errstate(invalid='ignore', divide='ignore'):
            return (X - X.mean(axis=0)) / X.std(axis=0)

    def decode_session(T, n_rois, rng, n_sub=N_SUBSAMPLES):
        """ T: (nROIs, nImages, nRepeats) single-trial responses
            -> confusion matrix (true image x predicted image, rows sum to 1),
               None if the session has fewer than n_rois ROIs """
        nI, nR = T.shape[1], T.shape[2]
        if len(T) < n_rois:
            return None
        conf = np.zeros((nI, nI))
        for _ in range(n_sub):
            X = T[rng.choice(len(T), n_rois, replace=False)]
            for k in range(nR):
                test = X[:, :, k]                                        # (n_rois, nImages)
                if np.isnan(test).all():
                    continue
                train = np.nanmean(np.delete(X, k, axis=2), axis=2)     # (n_rois, nImages)
                C = zscore_columns(test).T @ zscore_columns(train) / n_rois
                C = np.where(np.isfinite(C), C, -np.inf)
                for true in np.flatnonzero(np.isfinite(test).all(axis=0)):
                    if np.isfinite(C[true]).any():
                        conf[true, C[true].argmax()] += 1
                    else:   # silent trial: random guess
                        conf[true, rng.integers(nI)] += 1
        return conf / np.clip(conf.sum(axis=1, keepdims=True), 1, None)

    rng = np.random.default_rng(DECODE_SEED)
    ACC, CONF, ACC_vs_N = {}, {}, {}
    for n in names:
        acc, conf = [], []
        ACC_vs_N[n] = {N: [] for N in ROI_NUMBERS}
        for s in sessions_with_trials(n):
            res = [decode_session(s['Responses_trials'][:, ic], N_DECODE_ROIS, rng)
                   for ic in range(len(contrasts))]
            if res[0] is not None:
                acc.append([100*np.mean(np.diag(c)) for c in res])
                conf.append(res[-1])
            for N in ROI_NUMBERS:
                c = decode_session(s['Responses_trials'][:, -1], N, rng, n_sub=5)
                if c is not None:
                    ACC_vs_N[n][N].append(100*np.mean(np.diag(c)))
        ACC[n] = np.array(acc).reshape(-1, len(contrasts))
        CONF[n] = np.mean(conf, axis=0) if conf else None
        print('%s: %i sessions with >= %i ROIs' % (short(n), len(acc), N_DECODE_ROIS))

    fig, AX = plt.subplots(1, 2, figsize=(9, 3.2))
    paired_contrast_panel(AX[0], ACC, 'decoding accuracy (%)')
    AX[0].axhline(100/len(images), color='k', lw=0.5, ls='--')
    AX[0].set_ylabel('decoding accuracy (%%)\n%i ROIs per session' % N_DECODE_ROIS)
    AX[0].set_title('contrast %.1f (left) vs %.1f (right)' % tuple(contrasts), fontsize=8)
    AX[0].set_xticks(range(len(names)))
    AX[0].set_xticklabels(['%s\nN=%i sessions' % (short(n), len(ACC[n])) for n in names], fontsize=7)

    for n in names:
        Ns = [N for N in ROI_NUMBERS if len(ACC_vs_N[n][N]) > 1]
        m = [np.mean(ACC_vs_N[n][N]) for N in Ns]
        e = [sem(ACC_vs_N[n][N]) for N in Ns]
        AX[1].errorbar(Ns, m, yerr=e, color=DATA[n]['color'], marker='o', capsize=2,
                       label=short(n))
    AX[1].axhline(100/len(images), color='k', lw=0.5, ls='--')
    AX[1].set_xscale('log')
    AX[1].set_xticks(ROI_NUMBERS)
    AX[1].set_xticklabels(ROI_NUMBERS)
    AX[1].minorticks_off()
    AX[1].set_xlabel('number of ROIs')
    AX[1].set_ylabel('decoding accuracy (%)')
    AX[1].set_title('contrast %.1f, mean ± sem over sessions' % contrasts[-1], fontsize=8)
    AX[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    savefig(fig, 'decoding')

    # confusion matrices: which images get mixed up
    fig, AX = plt.subplots(1, len(names), figsize=(2.6*len(names), 2.6))
    AX = np.atleast_1d(AX)
    vmax = 100*max(CONF[n].max() for n in names if CONF[n] is not None)
    for ax, n in zip(AX, names):
        if CONF[n] is None:
            ax.axis('off')
            continue
        im = ax.imshow(100*CONF[n], cmap='viridis', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(images)))
        ax.set_xticklabels(images.astype(int), fontsize=6)
        ax.set_yticks(range(len(images)))
        ax.set_yticklabels(images.astype(int), fontsize=6)
        ax.set_xlabel('predicted image')
        ax.set_title('%s (contrast %.1f)' % (short(n), contrasts[-1]), fontsize=8)
    AX[0].set_ylabel('shown image')
    fig.colorbar(im, ax=AX, label='% trials', shrink=0.8)
    savefig(fig, 'decoding_confusion')

    plt.show()


# %%
#######################################################################################
### 7) TRIAL-TO-TRIAL RELIABILITY ###
#######################################################################################
#   do the responses to the 10 images repeat from trial to trial ?
#   (natural scenes evoke "highly reliable" responses, e.g. Haider 2010, Baudot 2013)
#
#   split-half reliability per ROI and contrast (Marks 2021; Sweeney 2025, as in
#   physion's run_reliability_test, here on the 10-image response profile):
#       repeats split at random in 2 halves -> mean response to each image in each half
#       -> Pearson r between the two 10-image profiles, averaged over N_SPLITS splits
#   null: the same, with the image labels shuffled in one half
#   reliable ROI: r above the 95th percentile of its null distribution

for QUANTITY in QUANTITIES:
    DATA, names = DATA_ALL[QUANTITY], list(DATA_ALL[QUANTITY].keys())
    print('\n==================== %s ====================' % QUANTITY)
    N_SPLITS = 100
    RELIABILITY_SEED = 0

    def rowwise_corr(A, B):
        A = A - A.mean(axis=1, keepdims=True)
        B = B - B.mean(axis=1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            return (A*B).sum(axis=1) / np.sqrt((A**2).sum(axis=1) * (B**2).sum(axis=1))

    def split_half_reliability(T, rng, n_splits=N_SPLITS):
        """ T: (nROIs, nImages, nRepeats) -> r (nROIs,), reliable (nROIs,) bool """
        nR = T.shape[-1]
        r, r_null = [], []
        for _ in range(n_splits):
            p = rng.permutation(nR)
            A = np.nanmean(T[..., p[:nR//2]], axis=-1)
            B = np.nanmean(T[..., p[nR//2:]], axis=-1)
            r.append(rowwise_corr(A, B))
            r_null.append(rowwise_corr(A, B[:, rng.permutation(B.shape[1])]))
        r = np.nanmean(r, axis=0)
        return r, r > np.nanpercentile(r_null, 95, axis=0)

    rng = np.random.default_rng(RELIABILITY_SEED)
    REL, FRAC_REL = {}, {}
    for n in names:
        rel, frac = [], []
        for s in sessions_with_trials(n):
            out = [split_half_reliability(s['Responses_trials'][:, ic], rng)
                   for ic in range(len(contrasts))]
            if session_rois(s).any():
                rel.append([np.nanmean(r[session_rois(s)]) for r, _ in out])
            frac.append([100*ok.mean() for _, ok in out])
        REL[n] = np.array(rel).reshape(-1, len(contrasts))
        FRAC_REL[n] = np.array(frac).reshape(-1, len(contrasts))

    fig, AX = plt.subplots(1, 2, figsize=(9, 3.2))
    paired_contrast_panel(AX[0], REL, 'split-half reliability')
    AX[0].axhline(0, color='k', lw=0.5)
    AX[0].set_ylabel('split-half reliability (r)\n%s ROIs, session mean' %
                     ('responsive' if ONLY_RESPONSIVE else 'all'))
    set_group_xticks(AX[0], 'used')
    paired_contrast_panel(AX[1], FRAC_REL, '% reliable ROIs')
    AX[1].set_ylabel('% reliable ROIs\n(all ROIs, r > shuffled 95th pct.)')
    set_group_xticks(AX[1], 'all')
    for ax in AX:
        ax.set_title('contrast %.1f (left) vs %.1f (right)' % tuple(contrasts), fontsize=8)
    fig.tight_layout()
    savefig(fig, 'reliability')

    plt.show()


# %%
#######################################################################################
### 8) POPULATION SPARSENESS ###
#######################################################################################
#   complementary to the lifetime sparseness of cell 4:
#     lifetime sparseness   = one ROI, across the 10 images   ("how selective is a cell")
#     population sparseness = one image, across all ROIs      ("how few cells respond")
#   same formula (Vinje & Gallant 2000; Froudarakis 2014), computed for each image over
#   all ROIs of the session, then averaged over the 10 images
#   + % of ROIs significantly activated by each image (same Bonferroni test as cell 1)
#   (trial-averaged responses: works with any summary)

for QUANTITY in QUANTITIES:
    DATA, names = DATA_ALL[QUANTITY], list(DATA_ALL[QUANTITY].keys())
    print('\n==================== %s ====================' % QUANTITY)
    POP_SPARSE, FRAC_ACTIVE = {}, {}
    for n in names:
        ps, fa = [], []
        for s in DATA[n]['sessions']:
            ps.append([np.nanmean(lifetime_sparseness(s['Responses'][:, ic].T))
                       for ic in range(len(contrasts))])
            n_tests = s['pval_pos'][0].size if BONFERRONI else 1
            fa.append([100*(s['pval_pos'][:, ic] < ALPHA/n_tests).mean()
                       for ic in range(len(contrasts))])
        POP_SPARSE[n], FRAC_ACTIVE[n] = np.array(ps), np.array(fa)

    fig, AX = plt.subplots(1, 2, figsize=(9, 3.2))
    paired_contrast_panel(AX[0], POP_SPARSE, 'population sparseness')
    AX[0].set_ylabel('population sparseness\n(all ROIs, mean over images)')
    paired_contrast_panel(AX[1], FRAC_ACTIVE, '% ROIs activated per image')
    AX[1].set_ylabel('% ROIs activated by an image\n(mean over images)')
    for ax in AX:
        set_group_xticks(ax, 'all')
        ax.set_title('contrast %.1f (left) vs %.1f (right)' % tuple(contrasts), fontsize=8)
    fig.tight_layout()
    savefig(fig, 'population_sparseness')

    plt.show()


# %%
#######################################################################################
### 9) COMPARISON OF THE THREE SIGNALS: DECONVOLVED, dF/F, Z-SCORE ###
#######################################################################################
#   same measures, computed on each summary (NaturalImages_<quantity>_<group>.npy), per session:
#     % responsive ROIs (cell 1), contrast modulation index (cell 3), lifetime sparseness at
#     the highest contrast (cell 4), split-half reliability at the highest contrast (cell 7)
#   the response window differs: Deconvolved post [0, 1] s, dFoF and Zscore post [1, 2] s
#   (pre [-1, 0] s for all), as set in NaturalImages-Dataset.py
#   statistics: groups compared within each signal (Kruskal-Wallis + Mann-Whitney), as above

COMPARED_QUANTITIES = ['Deconvolved', 'dFoF', 'Zscore_correctedFluo']
QUANTITY = 'all signals' # title of the figure (savefig)

def load_quantity(quantity):
    """ {group name: sessions of the 10 images x 2 contrasts protocol}, groups of DATA only """
    out = {}
    for n in names:
        fn = os.path.join(summary_folder, 'NaturalImages_%s_%s.npy' % (quantity, n))
        if os.path.isfile(fn):
            out[n] = [s for s in np.load(fn, allow_pickle=True) if np.shape(s['Responses'])[1:] == (2, 10)]
    return out

def session_measures(s, rng):
    rois = session_rois(s)
    R = s['Responses'][rois]                                     # (nROIs, nContrasts, nImages)
    m = dict(responsive=100*responsive_mask(s).mean())
    if rois.any():
        Rc = R.mean(axis=2)
        m['CMI'] = np.mean((Rc[:, 1] - Rc[:, 0]) / (np.abs(Rc[:, 1]) + np.abs(Rc[:, 0]) + 1e-12))
        m['sparseness'] = np.nanmean(lifetime_sparseness(R[:, -1]))
        if 'Responses_trials' in s:
            r, _ = split_half_reliability(s['Responses_trials'][rois, -1], rng)
            m['reliability'] = np.nanmean(r)
    return m

measures = [('responsive', '% responsive ROIs'),
            ('CMI', 'contrast modulation index'),
            ('sparseness', 'lifetime sparseness\n(contrast %.1f)' % contrasts[-1]),
            ('reliability', 'split-half reliability\n(contrast %.1f)' % contrasts[-1])]

rng = np.random.default_rng(RELIABILITY_SEED)
COMPARISON = {}
for q in COMPARED_QUANTITIES:
    sessions = load_quantity(q)
    if not sessions:
        print('  [!!] no NaturalImages_%s_ summaries -> run create_summaries/NaturalImages-Dataset.py %s' % (q, q))
        continue
    COMPARISON[q] = {n: [session_measures(s, rng) for s in S] for n, S in sessions.items()}

fig, AX = plt.subplots(1, len(measures), figsize=(4*len(measures), 3.4))
width = 0.8/len(COMPARISON)
for k, (key, label) in enumerate(measures):
    ax = AX[k]
    for j, q in enumerate(COMPARISON):
        values = {}
        for i, n in enumerate(names):
            v = np.array([m[key] for m in COMPARISON[q].get(n, []) if key in m])
            values[n] = v
            if len(v) == 0:
                continue
            x = i + (j - (len(COMPARISON) - 1)/2)*width
            ax.bar(x, v.mean(), yerr=sem(v) if len(v) > 1 else 0, width=0.9*width, color=DATA[n]['color'],
                   alpha=[0.9, 0.55, 0.3][j % 3], hatch=['', '//', '..'][j % 3], edgecolor='k', lw=0.4, capsize=2)
            ax.scatter(x + np.random.uniform(-0.25, 0.25, len(v))*width, v, s=5, color='k', zorder=3)
        compare_groups(values, '%s - %s (per session)' % (label.replace('\n', ' '), quantity_labels[q]))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([short(n) for n in names], fontsize=8)
    ax.set_ylabel(label)
    if key == 'CMI':
        ax.axhline(0, color='k', lw=0.5)
handles = [plt.Rectangle((0, 0), 1, 1, facecolor='0.5', alpha=[0.9, 0.55, 0.3][j % 3], hatch=['', '//', '..'][j % 3],
                         edgecolor='k', lw=0.4) for j in range(len(COMPARISON))]
AX[-1].legend(handles, [quantity_labels[q] for q in COMPARISON], frameon=False, fontsize=8,
              loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
savefig(fig, 'quantity_comparison')

plt.show()

# %%
