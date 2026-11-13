#%%
###################################################################################
###### LEADERBOARD + INTERACTIVE ROI SELECTOR -> DECONVOLVED TRACES FIGURE ########
###### FLEXIBLE VERSION: Works with Adult and Young (by age groups) data        ########
###################################################################################

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

# merged/ root (parent of plots/): physion and general_tools are taken from there
try:
    merged_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError: # interactive window
    merged_root = os.path.join(os.path.expanduser('~'), 'code', 'merged')
sys.path.insert(0, os.path.join(merged_root, 'physion', 'src'))
sys.path.insert(0, merged_root)
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.dataviz.raw import plot as plot_raw
from physion.imaging.dcnv import oasis
from physion.utils import plot_tools as pt
from general_tools.cell_type_color_codes import get_color

import physion
print('physion imported from:', physion.__file__)

# keep text editable in the SVG (Illustrator / Inkscape)
plt.rcParams['svg.fonttype'] = 'none'

# NOTE: the add_VisualStim hack is GONE. physion draws the grating icons by itself
# as long as Data(..., with_visual_stim=True) and init_stim(data) are used.

# ------------------------------------------------------------
# 1) CONFIG - FLEXIBLE DATASET SELECTION
# ------------------------------------------------------------

# ============================================================
# DATASET CONFIGURATION
# ============================================================

# Choose which dataset to analyze:
# - Set to one of the folder names from Dataset_Organization.py
DATASET_NAME = "SST-cells_cond-GluN1-KO_Adult_V1"  # Adult example
#DATASET_NAME = "PV-cells_WT_Adult_V1"          # Adult example
#DATASET_NAME = "PV-cells_cond-GluN1-KO_Adult_V1"  # Adult example
#DATASET_NAME = "SST-cells_WT_Young_V1"          # Young example (will split by age)
#DATASET_NAME = "PYR-SynGCaMP_WT_Adult_V1"      # Adult pyramidal
#DATASET_NAME = "PYR-PV-SynGCaMP_WT_Young_V1"   # Young pyramidal

# Age intervals for young animals (from Dataset_Organization.py)
AGE_INTERVALS = [(15, 19), (20, 23), (24, 27)]

# Automatically determine if this is a young dataset
IS_YOUNG = 'Young' in DATASET_NAME

# Base path (from Dataset_Organization.py)
BASE_PATH = os.path.expanduser('~/CURATED/Cibele/')
NWB_FOLDER = os.path.join(BASE_PATH, DATASET_NAME, 'NWBs')

print(f"\nDataset: {DATASET_NAME}")
print(f"NWB Folder: {NWB_FOLDER}")
print(f"Type: {'YOUNG (by age groups)' if IS_YOUNG else 'ADULT'}")

# ============================================================
# VISUALIZATION CONFIGURATION
# ============================================================

TARGET_PROTOCOL = "ff-gratings-8orientation-2contrasts-15repeats"
#TARGET_PROTOCOL = "ff-gratings-2orientations-8contrasts-15repeats"
# TARGET_PROTOCOL = ""  # <- empty string = no protocol filtering

COLOR = 'darkgray'       # deconvolved fill color
LINE_COLOR = get_color(DATASET_NAME, default='tab:orange') # dF/F line color: group color (general_tools/cell_type_color_codes.py), orange for Young
ICONS = 'own'             # 'own' = draw grating icons here, 'physion' = try physion's renderer
N_STIM = 16               # number of stimuli shown
PAD = 2                   # seconds before first / after last stimulus
N_ROIS = 5                # ROIs shown by default (image uses 5 per block)
TAU = 1.5                 # deconvolution decay (s)
SMOOTHING = 1.5           # display smoothing (samples)
SCALE = 1                 # dF/F amplitude of the little black scale bars
SCORE_MODE = 'evoked'     # 'evoked' (deconvolved stim-modulation) or 'snr' (dF/F SNR)
ALPHA = 1.0               # fill alpha/transparency (1.0 = opaque/exact color, 0.5 = semi-transparent)

dFoF_parameters = dict(
    roi_to_neuropil_fluo_inclusion_factor=1.15,
    neuropil_correction_factor=0.7,
    computed_neuropil_fact_method=None, # fixed factor above (merged physion default: 'slope')
    method_for_F0='sliding_percentile',
    percentile=5.,        # percent
    sliding_window=5*60,  # seconds
)

# ------------------------------------------------------------
# 2) HELPERS
# ------------------------------------------------------------

def protocol_strings(data):
    """All strings in which the protocol name could live."""
    out = []
    protos = getattr(data, 'protocols', None)
    if protos is not None:
        if isinstance(protos, (list, tuple, np.ndarray)):
            out += [str(p) for p in protos]
        else:
            out.append(str(protos))
    out.append(str(getattr(data, 'metadata', '')))
    return out


def matches_protocol(data):
    if not TARGET_PROTOCOL:
        return True
    return any(TARGET_PROTOCOL.lower() in s.lower() for s in protocol_strings(data))


def prepare(fpath):
    data = Data(fpath, with_visual_stim=True, verbose=False)
    init_stim(data)
    data.build_dFoF(verbose=False, **dFoF_parameters)
    return data


def init_stim(data):
    """Works with physion versions that have / don't have init_visual_stim."""
    # In new physion (v2+), init_visual_stim was renamed to build_visual_stim
    if hasattr(data, 'build_visual_stim'):
        data.build_visual_stim(verbose=False)
    elif hasattr(data, 'init_visual_stim'):
        data.init_visual_stim()


def get_stim(data):
    """dict-like with time_start, time_stop, contrast, time_duration (arrays)."""
    vs = getattr(data, 'visual_stim', None)
    exp = getattr(vs, 'experiment', None)
    if exp is not None and 'time_start' in exp:
        return exp
    # fallback: read the stimulus table straight from the NWB file
    stim = data.nwbfile.stimulus
    def grab(*keys):
        for k in keys:
            if k in stim:
                return np.array(stim[k].data[:])
        return None
    ts = grab('time_start_realigned', 'time_start')
    te = grab('time_stop_realigned', 'time_stop')
    c = grab('contrast')
    if ts is None:
        raise RuntimeError('no stimulus times found; NWB stimulus keys: %s' % list(stim.keys()))
    if te is None:
        te = ts + 2.
    out = dict(time_start=ts, time_stop=te,
               contrast=c if c is not None else np.ones_like(ts),
               time_duration=te-ts)
    ang = grab('angle', 'orientation')
    if ang is not None:
        out['angle'] = ang
    return out


def get_dff(data):
    """dF/F as (nROIs, time)."""
    dff = np.asarray(data.dFoF)
    nR = getattr(data, 'nROIs', None)
    if nR is not None and dff.shape[0] != nR and dff.shape[1] == nR:
        dff = dff.T
    return dff


def get_dcnv(data, tau=TAU):
    dff = get_dff(data)
    return oasis(dff, dff.shape[0], tau, 1./data.CaImaging_dt)


def stim_window(data, n_stim=N_STIM, pad=PAD):
    exp = get_stim(data)
    ts = np.array(exp['time_start'])
    dur = exp['time_duration'][0]
    n = min(n_stim, len(ts))
    return [float(ts[0]-pad), float(ts[n-1]+dur+pad)]


def build_contrast_trace(data):
    contrast = 0*data.t_dFoF
    exp = get_stim(data)
    for tstart, tstop, c in zip(exp['time_start'], exp['time_stop'], exp['contrast']):
        contrast[(data.t_dFoF > tstart) & (data.t_dFoF < tstop)] = c
    return contrast


def rank_rois(data, tlim):
    t = data.t_dFoF
    win = (t > tlim[0]) & (t < tlim[1])
    dff = get_dff(data)[:, win]

    if SCORE_MODE == 'snr':
        smooth = gaussian_filter1d(dff, 2, axis=1)
        score = smooth.std(axis=1)/((dff-smooth).std(axis=1)+0.05)
    else:  # 'evoked': deconvolved activity during stimuli vs. outside
        dcnv = get_dcnv(data)[:, win]
        exp = get_stim(data)
        in_stim = np.zeros(t.shape, dtype=bool)
        for ts, te in zip(exp['time_start'], exp['time_stop']):
            in_stim |= (t > ts) & (t < te)
        in_stim = in_stim[win]
        score = (dcnv[:, in_stim].mean(axis=1) - dcnv[:, ~in_stim].mean(axis=1)) \
                / (dcnv.std(axis=1) + 1e-6)

    score = np.nan_to_num(score, nan=0.0)
    return np.argsort(score)[::-1], score


# ------------------------------------------------------------
# 3) THE FIGURE (deconvolved fill + dF/F line, stim bands, contrast bar, icons)
# ------------------------------------------------------------

def draw_stim_icons(data, tlim, ax_s, fig):
    """Grating icons above the traces (stripes oriented by 'angle', shaded by contrast)."""
    ax_s.set_xlim(tlim); ax_s.set_ylim(0, 1); ax_s.axis('off')
    if ICONS == 'physion':
        try:
            plot_raw(data, tlim=tlim,
                     settings={'VisualStim': {'fig_fraction': 0.5, 'color': 'black',
                                              'with_screen_inset': True}},
                     ax=ax_s)
            return
        except Exception as e:
            print('[icons] physion renderer failed (%r) -> drawing icons myself' % (e,))
            ax_s.cla(); ax_s.set_xlim(tlim); ax_s.set_ylim(0, 1); ax_s.axis('off')

    exp = get_stim(data)
    ts, dur = np.array(exp['time_start']), np.array(exp['time_duration'])
    ctr = np.array(exp['contrast']) if 'contrast' in exp else np.ones_like(ts)
    ang = None
    for k in ('angle', 'orientation'):
        if k in exp:
            ang = np.array(exp[k]); break

    pos = ax_s.get_position()
    W_in = fig.get_size_inches()[0]*pos.width
    H_in = fig.get_size_inches()[1]*pos.height
    dt = tlim[1]-tlim[0]
    for i, (t0, d) in enumerate(zip(ts, dur)):
        if not (tlim[0] <= t0 <= tlim[1]):
            continue
        s_in = min(0.92*H_in, 0.9*d/dt*W_in)          # square icon, in inches
        w, h = s_in/W_in*dt, s_in/H_in                # ... converted to data units
        xc, yc = t0+d/2., 0.5
        rect = Rectangle((xc-w/2, yc-h/2), w, h, transform=ax_s.transData,
                         fc='white', ec='k', lw=0.3)
        ax_s.add_patch(rect)
        if ang is None:
            continue
        th = np.deg2rad(float(ang[i])+90.)            # stripes perpendicular to drift direction
        dx, dy = np.cos(th), np.sin(th)
        gray = str(float(np.clip(1-ctr[i], 0, 1)))
        for o in np.linspace(-1, 1, 6):
            ux, uy = -dy*o, dx*o
            line, = ax_s.plot([xc+(ux-2*dx)*w/2, xc+(ux+2*dx)*w/2],
                              [yc+(uy-2*dy)*h/2, yc+(uy+2*dy)*h/2],
                              color=gray, lw=0.8, solid_capstyle='butt')
            line.set_clip_path(rect)
    ax_s.set_xlim(tlim); ax_s.set_ylim(0, 1)


def plot_traces(groups, tlim, figsize=(7, 4), savepath=None):
    """
    groups: list of dict(data=Data, rois=[...], color='tab:orange', scale=1, dcnv=None)
    Several groups are stacked (like PV / SST in panel A).
    """
    fig, (ax_s, ax) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 8], hspace=0.02))
    first, row = None, 0

    for g in groups:
        data = g['data']
        first = first or data
        dcnv_all = g.get('dcnv')
        if dcnv_all is None:
            dcnv_all = get_dcnv(data)
        dff_all = get_dff(data)
        cond = (data.t_dFoF > tlim[0]) & (data.t_dFoF < tlim[1])
        t = data.t_dFoF[cond]
        row0 = row

        for roi in list(g['rois'])[::-1]:   # first ROI listed = top row
            # deconvolved: filled, colored
            d = gaussian_filter1d(dcnv_all[roi, cond], SMOOTHING)
            d = d/(np.max(d)+1e-12)
            ax.fill_between(t, row+0*d, row+d, lw=0.1, alpha=g.get('alpha', ALPHA), color=g['color'])
            # dF/F: green line
            f = gaussian_filter1d(dff_all[roi, cond], SMOOTHING)
            amp = (f.max()-f.min())+1e-12
            f = (f-f.min())/amp
            ax.plot(t, row+f, color=g.get('line_color', LINE_COLOR), lw=0.5)
            # per-ROI scale bar
            ax.plot([tlim[0]-1]*2, [row, row+g.get('scale', SCALE)/amp], 'k-', lw=0.5)
            row += 1

        ax.text(tlim[0]-3, (row0+row)/2., '%s\u0394F/F' % g.get('scale', SCALE),
                rotation=90, ha='center', va='center', fontsize=6)
        row += 1  # gap between groups

    # contrast bar + grey stimulus bands (from the first session)
    contrast = build_contrast_trace(first)
    cond = (first.t_dFoF > tlim[0]) & (first.t_dFoF < tlim[1])
    ax.fill_between(first.t_dFoF[cond], -2+0*contrast[cond], -2+contrast[cond],
                    color='tab:grey', lw=0.1)
    exp = get_stim(first)
    dur = exp['time_duration'][0]
    for ts in np.array(exp['time_start']):
        if tlim[0] < ts < tlim[1]:
            ax.fill_between([ts, ts+dur], [-2, -2], [row, row], color='k', alpha=0.1, lw=0)

    pt.draw_bar_scales(ax, Xbar=5, Xbar_label='5s', Ybar=1e-12, fontsize=6, lw=0.2)
    pt.set_plot(ax, [])

    # grating icons strip on top (physion's own VisualStim renderer)
    draw_stim_icons(first, tlim, ax_s, fig)
    ax.set_xlim(tlim)

    if savepath:
        fig.savefig(savepath, format=os.path.splitext(savepath)[1][1:] or 'svg',
                    dpi=300, bbox_inches='tight', facecolor='white')
    return fig


# ------------------------------------------------------------
# 4) LEADERBOARD FOR SINGLE FOLDER WITH AGE FILTERING
# ------------------------------------------------------------

def build_leaderboard_from_scan(folder_path, age_interval=None, group_name=""):
    """
    Build leaderboard by scanning folder and optionally filtering by age.
    
    Args:
        folder_path: Path to NWB folder
        age_interval: Tuple of (min_age, max_age) or None for all ages
        group_name: Label for this group (e.g., 'P15-19')
    """
    # Get all NWB files in the folder
    nwb_files = sorted(glob.glob(os.path.join(folder_path, "*.nwb")))
    
    if not nwb_files:
        print(f"  [WARNING] No NWB files found in {folder_path}")
        return [], 0
    
    print(f"  Found {len(nwb_files)} NWB files in {folder_path}")
    
    # If age filtering is requested, we need to load metadata to get ages
    if age_interval is not None:
        min_age, max_age = age_interval
        print(f"  Filtering by age range: {age_interval}")
        filtered_files = []
        for fpath in nwb_files:
            try:
                data = Data(fpath, metadata_only=True, verbose=False)
                age = getattr(data, 'age', -1)
                if min_age <= age <= max_age and age >= 0:
                    filtered_files.append(fpath)
                    print(f"    {os.path.basename(fpath)}: age={age} ✓")
                else:
                    print(f"    {os.path.basename(fpath)}: age={age} ✗")
            except Exception as e:
                print(f"    {os.path.basename(fpath)}: error reading age - {e}")
        nwb_files = filtered_files
        print(f"  Filtered to {len(nwb_files)} files in age range {age_interval}")
    
    print(f"  Screening for protocol: '{TARGET_PROTOCOL}' (score = {SCORE_MODE})")

    leaderboard, skipped_protocol, diag = [], 0, None

    for fpath in nwb_files:
        fname = os.path.basename(fpath)
        try:
            # Load only metadata first to check protocol
            data = Data(fpath, metadata_only=True, verbose=False)
            if diag is None:
                diag = (fname, protocol_strings(data))
            
            # Check if protocol matches
            if not matches_protocol(data):
                skipped_protocol += 1
                continue
            
            # Now load full data for ranking
            data = Data(fpath, with_visual_stim=True, verbose=False)
            init_stim(data)
            data.build_dFoF(verbose=False, **dFoF_parameters)
            tlim = stim_window(data)
            ranked, scores = rank_rois(data, tlim)
            
            age = getattr(data, 'age', -1)
            leaderboard.append(dict(
                file_path=fpath, 
                file_name=fname,
                top_score=float(scores[ranked[0]]),
                sorted_rois=ranked, 
                tlim_bounds=tlim,
                group=group_name,
                age=age
            ))
        except Exception as e:
            print(f"  !! {fname}: {e!r}")

    leaderboard.sort(key=lambda x: x['top_score'], reverse=True)
    print(f"  -> ranked {len(leaderboard)} files, {skipped_protocol} skipped by protocol filter.")
    
    if not leaderboard and diag is not None:
        print("\n  [DIAGNOSTIC] Nothing matched. What physion sees in the first file:")
        print("     file     :", diag[0])
        for s in diag[1]:
            print("     protocol :", s[:400])
        print("     -> adapt TARGET_PROTOCOL to one of the names above (or set it to '').")
    
    return leaderboard, skipped_protocol


# ------------------------------------------------------------
# 5) MAIN: FLEXIBLE ANALYSIS FOR ADULT OR YOUNG (BY AGE GROUP)
# ------------------------------------------------------------

def to_color(name):
    name = name.strip()
    if 'tab:'+name in mcolors.TABLEAU_COLORS:
        return 'tab:'+name
    if mcolors.is_color_like(name):
        return name
    raise ValueError(name)


def run_interactive_selector(leaderboard, dataset_label=""):
    """Interactive selector for a single leaderboard (one folder/age group)."""
    if not leaderboard:
        print(f"\n[WARNING] No valid files for dataset: '{dataset_label}'")
        return

    while True:
        plt.close('all')
        print("\n" + "="*60)
        label_text = f" {dataset_label}" if dataset_label else ""
        print("AVAILABLE SESSIONS LEADERBOARD" + label_text)
        print("="*60)
        for idx, item in enumerate(leaderboard[:10]):
            age_str = f" (Age: {item.get('age', 'N/A')})" if 'age' in item else ""
            print(f"  [{idx + 1}] {item['file_name']}{age_str} (Score: {item['top_score']:.2f})")
        print("  [Q] Quit")
        print("="*60)

        file_choice = input("\n>> Session index to load (1-10) [default 1]: ").strip()
        if file_choice.lower() == 'q':
            print("Exiting.")
            return
        try:
            champion = leaderboard[int(file_choice)-1 if file_choice else 0]
        except (ValueError, IndexError):
            print("[Error] Invalid option.")
            continue

        print(f"\nLoading {champion['file_name']} ...")
        data = prepare(champion['file_path'])
        dcnv = get_dcnv(data)
        tlim = champion['tlim_bounds']
        ranked = [int(r) for r in champion['sorted_rois']]
        color, line_color, alpha, n_rois = COLOR, LINE_COLOR, ALPHA, N_ROIS
        current_rois = ranked[:n_rois]

        while True:
            print(f"\nRanking (best first): {ranked[:15]} ...")
            print(f"Window {np.round(tlim, 1)} | {len(current_rois)} traces | ROIs (top row first): {current_rois}")
            print(f"fill color: {color} | dF/F line color: {line_color} | alpha: {alpha}")
            plt.close('all')
            fig = plot_traces([dict(data=data, rois=current_rois, color=color,
                                    line_color=line_color, alpha=alpha, scale=SCALE, dcnv=dcnv)], tlim)
            plt.show(block=False)
            plt.pause(0.5)

            print("\nOptions:")
            print("  [ENTER]     save this figure")
            print("  n 8         show the 8 best-ranked ROIs (any number)")
            print("  4,12,0,8    show these specific ROIs (first = top row)")
            print("  c purple    deconvolved fill color (name, tab:xxx or #hex)")
            print("  l black     dF/F line color")
            print("  a 0.8       fill alpha/transparency (0.0-1.0, default=1.0)")
            print("  b           back to the session list")
            user_choice = input(">> Your choice: ").strip()
            cmd = user_choice.lower()

            try:
                if cmd == 'b':
                    break
                elif not cmd:
                    out_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD')
                    os.makedirs(out_dir, exist_ok=True)
                    # Include age group in filename if present
                    label = champion.get('group', '')
                    age = champion.get('age', '')
                    age_str = f"_Age{age}" if age >= 0 else ""
                    suffix = f"_{label}{age_str}" if label or age >= 0 else ""
                    out = os.path.join(out_dir, f"TRACES_DECONV_{DATASET_NAME}{suffix}_{os.path.splitext(champion['file_name'])[0]}.svg")
                    fig.savefig(out, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
                    print(f"\nSaved: {out}")
                    plt.close('all')
                    return
                elif cmd.startswith('n '):
                    n_rois = int(cmd[2:])
                    current_rois = ranked[:n_rois]
                elif cmd.startswith('c '):
                    color = to_color(user_choice[2:])
                elif cmd.startswith('l '):
                    line_color = to_color(user_choice[2:])
                elif cmd.startswith('a '):
                    alpha = float(user_choice[2:])
                    alpha = max(0.0, min(1.0, alpha))  # Clamp between 0 and 1
                    print(f"  Alpha set to: {alpha}")
                else:
                    current_rois = [int(x.strip()) for x in user_choice.split(',')]
            except ValueError as e:
                print(f"[Error] could not parse '{user_choice}' ({e}). Examples: 'n 8' | 'c purple' | '1,5,2,9'")


# ============================================================
# MAIN EXECUTION
# ============================================================

print("\n" + "="*70)
print(" TRACES PLOT - FLEXIBLE DATASET ANALYSIS")
print("="*70)
print(f"Dataset: {DATASET_NAME}")
print(f"NWB Folder: {NWB_FOLDER}")
print(f"Type: {'YOUNG (by age groups)' if IS_YOUNG else 'ADULT'}")
print(f"Protocol: {TARGET_PROTOCOL}")
print("="*70 + "\n")

if not os.path.isdir(NWB_FOLDER):
    print(f"[ERROR] NWB folder does not exist: {NWB_FOLDER}")
    print("Available folders in CURATED/Cibele:")
    for item in os.listdir(BASE_PATH):
        print(f"  - {item}")
    sys.exit(1)

if IS_YOUNG:
    # ============================================================
    # YOUNG DATASET: Split by age groups and process each separately
    # ============================================================
    print("[YOUNG MODE] Will split by age groups and process each separately\n")
    
    # Build leaderboards for each age group
    age_group_leaderboards = {}
    
    for interval in AGE_INTERVALS:
        group_name = f"P{interval[0]}-{interval[1]}"
        print(f"\n{'='*70}")
        print(f"Processing age group: {group_name}")
        print(f"{'='*70}")
        leaderboard, _ = build_leaderboard_from_scan(NWB_FOLDER, interval, group_name)
        age_group_leaderboards[group_name] = leaderboard
    
    # Run interactive selector for each age group
    for group_name in sorted(age_group_leaderboards.keys()):
        leaderboard = age_group_leaderboards[group_name]
        if not leaderboard:
            print(f"\n[WARNING] No valid files for age group: {group_name}. Skipping.")
            continue
        
        print(f"\n{'='*70}")
        print(f"ANALYZING AGE GROUP: {group_name}")
        print(f"{'='*70}")
        run_interactive_selector(leaderboard, f"({group_name})")
        
        # Ask if user wants to continue to next age group
        if len(age_group_leaderboards) > 1:
            cont = input(f"\nContinue to next age group? (Y/n): ").strip().lower()
            if cont == 'n':
                break
    
    print("\n[YOUNG MODE] Analysis complete for all age groups.")

else:
    # ============================================================
    # ADULT DATASET: Process single folder
    # ============================================================
    print("[ADULT MODE] Processing single folder...\n")
    
    leaderboard, _ = build_leaderboard_from_scan(NWB_FOLDER, None, DATASET_NAME)
    run_interactive_selector(leaderboard, f"({DATASET_NAME})")

print("\nDone!")









#%%
######################################################################
# Combined selected NWBs, ROIs, and colors for plotting
# Format: list of dicts with nwb_file, rois, dataset (-> group color) for each group
# Protocol: 2orientations-8contrasts-15repeats
######################################################################

# two sets of example cells: choose which one to plot with SELECTED_SET below
combinations_set_2 = [
    {
        "group": "SST-cre",
        "nwb_file": "2024_09_11-17-04-46",
        "rois": [17, 1],
        "dataset": "SST-cells_WT_Adult_V1" # for the color (#12522e)
    },
    {
        "group": "SST NR1 KO",
        "nwb_file": "2026_02_23-15-36-39",
        "rois": [1, 4],
        "dataset": "SST-GluN1_KO" # for the color (#5d1490)
    },
    {
        "group": "PV-cre adult",
        "nwb_file": "2024_05_28-15-49-09",
        "rois": [7, 6],
        "dataset": "PV-cells_WT_Adult_V1" # for the color (#b30a7b)
    },
    {
        "group": "PV-cre NR1 adult",
        "nwb_file": "2024_10_24-15-31-25",
        "rois": [5, 21],
        "dataset": "PV-GluN1_KO" # for the color (#9e0004)
    },
]

combinations_set_1 = [
   # {
   #     "group": "SST-cre P20-23",
   #     "nwb_file": "2025_07_18-16-09-54",
   #     "rois": [2, 3],
   #     "dataset": "SST-cells_WT_P20-P23_V1" # for the color
   # },
    {
        "group": "SST-cre adult",
        "nwb_file": "2024_12_26-15-44-29",
        "rois": [15, 20],
        "dataset": "SST-cells_WT_Adult_V1" # for the color
    },
    {
        "group": "SST-cre NR1 KO",
        "nwb_file": "2026_02_18-15-08-09",
        "rois": [2, 0],
        "dataset": "SST-GluN1_KO" # for the color
    },
    {
        "group": "PV-cre adult",
        "nwb_file": "2024_04_15-15-03-35",
        "rois": [2, 3],
        "dataset": "PV-cells_WT_Adult_V1" # for the color
    },
    {
        "group": "PV-cre NR1 KO",
        "nwb_file": "2024_11_19-15-56-51",
        "rois": [38, 37],
        "dataset": "PV-GluN1_KO" # for the color
    },
   # {
   #     "group": "Pyramidal adult",
   #     "nwb_file": "2025_10_29-14-39-31",
   #     "rois": [9, 13],
   #     "dataset": "PYR-SynGCaMP_WT_Adult_V1" # for the color
   # }
]

SELECTED_SET = 2 # 1 or 2
selected_combinations = combinations_set_1 if SELECTED_SET == 1 else combinations_set_2
# Plot traces for selected combinations
# Search for NWB files across all relevant dataset folders and plot them together
# Using protocol: ff-gratings-2orientations-8contrasts-15repeats

# Define dataset folders to search in for the selected NWB files
# Note: P20-23 files are in Young datasets, adults in Adult datasets
dataset_folders = [
    "SST-cells_WT_Young_V1",         # SST-cre P20-23
    "SST-cells_WT_Adult_V1",        # SST-cre adult
    "PV-cells_WT_Young_V1",         # PV-cre P20-23
    "PV-cells_WT_Adult_V1",         # PV-cre adult
    "PYR-SynGCaMP_WT_Young_V1",      # Pyramidal P20-23
    "PYR-SynGCaMP_WT_Adult_V1",      # Pyramidal adult
    "PYR-PV-SynGCaMP_WT_Young_V1",   # Pyramidal P20-23 (alternative folder)
]

# Find the NWB files
groups = []
base_path = os.path.expanduser('~/CURATED/Cibele/')

for combo in selected_combinations:
    nwb_file = combo["nwb_file"] + ".nwb"
    found_path = None
    
    for dataset in dataset_folders:
        folder = os.path.join(base_path, dataset, 'NWBs')
        candidate = os.path.join(folder, nwb_file)
        if os.path.exists(candidate):
            found_path = candidate
            break
    
    # If not found in the standard folders, search all dataset folders
    if found_path is None:
        all_folders = glob.glob(os.path.join(base_path, '*', 'NWBs'))
        for folder in all_folders:
            candidate = os.path.join(folder, nwb_file)
            if os.path.exists(candidate):
                found_path = candidate
                break
    
    # Debug: if still not found, list all NWB files that exist
    if found_path is None:
        print(f"\n[DEBUG] Could not find {nwb_file}")
        print(f"[DEBUG] Searched folders:")
        for dataset in dataset_folders:
            folder = os.path.join(base_path, dataset, 'NWBs')
            print(f"  - {folder}")
        all_folders = glob.glob(os.path.join(base_path, '*', 'NWBs'))
        for folder in all_folders:
            print(f"  - {folder}")
        print(f"[DEBUG] All NWB files found in CURATED/Cibele:")
        for folder in all_folders:
            files = glob.glob(os.path.join(folder, '*.nwb'))
            for f in files:
                print(f"  - {f}")
    
    if found_path:
        print(f"Found {nwb_file} in {found_path}")
        data = prepare(found_path)
        dcnv = get_dcnv(data)
        tlim = stim_window(data)
        
        groups.append({
            'data': data,
            'rois': combo["rois"],
            'color': COLOR, # deconvolved fill in lightgray
            'line_color': get_color(combo["dataset"]), # dF/F trace in the group color
            'alpha': ALPHA,
            'scale': SCALE,
            'dcnv': dcnv,
            'label': combo["group"]
        })
    else:
        print(f"WARNING: Could not find {nwb_file} in any dataset folder")

if groups:
    # Use the tlim from the first group as reference
    first_tlim = stim_window(groups[0]['data'])
    
    # Save path for SVG
    out_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'TRACES_COMBINED_selected_combinations.svg')
    
    # Reverse groups to plot Pyramidal at top, PV-cre middle, PV-cre NR1 KO at bottom
    # (matplotlib plots with y=0 at bottom, so first group goes to bottom by default)
    groups = groups[::-1]
    
    # Plot and save
    fig = plot_traces(groups, first_tlim, figsize=(10, 8), savepath=out_path)
    plt.show()
    print(f"\nFigure saved as: {out_path}")
else:
    print("No valid groups to plot")











#%%
######################################################################
# RUN / REST  A) SETTINGS + HELPERS -> run this cell first (self-contained: does not need the cells above)
#   all the settings of cells B to F are here:
#     DATASET_NAME -> the group of cells B and C (one group)
#     GROUPS       -> the groups of cells D, E and F (several groups, one row each)
#     protocol, ROIs (ONLY_RESPONSIVE), running threshold, colors, scale bar
#   cells:
#     B) one group (DATASET_NAME): one row per session + group mean, rest vs run
#     C) one group (DATASET_NAME): one figure per session + grand average, rest vs run
#     D) several groups (GROUPS): rest (group color) vs run (grey)
#     E) several groups (GROUPS): all episodes only
#     F) several groups (GROUPS): all episodes (group color) vs rest (grey)
#   traces: mean dF/F over the ROIs of each episode, per stimulus (angle x contrast);
#   run = mean running speed during the stimulus > RUN_SPEED_THRESHOLD, rest = the other episodes
######################################################################

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# merged/ root (parent of plots/): physion and general_tools are taken from there
try:
    merged_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError: # interactive window
    merged_root = os.path.join(os.path.expanduser('~'), 'code', 'merged')
sys.path.insert(0, os.path.join(merged_root, 'physion', 'src'))
sys.path.insert(0, merged_root)
from physion.analysis.read_NWB import Data
from physion.analysis.episodes.build import EpisodeData
from general_tools.cell_type_color_codes import get_color
from create_summaries.Preprocessing_Settings import get_dFoF_params

plt.rcParams['svg.fonttype'] = 'none'

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

DATASET_NAME = "PYR-SynGCaMP_WT_Adult_V1"
DATASET_NAME = "PV-cells_WT_Adult_V1"
DATASET_NAME = "SST-cells_WT_Adult_V1"
DATASET_NAME = "PV-cells_cond-GluN1-KO_Adult_V1"
DATASET_NAME = "SST-cells_cond-GluN1-KO_Adult_V1"

# groups of cells D, E and F (one row per group)
GROUPS = [
    "SST-cells_WT_Adult_V1",
    "SST-cells_cond-GluN1-KO_Adult_V1",
    "PV-cells_WT_Adult_V1",
    #PYR-SynGCaMP_WT_Adult_V1",
    "PV-cells_cond-GluN1-KO_Adult_V1",
]

BASE_PATH = os.path.expanduser('~/CURATED/Cibele/')
NWB_FOLDER = os.path.join(BASE_PATH, DATASET_NAME, 'NWBs')

TARGET_PROTOCOL = "ff-gratings-2orientations-8contrasts-15repeats"
ANGLES = None                # None = all orientations of the protocol (one block of contrasts per angle)

SESSIONS = []                # e.g. ["2025_10_29-14-39-31"]; empty = all sessions of the folder
ROIS = None                  # None = mean over all ROIs, or a list of ROI indices
# only the visually responsive ROIs of each session, as in the contrast summaries
# (Sensitivities_<group>_angle-*.npy: significant at >= 1 contrast, for any of the angles)
ONLY_RESPONSIVE = True      # False = all ROIs of the session
RESPONSIVE_SIGN = 'pos'      # 'pos' (significant increase) or 'pos+neg' (increase or decrease)

RUN_SPEED_THRESHOLD = 0.1    # cm/s, mean speed during the stimulus (0.1 = physion's behavior default)
MIN_EPISODES = 2             # min episodes per state and stimulus to draw a trace
BASELINE_SUBTRACT = False    # True = subtract the pre-stimulus mean of each episode

REST_COLOR = get_color(DATASET_NAME, default='tab:orange') # group color (general_tools/cell_type_color_codes.py), orange for Young
RUN_COLOR = 'grey'           # grey
STIM_COLOR = (0, 0, 0, 0.1)    # stimulus band: black at alpha 0.1, as the stimulus bands of the trace cells above
SCALE = 0.5                  # dF/F of the vertical scale bar
WITH_GRAND_AVERAGE = True    # extra figure: mean of the session means

OUT_DIR = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD', 'REST-vs-RUN')

# same dF/F settings as the summaries (PYR: no ROI/neuropil inclusion factor, interneurons: 1.15)
dFoF_parameters = get_dFoF_params(DATASET_NAME)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def find_session_files(folder, sessions=SESSIONS):
    files = sorted(glob.glob(os.path.join(folder, '*.nwb')))
    if sessions:
        names = [s.replace('.nwb', '') for s in sessions]
        files = [f for f in files if os.path.basename(f).replace('.nwb', '') in names]
    out = []
    for f in files:
        try:
            data = Data(f, metadata_only=True, verbose=False)
            if any(TARGET_PROTOCOL in p for p in data.protocols):
                out.append(f)
        except Exception as e:
            print(f"  !! {os.path.basename(f)}: {e!r}")
    print(f"{len(out)} sessions with protocol '{TARGET_PROTOCOL}' in {folder}")
    return out


# results kept in memory by the cells below depend on these settings: forget them when they change
# (RECOMPUTE = True in the cells is then only needed if the NWB or summary files changed)
try:
    if LAST_SETTINGS != (ONLY_RESPONSIVE, RESPONSIVE_SIGN, RUN_SPEED_THRESHOLD, ROIS,
                         TARGET_PROTOCOL, ANGLES, MIN_EPISODES, BASELINE_SUBTRACT):
        GROUP_CACHE, ALL_CACHE, ALLREST_CACHE = {}, {}, {}
except NameError:
    pass
LAST_SETTINGS = (ONLY_RESPONSIVE, RESPONSIVE_SIGN, RUN_SPEED_THRESHOLD, ROIS,
                  TARGET_PROTOCOL, ANGLES, MIN_EPISODES, BASELINE_SUBTRACT)

SENSITIVITIES = {} # group -> {session name: responsive ROIs (bool)}

def responsive_from_summaries(group):
    """ {session name: (nROIs,) bool} from the Sensitivities summaries of the group (all angles) """
    if group not in SENSITIVITIES:
        out = {}
        for fn in sorted(glob.glob(os.path.join(BASE_PATH, 'summary', 'Sensitivities_%s_angle-*.npy' % group))):
            for S in np.load(fn, allow_pickle=True):
                sig = S['significant_pos'] if RESPONSIVE_SIGN == 'pos' else S['significant_pos'] + S['significant_neg']
                name = os.path.basename(S['datafile']).replace('.nwb', '')
                out[name] = out.get(name, np.zeros(len(sig), bool)) | np.asarray(sig, bool).any(axis=1)
        if not out:
            print(f"  [!!] no Sensitivities_{group}_angle-*.npy summary: responsive ROIs unknown")
        SENSITIVITIES[group] = out
    return SENSITIVITIES[group]


def select_rois(fpath, nROIs):
    """ ROIs averaged for one session: ROIS (or all), and only the responsive ones if ONLY_RESPONSIVE
        (None = session skipped: not in the summaries, or other number of ROIs) """
    rois = np.arange(nROIs) if ROIS is None else np.array(ROIS)
    if not ONLY_RESPONSIVE:
        return rois
    group = os.path.basename(os.path.dirname(os.path.dirname(fpath))) # <BASE_PATH>/<group>/NWBs/<session>.nwb
    name = os.path.basename(fpath).replace('.nwb', '')
    responsive = responsive_from_summaries(group).get(name)
    if responsive is None or len(responsive) != nROIs:
        print(f"  [X] {name}: " + ("not in the Sensitivities summaries" if responsive is None else
              f"{len(responsive)} ROIs in the summary vs {nROIs} here") + " -> skipped")
        return None
    rois = rois[responsive[rois]]
    if len(rois) == 0:
        print(f"  [X] {name}: no responsive ROI -> skipped")
        return None
    return rois


def episode_speeds(data, Episodes):
    """Mean running speed of each episode during the stimulus window."""
    data.build_running()
    speeds = np.zeros(len(Episodes.time_start_realigned))
    for i, (t0, d) in enumerate(zip(Episodes.time_start_realigned, Episodes.time_duration)):
        cond = (data.t_running >= t0) & (data.t_running < t0+d)
        speeds[i] = data.running[cond].mean() if np.any(cond) else np.nan
    return speeds


def session_responses(fpath):
    """
    returns dict(t, conds, rest=[...], run=[...], n_rest, n_run, ...)
    conds: list of (angle, contrast), angle first then increasing contrast
    rest/run: per condition, array (n_episodes, time) of the ROI-averaged dF/F (or None)
    """
    data = Data(fpath, verbose=False)
    if not data.has_running():
        print(f"  [X] {os.path.basename(fpath)}: no running speed in this NWB")
        return None
    data.build_dFoF(verbose=False, **dFoF_parameters)
    protocol = [p for p in data.protocols if TARGET_PROTOCOL in p][0]
    Episodes = EpisodeData(data, quantities=['dFoF'], protocol_name=protocol, verbose=False)

    rois = select_rois(fpath, data.nROIs)
    if rois is None:
        return None
    resp = Episodes.dFoF[:, rois, :].mean(axis=1)           # (episodes, time)
    if BASELINE_SUBTRACT:
        resp = resp - resp[:, Episodes.t < 0].mean(axis=1, keepdims=True)

    speeds = episode_speeds(data, Episodes)
    running = speeds > RUN_SPEED_THRESHOLD
    angles = np.unique(Episodes.angle) if ANGLES is None else np.array(ANGLES)
    conds = [(a, c) for a in angles for c in np.unique(Episodes.contrast)]

    out = dict(name=os.path.basename(fpath).replace('.nwb', ''),
               t=Episodes.t, conds=conds, nROIs=len(rois),
               duration=float(Episodes.time_duration[0]),
               rest=[], run=[], n_rest=int(np.sum(~running)), n_run=int(np.sum(running)))
    for a, c in conds:
        cond = np.isclose(Episodes.angle, a) & np.isclose(Episodes.contrast, c)
        out['rest'].append(resp[cond & ~running] if np.sum(cond & ~running) >= MIN_EPISODES else None)
        out['run'].append(resp[cond & running] if np.sum(cond & running) >= MIN_EPISODES else None)
    print(f"  [v] {out['name']}: {out['nROIs']} ROIs | rest n={out['n_rest']} | run n={out['n_run']} episodes")
    return out


def draw_grating_icon(ax, angle, contrast=1.0):
    """Small static-grating square, stripes darker for higher contrast."""
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal'); ax.axis('off')
    box = Rectangle((-1, -1), 2, 2, fc='white', ec='0.5', lw=0.3)
    ax.add_patch(box)
    th = np.deg2rad(angle)
    dx, dy = np.cos(th), np.sin(th)
    gray = str(float(np.clip(1-contrast, 0, 0.95)))
    for o in np.linspace(-1.4, 1.4, 8):          # stripes perpendicular to the angle direction
        line, = ax.plot([o*dx-1.5*dy, o*dx+1.5*dy], [o*dy+1.5*dx, o*dy-1.5*dx],
                        color=gray, lw=0.6, solid_capstyle='butt')
        line.set_clip_path(box)


def add_figure_title(fig, text):
    """ title of the figure, above the panels (what the figure draws) """
    # just above the highest panel (and its titles, e.g. the angles), whatever the height of the figure
    top = max(ax.get_position().y1 for ax in fig.axes)
    fig.suptitle(text, fontsize=8, fontweight='bold', va='bottom', y=top + 0.22/fig.get_figheight())


def mean_sem(x):
    return x.mean(axis=0), x.std(axis=0)/np.sqrt(len(x))


def plot_rest_vs_run(resp, title, savepath=None):
    t, conds = resp['t'], resp['conds']
    n = len(conds)
    fig, AX = plt.subplots(2, n+1, figsize=(0.8*(n+1), 2.),
                           gridspec_kw=dict(height_ratios=[4, 1], wspace=0.05, hspace=0.05))

    # common y range across panels
    lo, hi = np.inf, -np.inf
    for state in ['rest', 'run']:
        for x in resp[state]:
            if x is not None:
                m, s = mean_sem(x)
                lo, hi = min(lo, (m-s).min()), max(hi, (m+s).max())
    if not np.isfinite(lo):
        print(f"  [X] {title}: not enough episodes to plot")
        plt.close(fig)
        return None
    pad = 0.05*(hi-lo)
    ylim = (lo-pad, hi+pad)

    for i, (a, c) in enumerate(conds):
        ax = AX[0, i]
        ax.axvspan(0, resp['duration'], color=STIM_COLOR, lw=0)
        for state, color in [('rest', REST_COLOR), ('run', RUN_COLOR)]:
            x = resp[state][i]
            if x is None:
                continue
            m, s = mean_sem(x)
            ax.fill_between(t, m-s, m+s, color=color, alpha=0.3, lw=0)
            ax.plot(t, m, color=color, lw=0.8)
        ax.set_ylim(ylim); ax.set_xlim(t[0], t[-1]); ax.axis('off')
        draw_grating_icon(AX[1, i], a, c)
        AX[1, i].text(0, -1.3, '%.2f' % c, ha='center', va='top', fontsize=5)
        if i == 0 or a != conds[i-1][0]:
            AX[1, i].set_title('%g°' % a, fontsize=6, loc='left', pad=1)
    AX[1, 0].text(-1.4, -1.3, 'contrast', ha='right', va='top', fontsize=5)

    # scale bars in the last column
    ax = AX[0, n]
    ax.set_ylim(ylim); ax.set_xlim(t[0], t[-1]); ax.axis('off')
    y0 = ylim[0]+0.1*(ylim[1]-ylim[0])
    ax.plot([t[0]]*2, [y0, y0+SCALE], 'k-', lw=1)
    ax.plot([t[0], t[0]+resp['duration']], [y0]*2, 'k-', lw=1)
    ax.text(t[0]+0.05*(t[-1]-t[0]), y0+SCALE/2, '%s ΔF/F' % SCALE,
            rotation=90, ha='left', va='center', fontsize=7)
    ax.text(t[0]+resp['duration']/2, y0, '%gs' % resp['duration'], ha='center', va='top', fontsize=7)
    AX[1, n].axis('off')

    AX[0, 0].set_title(title, loc='left', fontsize=8)
    AX[0, n].plot([], [], color=REST_COLOR, label='rest (n=%i)' % resp['n_rest'])
    AX[0, n].plot([], [], color=RUN_COLOR, label='run (n=%i)' % resp['n_run'])
    AX[0, n].legend(frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(0, 1.15))

    add_figure_title(fig, 'Rest (group color) vs run (grey) - %s' % title)

    if savepath:
        fig.savefig(savepath, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  saved: {savepath}")
    return fig


def grand_average(responses):
    """Mean across sessions of the session means (sessions treated as samples)."""
    ok = [r for r in responses if r is not None]
    conds = ok[0]['conds']
    out = dict(t=ok[0]['t'], conds=conds, duration=ok[0]['duration'],
               rest=[], run=[], n_rest=0, n_run=0)
    for state in ['rest', 'run']:
        for i in range(len(conds)):
            means = [r[state][i].mean(axis=0) for r in ok
                     if np.allclose(r['conds'], conds) and r[state][i] is not None]
            out[state].append(np.array(means) if len(means) >= MIN_EPISODES else None)
        out['n_%s' % state] = max([0]+[len(x) for x in out[state] if x is not None])
    return out


def plot_sessions_stacked(rows, labels, title, savepath=None):
    """
    One row per response (e.g. the mean of each NWB, then the group mean),
    each row with its own y range and a SCALE dF/F bar, grating icons at the bottom.
    """
    t, conds = rows[0]['t'], rows[0]['conds']
    n = len(conds)
    fig, AX = plt.subplots(len(rows)+1, n+1, figsize=(0.8*(n+1), 0.9*len(rows)+0.6),
                           gridspec_kw=dict(height_ratios=[4]*len(rows)+[1], wspace=0.05, hspace=0.1))

    for r, (resp, label) in enumerate(zip(rows, labels)):
        lo, hi = np.inf, -np.inf
        for state in ['rest', 'run']:
            for x in resp[state]:
                if x is not None:
                    m, s = mean_sem(x)
                    lo, hi = min(lo, (m-s).min()), max(hi, (m+s).max())
        ylim = (lo-0.05*(hi-lo), hi+0.05*(hi-lo)) if np.isfinite(lo) else (0, 1)

        for i in range(n):
            ax = AX[r, i]
            ax.axvspan(0, resp['duration'], color=STIM_COLOR, lw=0)
            for state, color in [('rest', REST_COLOR), ('run', RUN_COLOR)]:
                x = resp[state][i]
                if x is None:
                    continue
                m, s = mean_sem(x)
                ax.fill_between(t, m-s, m+s, color=color, alpha=0.3, lw=0)
                ax.plot(t, m, color=color, lw=0.8)
            ax.set_ylim(ylim); ax.set_xlim(t[0], t[-1]); ax.axis('off')
        AX[r, 0].set_title(label, loc='left', fontsize=6, pad=1)

        # row scale bar
        ax = AX[r, n]
        ax.set_ylim(ylim); ax.set_xlim(t[0], t[-1]); ax.axis('off')
        y0 = ylim[0]+0.1*(ylim[1]-ylim[0])
        ax.plot([t[0]]*2, [y0, y0+SCALE], 'k-', lw=1)
        ax.text(t[0]+0.05*(t[-1]-t[0]), y0+SCALE/2, '%s ΔF/F' % SCALE,
                rotation=90, ha='left', va='center', fontsize=5)

    # time bar under the last row, icons in the bottom row
    ax = AX[len(rows)-1, n]
    ax.plot([t[0], t[0]+rows[0]['duration']], [ax.get_ylim()[0]]*2, 'k-', lw=1)
    ax.text(t[0]+rows[0]['duration']/2, ax.get_ylim()[0], '%gs' % rows[0]['duration'],
            ha='center', va='top', fontsize=6)
    for i, (a, c) in enumerate(conds):
        draw_grating_icon(AX[-1, i], a, c)
        AX[-1, i].text(0, -1.3, '%.2f' % c, ha='center', va='top', fontsize=5)
        if i == 0 or a != conds[i-1][0]:
            AX[-1, i].set_title('%g°' % a, fontsize=6, loc='left', pad=1)
    AX[-1, 0].text(-1.4, -1.3, 'contrast', ha='right', va='top', fontsize=5)
    AX[-1, n].axis('off')

    AX[0, 0].annotate(title, (0, 1.4), xycoords='axes fraction', fontsize=8)
    AX[0, n].plot([], [], color=REST_COLOR, label='rest')
    AX[0, n].plot([], [], color=RUN_COLOR, label='run')
    AX[0, n].legend(frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(0, 1.3))

    add_figure_title(fig, 'Rest vs run, one row per session + group mean - %s' % title)

    if savepath:
        fig.savefig(savepath, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  saved: {savepath}")
    return fig

os.makedirs(OUT_DIR, exist_ok=True)

#%%
######################################################################
# RUN / REST  B) ONE GROUP (DATASET_NAME of cell A): ALL SESSIONS IN ONE FIGURE, rest vs run
# - one row per NWB (session): mean over its ROIs
# - last row: group mean (mean of the session means)
######################################################################

group_responses = []
for fpath in find_session_files(NWB_FOLDER):
    try:
        resp = session_responses(fpath)
    except Exception as e:
        print(f"  !! {os.path.basename(fpath)}: {e!r}")
        continue
    if resp is not None:
        group_responses.append(resp)

if group_responses:
    rows = group_responses + [grand_average(group_responses)]
    labels = [f"{r['name']} ({r['nROIs']} ROIs, rest n={r['n_rest']}, run n={r['n_run']})"
              for r in group_responses] + \
             [f"group mean ({len(group_responses)} sessions)"]
    out = os.path.join(OUT_DIR, f"REST-vs-RUN_{DATASET_NAME}_all-sessions.svg")
    plot_sessions_stacked(rows, labels, f"{DATASET_NAME} - mean of each NWB", savepath=out)
    plt.show()

#%%
######################################################################
# RUN / REST  C) ONE GROUP (DATASET_NAME of cell A): ONE FIGURE PER SESSION + GRAND AVERAGE, rest vs run
# - one figure per session (mean over its ROIs)
# - plus the grand average (mean of session means)
######################################################################

print(f"\nREST vs RUN | {DATASET_NAME} | {TARGET_PROTOCOL} | "
      f"run threshold={RUN_SPEED_THRESHOLD} cm/s")

responses = []
for fpath in find_session_files(NWB_FOLDER):
    try:
        resp = session_responses(fpath)
    except Exception as e:
        print(f"  !! {os.path.basename(fpath)}: {e!r}")
        continue
    if resp is None:
        continue
    responses.append(resp)
    out = os.path.join(OUT_DIR, f"REST-vs-RUN_{DATASET_NAME}_{resp['name']}.svg")
    plot_rest_vs_run(resp, f"{resp['name']} - mean of {resp['nROIs']} ROIs", savepath=out)
    plt.close('all')

if WITH_GRAND_AVERAGE and responses:
    ga = grand_average(responses)
    out = os.path.join(OUT_DIR, f"REST-vs-RUN_{DATASET_NAME}_grand-average.svg")
    fig = plot_rest_vs_run(ga, f"{DATASET_NAME} - mean of sessions", savepath=out)
    if fig is not None:
        # in the grand average, n = number of sessions
        for txt in fig.axes[len(ga['conds'])].get_legend().get_texts():
            txt.set_text(txt.get_text().replace('n=', 'sessions='))
        fig.savefig(out, format='svg', dpi=300, bbox_inches='tight', facecolor='white')

plt.show()





#%%
######################################################################
# RUN / REST  D) SEVERAL GROUPS (GROUPS of cell A): REST vs RUN
# - one row per group: mean of the session means (as the grand average of cell C),
#   rest in the group color, run in grey, grating icons on top
# - the session means of each group are kept in GROUP_CACHE: re-running only redraws
######################################################################

# GROUPS: in cell A
RECOMPUTE = False # True = read the NWBs again even if the group is already in GROUP_CACHE

try:
    GROUP_CACHE
except NameError:
    GROUP_CACHE = {}

for group in GROUPS:
    if group in GROUP_CACHE and not RECOMPUTE:
        continue
    print(f"\n--- {group} ---")
    dFoF_parameters = get_dFoF_params(group) # session_responses uses this global (dF/F settings of the group)
    responses = []
    for fpath in find_session_files(os.path.join(BASE_PATH, group, 'NWBs')):
        try:
            resp = session_responses(fpath)
        except Exception as e:
            print(f"  !! {os.path.basename(fpath)}: {e!r}")
            continue
        if resp is not None:
            responses.append(resp)
    GROUP_CACHE[group] = grand_average(responses) if responses else None
dFoF_parameters = get_dFoF_params(DATASET_NAME) # back to the dataset of cell A


def plot_groups_stacked(groups, savepath=None):
    """
    One row per group (mean of the session means), rest in the group color, run in RUN_COLOR,
    grating icons on top, legend (number of sessions) and SCALE dF/F bar on the right of each row.
    """
    rows = [(g, GROUP_CACHE[g]) for g in groups if GROUP_CACHE.get(g) is not None]
    for g in groups:
        if GROUP_CACHE.get(g) is None:
            print(f"  [X] {g}: no session to plot")
    t, conds, duration = rows[0][1]['t'], rows[0][1]['conds'], rows[0][1]['duration']
    n = len(conds)
    fig, AX = plt.subplots(len(rows)+1, n+1, figsize=(0.8*(n+1)+1, 1.2*len(rows)+0.6),
                           gridspec_kw=dict(height_ratios=[1]+[4]*len(rows), wspace=0.05, hspace=0.1))

    # grating icons on top
    for i, (a, c) in enumerate(conds):
        draw_grating_icon(AX[0, i], a, c)
        AX[0, i].text(0, -1.3, '%.2f' % c, ha='center', va='top', fontsize=5)
        if i == 0 or a != conds[i-1][0]:
            AX[0, i].set_title('%g°' % a, fontsize=6, loc='left', pad=1)
    AX[0, 0].text(-1.4, -1.3, 'contrast', ha='right', va='top', fontsize=5)
    AX[0, n].axis('off')

    for r, (group, resp) in enumerate(rows, start=1):
        group_color = get_color(group, default='tab:orange')

        # y range of the row (rest and run)
        lo, hi = np.inf, -np.inf
        for state in ['rest', 'run']:
            for x in resp[state]:
                if x is not None:
                    m, s = mean_sem(x)
                    lo, hi = min(lo, (m-s).min()), max(hi, (m+s).max())
        ylim = (lo-0.05*(hi-lo), hi+0.05*(hi-lo)) if np.isfinite(lo) else (0, 1)

        for i in range(n):
            ax = AX[r, i]
            ax.axvspan(0, duration, color=STIM_COLOR, lw=0)
            for state, color in [('rest', group_color), ('run', RUN_COLOR)]:
                x = resp[state][i]
                if x is None:
                    continue
                m, s = mean_sem(x)
                ax.fill_between(t, m-s, m+s, color=color, alpha=0.3, lw=0)
                ax.plot(t, m, color=color, lw=0.8)
            ax.set_ylim(ylim); ax.set_xlim(t[0], t[-1]); ax.axis('off')

        # scale bars and legend on the right of the row
        ax = AX[r, n]
        ax.set_ylim(ylim); ax.set_xlim(t[0], t[-1]); ax.axis('off')
        y0 = ylim[0]+0.05*(ylim[1]-ylim[0])
        ax.plot([t[0]]*2, [y0, y0+SCALE], 'k-', lw=1)
        ax.plot([t[0], t[0]+duration], [y0]*2, 'k-', lw=1)
        ax.text(t[0]+0.05*(t[-1]-t[0]), y0+SCALE/2, '%s ΔF/F' % SCALE,
                rotation=90, ha='left', va='center', fontsize=6)
        ax.text(t[0]+duration/2, y0, '%gs' % duration, ha='center', va='top', fontsize=6)
        ax.plot([], [], color=group_color, label='rest (sessions=%i)' % resp['n_rest'])
        ax.plot([], [], color=RUN_COLOR, label='run (sessions=%i)' % resp['n_run'])
        ax.legend(frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(0.2, 1.05))

    add_figure_title(fig, 'Rest (group color) vs run (grey) - mean of sessions')

    if savepath:
        fig.savefig(savepath, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  saved: {savepath}")
    return fig


out = os.path.join(OUT_DIR, "REST-vs-RUN_groups_%s.svg" % '_'.join(GROUPS))
plot_groups_stacked(GROUPS, savepath=out)
plt.show()







#%%
######################################################################
# RUN / REST  E) SEVERAL GROUPS (GROUPS of cell A): ALL EPISODES ONLY
# - same figure as cell D, but one trace per group with all the episodes (no rest / run split)
# - per session: mean over ROIs (all, or the responsive ones with ONLY_RESPONSIVE of cell A) and episodes,
#   then mean +/- sem over sessions
# - sessions without running speed are also included (no locomotion needed here)
######################################################################

# GROUPS: in cell A
RECOMPUTE = False # True = read the NWBs again even if the group is already in ALL_CACHE

try:
    ALL_CACHE
except NameError:
    ALL_CACHE = {}


def session_responses_all(fpath, dFoF_params):
    """ per condition (angle, contrast): mean over ROIs and over all episodes of one session """
    data = Data(fpath, verbose=False)
    data.build_dFoF(verbose=False, **dFoF_params)
    protocol = [p for p in data.protocols if TARGET_PROTOCOL in p][0]
    Episodes = EpisodeData(data, quantities=['dFoF'], protocol_name=protocol, verbose=False)

    rois = select_rois(fpath, data.nROIs)
    if rois is None:
        return None
    resp = Episodes.dFoF[:, rois, :].mean(axis=1)           # (episodes, time)
    if BASELINE_SUBTRACT:
        resp = resp - resp[:, Episodes.t < 0].mean(axis=1, keepdims=True)

    angles = np.unique(Episodes.angle) if ANGLES is None else np.array(ANGLES)
    conds = [(a, c) for a in angles for c in np.unique(Episodes.contrast)]
    means = []
    for a, c in conds:
        cond = np.isclose(Episodes.angle, a) & np.isclose(Episodes.contrast, c)
        means.append(resp[cond].mean(axis=0) if np.sum(cond) >= MIN_EPISODES else None)
    print(f"  [v] {os.path.basename(fpath)}: {len(rois)}{' responsive' if ONLY_RESPONSIVE else ''} ROIs | {len(resp)} episodes")
    return dict(t=Episodes.t, conds=conds, duration=float(Episodes.time_duration[0]), means=means)


for group in GROUPS:
    if group in ALL_CACHE and not RECOMPUTE:
        continue
    print(f"\n--- {group} ---")
    sessions = []
    for fpath in find_session_files(os.path.join(BASE_PATH, group, 'NWBs')):
        try:
            s = session_responses_all(fpath, get_dFoF_params(group))
        except Exception as e:
            print(f"  !! {os.path.basename(fpath)}: {e!r}")
            continue
        if s is not None:
            sessions.append(s)
    if not sessions:
        ALL_CACHE[group] = None
        continue
    conds = sessions[0]['conds']
    ok = [s for s in sessions if np.allclose(s['conds'], conds)]
    ALL_CACHE[group] = dict(t=ok[0]['t'], conds=conds, duration=ok[0]['duration'], n_sessions=len(ok),
                            all=[np.array([s['means'][i] for s in ok if s['means'][i] is not None])
                                 for i in range(len(conds))])


def plot_groups_all_stacked(groups, savepath=None):
    """ as plot_groups_stacked (cell 4/4), one trace per group with all the episodes """
    rows = [(g, ALL_CACHE[g]) for g in groups if ALL_CACHE.get(g) is not None]
    for g in groups:
        if ALL_CACHE.get(g) is None:
            print(f"  [X] {g}: no session to plot")
    t, conds, duration = rows[0][1]['t'], rows[0][1]['conds'], rows[0][1]['duration']
    n = len(conds)
    fig, AX = plt.subplots(len(rows)+1, n+1, figsize=(0.8*(n+1)+1, 1.2*len(rows)+0.6),
                           gridspec_kw=dict(height_ratios=[1]+[4]*len(rows), wspace=0.05, hspace=0.1))

    # grating icons on top
    for i, (a, c) in enumerate(conds):
        draw_grating_icon(AX[0, i], a, c)
        AX[0, i].text(0, -1.3, '%.2f' % c, ha='center', va='top', fontsize=5)
        if i == 0 or a != conds[i-1][0]:
            AX[0, i].set_title('%g°' % a, fontsize=6, loc='left', pad=1)
    AX[0, 0].text(-1.4, -1.3, 'contrast', ha='right', va='top', fontsize=5)
    AX[0, n].axis('off')

    for r, (group, resp) in enumerate(rows, start=1):
        group_color = get_color(group, default='tab:orange')

        lo, hi = np.inf, -np.inf
        for x in resp['all']:
            if len(x) > 1:
                m, s = mean_sem(x)
                lo, hi = min(lo, (m-s).min()), max(hi, (m+s).max())
        ylim = (lo-0.05*(hi-lo), hi+0.05*(hi-lo)) if np.isfinite(lo) else (0, 1)

        for i in range(n):
            ax = AX[r, i]
            ax.axvspan(0, duration, color=STIM_COLOR, lw=0)
            x = resp['all'][i]
            if len(x) > 1:
                m, s = mean_sem(x)
                ax.fill_between(t, m-s, m+s, color=group_color, alpha=0.3, lw=0)
                ax.plot(t, m, color=group_color, lw=0.8)
            ax.set_ylim(ylim); ax.set_xlim(t[0], t[-1]); ax.axis('off')

        # scale bars and legend on the right of the row
        ax = AX[r, n]
        ax.set_ylim(ylim); ax.set_xlim(t[0], t[-1]); ax.axis('off')
        y0 = ylim[0]+0.05*(ylim[1]-ylim[0])
        ax.plot([t[0]]*2, [y0, y0+SCALE], 'k-', lw=1)
        ax.plot([t[0], t[0]+duration], [y0]*2, 'k-', lw=1)
        ax.text(t[0]+0.05*(t[-1]-t[0]), y0+SCALE/2, '%s ΔF/F' % SCALE,
                rotation=90, ha='left', va='center', fontsize=6)
        ax.text(t[0]+duration/2, y0, '%gs' % duration, ha='center', va='top', fontsize=6)
        ax.plot([], [], color=group_color, label='all (sessions=%i)' % resp['n_sessions'])
        ax.legend(frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(0.2, 1.05))

    add_figure_title(fig, 'All episodes - mean of sessions')

    if savepath:
        fig.savefig(savepath, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  saved: {savepath}")
    return fig


out = os.path.join(OUT_DIR, "ALL-EPISODES_groups_%s.svg" % '_'.join(GROUPS))
plot_groups_all_stacked(GROUPS, savepath=out)
plt.show()

#%%
######################################################################
# RUN / REST  F) SEVERAL GROUPS (GROUPS of cell A): ALL EPISODES vs REST
# - one row per group: all the episodes in the group color, the rest episodes in grey (REST_VS_ALL_COLOR)
# - both traces from the same sessions: only sessions with a running speed are used (unlike cell E)
# - calculation, for each session and each stimulus (angle x contrast):
#     1) ROI-averaged dF/F of each episode: mean over all the ROIs of the session (or over ROIS),
#        or only over its responsive ROIs with ONLY_RESPONSIVE = True in cell A
#        (significant at >= 1 contrast in the Sensitivities summaries, RESPONSIVE_SIGN)
#     2) all  = mean over ALL the episodes of that stimulus (rest + run)
#        rest = mean over the episodes with a mean speed during the stimulus <= RUN_SPEED_THRESHOLD
#               (episodes without any speed sample are not rest: they only count in "all";
#                drawn only if the session has at least MIN_EPISODES rest episodes for that stimulus)
#     3) group trace = mean +/- sem over the session traces (sessions are the samples)
#   the rest episodes are part of "all": the two traces are not independent. The difference between them
#   is the effect of the run episodes on the average: if the animal rarely runs, the two traces overlap
######################################################################

# GROUPS: in cell A
REST_VS_ALL_COLOR = 'grey'   # color of the rest trace (all episodes: group color)
RECOMPUTE = False # True = read the NWBs again even if the group is already in ALLREST_CACHE

try:
    ALLREST_CACHE
except NameError:
    ALLREST_CACHE = {}


def session_responses_all_and_rest(fpath, dFoF_params):
    """ per condition (angle, contrast): mean over all episodes and mean over the rest episodes,
        of the ROI-averaged dF/F, for one session (None if no running speed) """
    data = Data(fpath, verbose=False)
    if not data.has_running():
        print(f"  [X] {os.path.basename(fpath)}: no running speed in this NWB")
        return None
    data.build_dFoF(verbose=False, **dFoF_params)
    protocol = [p for p in data.protocols if TARGET_PROTOCOL in p][0]
    Episodes = EpisodeData(data, quantities=['dFoF'], protocol_name=protocol, verbose=False)

    rois = select_rois(fpath, data.nROIs)
    if rois is None:
        return None
    resp = Episodes.dFoF[:, rois, :].mean(axis=1)           # (episodes, time)
    if BASELINE_SUBTRACT:
        resp = resp - resp[:, Episodes.t < 0].mean(axis=1, keepdims=True)
    resting = episode_speeds(data, Episodes) <= RUN_SPEED_THRESHOLD # False for episodes without speed (NaN)

    angles = np.unique(Episodes.angle) if ANGLES is None else np.array(ANGLES)
    conds = [(a, c) for a in angles for c in np.unique(Episodes.contrast)]
    all_means, rest_means = [], []
    for a, c in conds:
        cond = np.isclose(Episodes.angle, a) & np.isclose(Episodes.contrast, c)
        all_means.append(resp[cond].mean(axis=0) if np.sum(cond) >= MIN_EPISODES else None)
        rest_means.append(resp[cond & resting].mean(axis=0) if np.sum(cond & resting) >= MIN_EPISODES else None)
    print(f"  [v] {os.path.basename(fpath)}: {len(rois)}{' responsive' if ONLY_RESPONSIVE else ''} ROIs | "
          f"{len(resp)} episodes, {np.sum(resting)} rest")
    return dict(t=Episodes.t, conds=conds, duration=float(Episodes.time_duration[0]),
                all=all_means, rest=rest_means, n_episodes=len(resp), n_rest=int(np.sum(resting)))


for group in GROUPS:
    if group in ALLREST_CACHE and not RECOMPUTE:
        continue
    print(f"\n--- {group} ---")
    sessions = []
    for fpath in find_session_files(os.path.join(BASE_PATH, group, 'NWBs')):
        try:
            s = session_responses_all_and_rest(fpath, get_dFoF_params(group))
        except Exception as e:
            print(f"  !! {os.path.basename(fpath)}: {e!r}")
            continue
        if s is not None:
            sessions.append(s)
    if not sessions:
        ALLREST_CACHE[group] = None
        continue
    conds = sessions[0]['conds']
    ok = [s for s in sessions if np.allclose(s['conds'], conds)]
    ALLREST_CACHE[group] = dict(t=ok[0]['t'], conds=conds, duration=ok[0]['duration'], n_sessions=len(ok),
                                episodes=sum(s['n_episodes'] for s in ok), rest_episodes=sum(s['n_rest'] for s in ok))
    for state in ['all', 'rest']:
        ALLREST_CACHE[group][state] = [np.array([s[state][i] for s in ok if s[state][i] is not None])
                                       for i in range(len(conds))]
    # number of sessions with a rest trace (max over the stimuli)
    ALLREST_CACHE[group]['n_rest_sessions'] = max(len(x) for x in ALLREST_CACHE[group]['rest'])
    print(f"  {len(ok)} sessions | {ALLREST_CACHE[group]['episodes']} episodes, "
          f"{ALLREST_CACHE[group]['rest_episodes']} rest "
          f"({100*ALLREST_CACHE[group]['rest_episodes']/ALLREST_CACHE[group]['episodes']:.0f}%)")


def plot_groups_all_vs_rest(groups, savepath=None):
    """ as plot_groups_stacked (cell D): all episodes in the group color, rest episodes in REST_VS_ALL_COLOR """
    rows = [(g, ALLREST_CACHE[g]) for g in groups if ALLREST_CACHE.get(g) is not None]
    for g in groups:
        if ALLREST_CACHE.get(g) is None:
            print(f"  [X] {g}: no session to plot")
    t, conds, duration = rows[0][1]['t'], rows[0][1]['conds'], rows[0][1]['duration']
    n = len(conds)
    fig, AX = plt.subplots(len(rows)+1, n+1, figsize=(0.8*(n+1)+1, 1.2*len(rows)+0.6),
                           gridspec_kw=dict(height_ratios=[1]+[4]*len(rows), wspace=0.05, hspace=0.1))

    # grating icons on top
    for i, (a, c) in enumerate(conds):
        draw_grating_icon(AX[0, i], a, c)
        AX[0, i].text(0, -1.3, '%.2f' % c, ha='center', va='top', fontsize=5)
        if i == 0 or a != conds[i-1][0]:
            AX[0, i].set_title('%g°' % a, fontsize=6, loc='left', pad=1)
    AX[0, 0].text(-1.4, -1.3, 'contrast', ha='right', va='top', fontsize=5)
    AX[0, n].axis('off')

    for r, (group, resp) in enumerate(rows, start=1):
        group_color = get_color(group, default='tab:orange')

        # y range of the row (all and rest)
        lo, hi = np.inf, -np.inf
        for state in ['all', 'rest']:
            for x in resp[state]:
                if len(x) > 1:
                    m, s = mean_sem(x)
                    lo, hi = min(lo, (m-s).min()), max(hi, (m+s).max())
        ylim = (lo-0.05*(hi-lo), hi+0.05*(hi-lo)) if np.isfinite(lo) else (0, 1)

        for i in range(n):
            ax = AX[r, i]
            ax.axvspan(0, duration, color=STIM_COLOR, lw=0)
            # rest first, all on top: the group color stays visible where the traces overlap
            for state, color in [('rest', REST_VS_ALL_COLOR), ('all', group_color)]:
                x = resp[state][i]
                if len(x) < 2: # mean +/- sem needs at least 2 sessions
                    continue
                m, s = mean_sem(x)
                ax.fill_between(t, m-s, m+s, color=color, alpha=0.3, lw=0)
                ax.plot(t, m, color=color, lw=0.8)
            ax.set_ylim(ylim); ax.set_xlim(t[0], t[-1]); ax.axis('off')

        # scale bars and legend on the right of the row
        ax = AX[r, n]
        ax.set_ylim(ylim); ax.set_xlim(t[0], t[-1]); ax.axis('off')
        y0 = ylim[0]+0.05*(ylim[1]-ylim[0])
        ax.plot([t[0]]*2, [y0, y0+SCALE], 'k-', lw=1)
        ax.plot([t[0], t[0]+duration], [y0]*2, 'k-', lw=1)
        ax.text(t[0]+0.05*(t[-1]-t[0]), y0+SCALE/2, '%s ΔF/F' % SCALE,
                rotation=90, ha='left', va='center', fontsize=6)
        ax.text(t[0]+duration/2, y0, '%gs' % duration, ha='center', va='top', fontsize=6)
        ax.plot([], [], color=group_color, label='all (sessions=%i)' % resp['n_sessions'])
        ax.plot([], [], color=REST_VS_ALL_COLOR, label='rest (sessions=%i, %.0f%% of episodes)' % (
                resp['n_rest_sessions'], 100*resp['rest_episodes']/resp['episodes']))
        ax.legend(frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(0.2, 1.05))

    add_figure_title(fig, 'All episodes (group color) vs rest (grey) - mean of sessions')

    if savepath:
        fig.savefig(savepath, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  saved: {savepath}")
    return fig


out = os.path.join(OUT_DIR, "ALL-vs-REST_groups_%s.svg" % '_'.join(GROUPS))
plot_groups_all_vs_rest(GROUPS, savepath=out)
plt.show()

#%%
