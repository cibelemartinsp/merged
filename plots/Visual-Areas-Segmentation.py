# %% [markdown]
# # Visual Area Segmentation
#
# ##### copied/adapted from:
#
# https://github.com/zhuangjun1981/NeuroAnalysisTools/blob/master/NeuroAnalysisTools/RetinotopicMapping.py
#
# ##### Cite the original work/implementation:
#
# [Zhuang et al., Elife (2017)](https://elifesciences.org/articles/18372)

# %%
# load packages:
import sys, os
# merged/ root (parent of plots/): physion and general_tools are taken from there
try:
    merged_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError: # interactive window
    merged_root = os.path.join(os.path.expanduser('~'), 'code', 'merged')
sys.path.insert(0, os.path.join(merged_root, 'physion', 'src'))
sys.path.insert(0, merged_root)

import numpy as np
import physion.utils.plot_tools as pt
from physion.intrinsic.tools import *
from physion.intrinsic.analysis import RetinotopicMapping
import matplotlib.pylab as plt
from PIL import Image

# %% [markdown]
# ## Load data

# %%
dataFolder = os.path.join(
    os.path.expanduser('~'),
    'DATA',
    'intrinsic_img',
    'SST',
    'SST_M2',
    '15-13-15'
)
#/Users/cibelemartinspinho/DATA/intrinsic_img/PYR_V1/PyrM_05
# retinotopic mapping data
maps = np.load(os.path.join(dataFolder, 'raw-maps.npy') ,
                allow_pickle=True).item()
maps['datafolder'] = dataFolder # needed by build_trial_data (looks there for vasculature.tif)

# recompute the retinotopic maps from the phase maps of each direction, with the physion convention
# (phase-diff = down-up / left-right, wrapped in [-pi/2, 3pi/2]) and the screen of the recording (metadata.npy):
# in some raw-maps.npy (SST_F3, SST_F4, SST_M1, PVNR1_M4) the phase-diff was stored with the opposite sign (up-down),
# which puts the whole map outside of the screen (e.g. altitude < -62 deg) -> saturated (red) position maps
RECOMPUTE_RETINOTOPY = 'auto' # 'auto': only if the stored phase-diff has the opposite sign, True or False
metadata_file = os.path.join(dataFolder, 'metadata.npy')
stim = np.load(metadata_file, allow_pickle=True).item().get('STIM', {}) if os.path.isfile(metadata_file) else {}
if 'zmin' not in stim and os.path.isfile(os.path.join(dataFolder, 'visual-stim.npy')): # as get_phase_to_angle_func
    stim = np.load(os.path.join(dataFolder, 'visual-stim.npy'), allow_pickle=True).item()
if 'zmin' not in stim:
    print('screen size not in metadata.npy or visual-stim.npy, taking default settings')
    stim = {'xmin':-57., 'xmax':57., 'zmin':-40., 'zmax':40.} # as get_phase_to_angle_func
if RECOMPUTE_RETINOTOPY == 'auto':
    # stored phase-diff = -(down-up) (resp. -(left-right)) everywhere -> opposite sign
    flipped = [np.median(np.abs(maps['%s-phase-diff' % map_type] + maps['%s-phase' % d0] - maps['%s-phase' % d1])) < 0.05
               for map_type, d0, d1 in [('altitude', 'down', 'up'), ('azimuth', 'left', 'right')]]
    RECOMPUTE_RETINOTOPY = any(flipped)
    print('stored phase-diff with the opposite sign -> retinotopic maps recomputed' if RECOMPUTE_RETINOTOPY else
          'stored retinotopic maps kept')
if RECOMPUTE_RETINOTOPY:
    for map_type, (d0, d1), bounds in [('altitude', ('down', 'up'), (stim['zmin'], stim['zmax'])),
                                       ('azimuth', ('left', 'right'), (stim['xmin'], stim['xmax']))]:
        phase_diff = (maps['%s-phase' % d0] - maps['%s-phase' % d1] + np.pi/2) % (2*np.pi) - np.pi/2
        maps['%s-phase-diff' % map_type] = phase_diff
        maps['%s-retinotopy' % map_type] = bounds[0] + (phase_diff + np.pi/2) / (2*np.pi) * (bounds[1] - bounds[0])
        print('%s: %.0f to %.0f deg (screen: %.0f to %.0f deg)' % (map_type,
              *np.percentile(maps['%s-retinotopy' % map_type], [5, 95]), *bounds))
#maps = dict(np.load(os.path.join(dataFolder, 'raw-maps.npz') , 
#               allow_pickle=True))
# vasculature picture: demo-mouse.tif, or vasculature-demo-Mouse.tif (e.g. PVNR1/2024_09_05, where
# demo-Mouse.xlsx is the mouse info sheet, not an image)
for vasc_name in ['demo-mouse.tif', 'vasculature-demo-Mouse.tif']:
    if os.path.isfile(os.path.join(dataFolder, vasc_name)):
        break
else:
    raise FileNotFoundError('no demo-mouse.tif or vasculature-demo-Mouse.tif in %s' % dataFolder)
print('vasculature picture:', vasc_name)
imVasc = np.array(Image.open(os.path.join(dataFolder, vasc_name))).astype(float)
fig, ax = pt.figure(ax_scale=(2,2))
ax.imshow(imVasc**1, cmap=plt.cm.gray) 
plt.axis('off');

# %% [markdown]
# # Retinotopic Maps

# %%
plot_retinotopic_maps(maps, map_type='altitude');

# %%
plot_retinotopic_maps(maps, map_type='azimuth');

# %% [markdown]
# # Perform Segmentation

# %%
data = build_trial_data(maps)
map_shape = data['aziPosMap'].shape
data['vasculatureMap'] = imVasc[::int(imVasc.shape[0]/map_shape[0]),\
                                ::int(imVasc.shape[1]/map_shape[1])]
if data['vasculatureMap'].shape != map_shape:
    # picture size not a multiple of the map size (e.g. 1652x1800 for 104x113 maps): resize it to the map size
    data['vasculatureMap'] = np.array(Image.fromarray(imVasc.astype(np.float32)).resize(map_shape[::-1],
                                                                                       Image.BILINEAR))
segmentation_params={'phaseMapFilterSigma': 2.,
                    'signMapFilterSigma': 1.,
                    'signMapThr': 0.8,
                    'eccMapFilterSigma': 10.,
                    'splitLocalMinCutStep': 5.,
                    'mergeOverlapThr': 0.4,
                    'closeIter': 3,
                    'openIter': 3,
                    'dilationIter': 15,
                    'borderWidth': 1,
                    'smallPatchThr': 100,
                    'visualSpacePixelSize': 0.5,
                    'visualSpaceCloseIter': 15,
                    'splitOverlapThr': 1.1}
data['params'] = segmentation_params
trial = RetinotopicMapping.RetinotopicMappingTrial(**data)
trial.processTrial(isPlot=False) # TURN TO TRUE TO VISUALIZE THE SEGMENTATION STEPS
_ = trial._getSignMap(onlySMplot=True)

# %%
fig, ax = pt.figure(ax_scale=(5,5))
h = RetinotopicMapping.plotPatches(trial.finalPatches, 
                                   plotaxis=ax)
ax.imshow(imVasc, cmap=pt.plt.cm.gray, 
          vmin=imVasc.min(), vmax=imVasc.max(), 
          extent=[*ax.get_xlim(), *ax.get_ylim()])
h = RetinotopicMapping.plotPatches(trial.finalPatches, 
                                   plotaxis=ax)
ax.axis('off');
#fig.savefig(os.path.join(os.path.expanduser('~'), # 'Desktop', 'fig.svg'))

# %% [markdown]
# # Summary Plot

# %%
fig, AX = pt.figure(axes=(5,1), 
                    ax_scale=(.9*2,.9*3), 
                    wspace=0.4)

for map_type, ax, bounds in zip(['altitude', 'azimuth'], AX[:2],
                                [[-50, 50],[-60, 60]]):
    im = ax.imshow(maps['%s-retinotopy' % map_type], cmap=plt.cm.PRGn,\
                vmin=bounds[0], vmax=bounds[1])
    fig.colorbar(im, ax=ax,
                label='visual e (deg.)')
    ax.set_title('%s map' % map_type)

ax = AX[2]
im = ax.imshow(trial.signMapf, cmap='jet', vmin=-1, vmax=1)
fig.colorbar(im, ax=ax, label='phase gradient sign')
ax.set_title('sign map')

ax=AX[-2]
h = RetinotopicMapping.plotPatches(trial.finalPatches, plotaxis=ax)
ax.imshow(imVasc, cmap=plt.cm.gray, vmin=imVasc.min(), vmax=imVasc.max(), 
        extent=[*ax.get_xlim(), *ax.get_ylim()])
h = RetinotopicMapping.plotPatches(trial.finalPatches, plotaxis=ax)
ax.set_title('w/ vasculature')

ax = AX[-1]
h = RetinotopicMapping.plotPatches(trial.finalPatches, 
                                   plotaxis=ax)
ax.imshow(imVasc, cmap=plt.cm.gray, 
        vmin=imVasc.min(), vmax=imVasc.max(), 
        extent=[*ax.get_xlim(), *ax.get_ylim()])

for ax in AX:
    ax.axis('equal')
    ax.axis('off')    

fig.savefig(os.path.join(os.path.expanduser('~'), 
                        'Desktop', 'fig.svg'))

# %% [markdown]
### Identify the Center (<20˚) of V1
# 
# %%

# Based on the excentricity center
fig, ax = pt.figure(ax_scale=(2,5))
m = 0*maps['altitude-retinotopy']
cond = ( np.abs(maps['altitude-retinotopy'])<10) &\
          ( np.abs(maps['azimuth-retinotopy'])<10)

# NOT WORKING
centerV1 = trial.finalPatches['patch01'].getCenter()
plt.plot([centerV1[1]], [centerV1[0]], 'ro', 
         alpha=0.3, ms=25)
m[~cond] = 1

h = RetinotopicMapping.plotPatches(trial.finalPatches, 
                                   alpha=0, plotaxis=ax)
ax.imshow(imVasc, cmap=plt.cm.gray, 
        vmin=imVasc.min(), vmax=imVasc.max(), 
        extent=[*ax.get_xlim(), *ax.get_ylim()])
ax.axis('off')
ax.axis('equal');

# %%
# Based on the center of retinotopy (NOT WORKING YET)
if False:
        fig, ax = pt.figure(ax_scale=(2,5))
        Center = 0*maps['altitude-retinotopy']
        cond = ( np.abs(maps['altitude-retinotopy'])<10) &\
                ( np.abs(maps['azimuth-retinotopy'])<10)
        Center[~cond] = 1

        # WEIGHT BY POWER MAPS ?
        # Center = maps['azimuth-power']/maps['azimuth-power'].max()
        # Center = maps['altitude-power']/maps['altitude-power'].max()

        ax.imshow(imVasc, cmap=plt.cm.gray, 
                vmin=imVasc.min(), vmax=imVasc.max(), 
                extent=[0, maps['vasculature'].shape[1]-1,
                        0, maps['vasculature'].shape[0]-1])
        ax.imshow(Center, cmap=plt.cm.gray, 
                vmin=0, vmax=1, alpha=0.3,
                extent=[0, maps['vasculature'].shape[1]-1,
                        0, maps['vasculature'].shape[0]-1])
        ax.axis('off')
        ax.axis('equal');


# %% [markdown]
# ## Illustrating the phase shift

# for shift in [0, np.pi/2., np.pi, 3*np.pi/2., 2*np.pi]:
for shift in np.linspace(0, 2*np.pi, 9):
        new_phase = perform_phase_shift(maps['up-phase'], shift)
        fig, ax = pt.figure()
        ax.axis('off')
        plot_phase_map(ax, fig, new_phase)
        ax.set_title('shift = %.2f Rd' % shift)













