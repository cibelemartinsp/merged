#%%
import numpy as np
import os, sys 
sys.path += ['.suite2p']
from suite2p.run_s2p import run_s2p
from suite2p.extraction.masks import create_masks
from suite2p.extraction.extract import extract_traces
from torch import device
#%%
#%% run_s2p in order to do registration and detection and to create data.bin


folderpath = "/home/user/Desktop/h5-11142025-001"
db = suite2p.default_db()
db['data_path'] = [folderpath] #folder of the h5 file
db['input_format'] = 'h5'
#db['file_list'] = os.listdir(folderpath)
db['save_path0'] = folderpath 

settings = suite2p.default_settings()
settings['torch_device'] = 'cpu'
#settings['fs'] = 
settings['run']['do_detection'] =  True
settings['run']['do_deconvolution'] = False
run_s2p(db=db, settings=settings)

#%%
#PARAMETERS TO VARY

folderpath = "/home/user/Desktop/h5-11142025-001"
neuropil_params = {
    'inner_neuropil_radius':3,
    'min_neuropil_pixels':350,
    'lam_percentile':50.0,
    'allow_overlap':False,
}

#%%

path = folderpath + '/suite2p/plane0/'

spks = np.load(path + 'spks.npy', allow_pickle=True)
db_inside_plane0 =  np.load(path + 'db.npy', allow_pickle=True)
db_outside_plane0 =  np.load(folderpath + '/suite2p/db.npy', allow_pickle=True)

settings_inside_plane0 =  np.load(path + 'settings.npy', allow_pickle=True)
settings_outside_plane0 =  np.load(folderpath + '/suite2p/settings.npy', allow_pickle=True)


ops = np.load(path +'ops.npy', allow_pickle=True).item()
detect_outputs =  np.load(path + 'detect_outputs.npy', allow_pickle=True)



F_orig =  np.load(path + 'F.npy', allow_pickle=True)

Fneu_orig =  np.load(path + 'Fneu.npy', allow_pickle=True)


stat = np.load(path + 'stat.npy', allow_pickle=True)
datadotbin = suite2p.io.BinaryFile(Ly=512, Lx=512, filename= '/home/user/Desktop/datawithbin/suite2p/plane0/data.bin')

settings =  np.load('/home/user/Desktop/datawithbintest/settings.npy', allow_pickle=True).item()
settings['torch_device'] = 'cpu'
np.save('/home/user/Desktop/test_data/settings.npy', settings)

ops = np.load('/home/user/Desktop/test_data/suite2p/plane0/ops.npy', allow_pickle=True).item()
ops2 = np.load('/home/user/Desktop/test_data/ops.npy', allow_pickle=True).item()


reg_outputs = np.load('/home/user/DATA/physion_Demo-Datasets/PYR-WT/processed/2025_11_14/13-54-32/h5-11142025-001/suite2p/plane0/reg_outputs.npy', allow_pickle=True)

#%%

#from suite2p.parameters import default_settings
f_raw = suite2p.io.BinaryFile(Ly=512, Lx=512, filename= '/home/user/DATA/physion_Demo-Datasets/PYR-WT/processed/2025_11_14/13-54-32/h5-11142025-001/Ch2-Green-plane0.h5')
f_reg_empty = suite2p.io.BinaryFile(Ly=512, Lx=512, filename='reg_file', n_frames=f_raw.shape[0], write=True) 

reg_outputs = registration_wrapper(f_reg_empty, f_raw=f_raw, f_reg_chan2=None, f_raw_chan2=None,
                        refImg=None, align_by_chan2=False, save_path='/home/user/Desktop', aspect=1.,
                        badframes=None, settings=default_settings()['registration'], device=torch.device("cpu"))


#%% CREATE DEFAULT STATS.NPY

ypix = np.array([253,254,255,256,257,258,259,
        253,254,255,256,257,258,259,
        253,254,255,256,257,258,259])

xpix = np.array([255,255,255,255,255,255,255,
        256,256,256,256,256,256,256,
        257,257,257,257,257,257,257])

lam = np.array([1/len(ypix)]*len(ypix))

npix = len(ypix)
npix_soma = len(ypix) #all pixels belong to the soma
soma_crop = [True]*npix
overlap = [False]*npix

for nroi in range(len(stat)-1):
      stat[nroi]['ypix'] = ypix
      stat[nroi]['xpix'] = xpix
      stat[nroi]['lam'] = lam
      stat[nroi]['npix'] = npix
      stat[nroi]['npix_soma'] = npix_soma
      stat[nroi]['soma_crop'] = soma_crop
      stat[nroi]['overlap'] = overlap

np.save('/home/user/Desktop/test_data/suite2p/plane0/stat.npy', stat)


#%%


ypix = [255,256,257,
        255,256,257,
        255,256,257]

xpix = [255,255,255,
        256,256,256,
        257,257,257]

lam = [1/len(ypix)]*len(ypix)

med = [256,256]

footprint = np.float64(0) #0 for sourcery/cellpose by default
mrs = stat[0]
mrs0 = np.nan
compact = np.nan
solidity = np.nan
npix = len(ypix)
npix_soma = len(ypix) #all pixels belong to the soma
soma_crop = [True]*npix
overlap = [False]*npix
radius = np.nan
aspect_ratio = np.nan
npix_norm_no_crop = np.nan
npix_norm = np.nan
skew = np.nan
std = np.nan
neuropil_mask = np.nan 
#%%

ypix = np.array([255,256,257,
        255,256,257,
        255,256,257])

xpix = np.array([255,255,255,
        256,256,256,
        257,257,257])

lam = np.array([1/len(ypix)]*len(ypix))

med = [256,256]

footprint = np.float64(0) #0 for sourcery/cellpose by default
npix = len(ypix)
npix_soma = len(ypix) #all pixels belong to the soma
soma_crop = np.array([True]*npix)
overlap = np.array([False]*npix)
neuropil_mask = np.array([])



default_stat = {}
for key in stat[0].keys() : 
    print(key)
    if str(key) in locals().keys() : 
        print('a')
        default_stat[key] = locals().get(key)
    else :
         default_stat[key] = stat[0][key]

default_stat = np.array([default_stat])

default_stat = np.array([stat[0]])
np.save('/home/user/Desktop/default_stat.npy', default_stat)


dstat = np.load('/home/user/Desktop/default_stat.npy', allow_pickle=True)




ypix = np.array([])

xpix = np.array([])

lam = np.array([])

med = []

footprint = np.float64() #0 for sourcery/cellpose by default
mrs = np.float64()
mrs0 = np.float64()
compact = np.float64()
solidity = np.float64()
npix = int()
npix_soma = int()
soma_crop = np.array([])
overlap = np.array([])
radius = np.float64()
aspect_ratio = np.float64()
npix_norm_no_crop = np.float64()
npix_norm = np.float64()
skew = np.float32()
std = np.float32()
neuropil_mask = np.array([])



default_stat = {}
for key in stat[0].keys() : 
    print(key)
    default_stat[key] = locals().get(key)
default_stat = np.array([default_stat])
np.save('/home/user/Desktop/test_data2/suite2p/planetest3/stat.npy', default_stat)
