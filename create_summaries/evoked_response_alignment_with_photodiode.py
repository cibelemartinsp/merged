#%%
import os, sys
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
from physion.analysis.read_NWB import Data
import numpy as np
import matplotlib.pyplot as plt



def plot_file_photodiode_shift(filename):

    #build
    data = Data(filename, verbose=False)
    data.build_rawFluo()
    quantities = ['rawFluo']

    if data.has_photodiode(): 
        data.build_photodiode()
        quantities.append('photodiode')

    Episodes = EpisodeData(data, 
                            quantities=quantities, 
                            protocol_name = data.protocols[0],
                            verbose=False)

    Episodes.rawFluo = np.mean(Episodes.rawFluo, axis = 1)
    N_ep = Episodes.rawFluo.shape[0]

    #compute values of interest
    period_evoked = []
    for quantity in quantities : 
        val = getattr(Episodes, quantity)

        for i in range(3):
            period_start = int((i/3) * N_ep)
            period_stop = int(((i+1)/3)*N_ep)
            period_evoked.append(np.mean(val[period_start:period_stop, :], axis = 0)) 

    #plot

    mosaic = "ABC" 
    fig = plt.figure(layout="constrained", figsize = (12,4))

    ax_dict = fig.subplot_mosaic(mosaic)
    labels = ['1st period', '2nd period', '3rd period']
    for i in range(3):
        ax_dict["A"].plot(Episodes.t, period_evoked[i]/np.max(period_evoked[i]), label = labels[i])
    ax_dict["A"].set_xlabel('time')
    ax_dict["A"].set_ylabel('norm mean rawFluo')
    ax_dict["A"].legend()

    if data.has_photodiode(): 
        for i in range(3,6):
            ax_dict["B"].plot(Episodes.t, period_evoked[i])

        ax_dict["B"].set_xlabel('time')
        ax_dict["B"].set_ylabel('mean Photodiode')

        mask_first_min = [data.t_photodiode <= 60][0]
        ax_dict["C"].plot(data.t_photodiode[mask_first_min], data.photodiode[mask_first_min], color = 'black')
        ax_dict["C"].set_xlabel('time')
        ax_dict["C"].set_ylabel('Photodiode')
    else : 
        ax_dict["B"].annotate(text = 'no photodiode', xy = (0.5,0.5), fontsize = 15)
        ax_dict["B"].axis("off")
        ax_dict["C"].axis("off")
    
    return fig 






#%% plot all session average raw and photodiode


base_path = os.path.join(os.path.expanduser('~'), 'CURATED', 'Cibele', 'SST-cells_cond-GluN1-KO_Adult_V1', 'NWBs') + '/'

filename = '2026_03_05-16-07-55.nwb'
data = Data(base_path + filename, verbose=False) 
print(data.protocols)
data.build_rawFluo()
#data.build_neuropil()
data.build_photodiode()
mean_raw = np.mean(data.rawFluo, axis = 0)
for plot_xlim in range (0,data.nwbfile.stimulus['time_start'].data[-1][0]-200, 200):

    fig, ax = plt.subplots(2,1,figsize = (15,3))
    ax[0].plot(data.t_rawFluo, mean_raw)
    #ax[0].plot(data.t_rawFluo, np.mean(data.neuropil, axis = 0))
    ax[1].plot(data.t_photodiode, data.photodiode, 'grey')
    for time_start, duration in zip(data.nwbfile.stimulus['time_start'].data[:], data.nwbfile.stimulus['time_duration'].data[:]) : 
        ax[0].axvspan(time_start[0], time_start[0] + duration[0], color='grey', alpha=0.3)
        ax[1].axvspan(time_start[0], time_start[0] + duration[0], color='grey', alpha=0.3)

    ax[0].set_ylabel('All roi \n average rawfluo')
    ax[1].set_ylabel('photodiode')

    for ax in [ax[0],ax[1]]:
        ax.set_xlim(plot_xlim, plot_xlim +200)
        ax.set_xlabel('time')

