import numpy as np
import matplotlib.pylab as plt

from .figure import figure
from .adjust_plots import set_plot
from .annotations import draw_bar_scales
from .legend import legend

def time_freq(t, freqs, coefs,
              ax=None, Tbar=0.1,
              ax_args=dict(ylabel='freq. (Hz)')):
    
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 2))
        plt.subplots_adjust(left=.15, bottom=.1, right=.95, top=.9)
    else:
        fig = plt.gcf()

    # # time frequency power plot
    c = ax.contourf(t, freqs, coefs, cmap='PRGn')

    set_plot(ax, ['left'], **ax_args)
    draw_bar_scales(ax, Xbar=Tbar, Xbar_label='%.1fs'%Tbar, Ybar=1e-12)

    return fig, ax

def time_freq_signal(t, freqs, data, coefs,
                     xlabel='time (s)',
                     unit='unit',
                     freq_scale = 'lin',
                     fig_args=dict(axes_extents=[[[4,1],[1,1]],
                                                 [[4,2],[1,2]]],
                                   hspace=0.3, wspace=0.2,
                                   figsize=(1,1))):
   

    fig, AX = figure(**fig_args)
    AX[0][1].axis('off')
    
    AX[0][0].plot(t, data)
    set_plot(AX[0][0], ['left'], 
             ylabel='(%s)' % unit,
             xlim=[t[0], t[-1]])
    
    # # time frequency power plot
    c = AX[1][0].contourf(t, freqs, coefs, cmap='PRGn')
    set_plot(AX[1][0], ylabel='frequency (Hz)', 
             ylim=[freqs[0], freqs[-1]],
             yscale=freq_scale,
             xlim=[t[0], t[-1]], xlabel=xlabel)

    # # mean power plot over intervals
    AX[1][1].plot(np.abs(coefs).mean(axis=1), freqs,\
                  label='mean', color='tab:purple')
    # # max of power over intervals
    AX[1][1].plot(np.abs(coefs).max(axis=1), freqs,\
                  label='max.', color='tab:red')
    set_plot(AX[1][1], ['bottom'],
             xlabel=' env. (%s)' % unit,
             yscale=freq_scale,
             ylim=[freqs[0], freqs[-1]])
    AX[1][1].legend(loc=(0.1,1.1))
    
    return fig, AX

if __name__=='__main__':

    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    import plot_tools as pt

    import sys
    # sys.path.append('../../')
    # from analyz.freq_analysis.wavelet_transform import my_cwt # continuous wavelet transform
    sys.path.append('../utils/')
    sys.path.append('../src/')
    from wavelet_transform import my_cwt # continuous wavelet transform
    
    dt, tstop = 1e-4, 1.
    t = np.arange(int(tstop/dt))*dt
    
    freq1, width1, freq2, width2, freq3, width3 = 10., 100e-3, 40., 40e-3, 70., 20e-3
    data = 3.2+np.cos(2*np.pi*freq1*t)*np.exp(-(t-.5)**2/2./width1**2)+\
        np.cos(2*np.pi*freq2*t)*np.exp(-(t-.2)**2/2./width2**2)+\
        np.cos(2*np.pi*freq3*t)*np.exp(-(t-.8)**2/2./width3**2)

    # Continuous Wavelet Transform analysis
    freqs = np.linspace(1, 90, 40)
    # freqs = np.logspace(0, 2, 20)
    coefs = my_cwt(data, freqs, dt)

    # fig, ax = time_freq(t, freqs, coefs)    
    fig, AX = time_freq_signal(t, freqs, data, coefs, freq_scale='lin')    

    plt.show()

