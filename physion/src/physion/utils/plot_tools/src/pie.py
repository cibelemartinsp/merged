# %%
import numpy as np
import matplotlib.pylab as plt

def pie(data,
        ax=None,
        ext_labels= None,
        pie_labels = None,
        explodes=None,
        COLORS=None,
        ext_labels_distance = 1.1,
        pie_labels_distance = 0.6,
        pie_labels_digits = 1,
        ext_text_settings=dict(weight='normal'),
        pie_text_settings=dict(weight='normal'),
        center_circle=0.3,
        title='',
        pie_args={},
        legend=None):

    """    
    return fig, ax
    """
    
    # getting or creating the axis
    if ax is None:
        if legend is not None:
            fig, ax = plt.subplots(1, figsize=(2,1.5))
            plt.subplots_adjust(right=.7)
        else:
            fig, ax = plt.subplots(1, figsize=(1.5,1))
            plt.subplots_adjust(left=.3, right=.8)
    else:
        fig = plt.gcf()
        
    if COLORS is None:
        COLORS = [plt.cm.tab10(i) for i in range(len(data))]
    if (explodes is None):
        explodes = np.zeros(len(data))
    if (ext_labels is None):
        ext_labels = np.zeros(len(data), dtype=str)

    if pie_labels is not None:
        pie_labels_map = {}
        for pl, val in zip(pie_labels, data):
            pie_labels_map[str(np.round(100.*val/np.sum(data),pie_labels_digits))] = pl
        def func(pct):
            return pie_labels_map[str(np.round(pct,pie_labels_digits))]
    else:
        def func(pct):
            return ''
        
    
    wedges, ext_texts, pie_texts = ax.pie(data,
                                          labels=ext_labels,
                                          autopct=func,
                                          explode=explodes,
                                          pctdistance=pie_labels_distance,
                                          labeldistance=ext_labels_distance,
                                          colors=COLORS, **pie_args)

    if 'fontsize' not in pie_text_settings:
        pie_text_settings['fontsize'] = 8
    if 'fontsize' not in ext_text_settings:
        ext_text_settings['fontsize'] = 8
    
    plt.setp(pie_texts, **pie_text_settings)
    plt.setp(ext_texts, **ext_text_settings)
    
    Centre_Circle = plt.Circle((0,0),
                                center_circle, 
                                fc='white')
    ax.add_artist(Centre_Circle)
                                  
    if legend is not None:
        if 'loc' not in legend:
            legend['loc']=(1.21,.2)
        ax.legend(**legend)

    if title!='':
        ax.set_title(title)
        
    ax.axis('equal')
    return fig, ax


if __name__=='__main__':
    pass    
    # %%
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    import plot_tools as pt

    data = .5+np.abs(np.random.randn(3))*.4

    #plotting
    fig, ax = pie(data,
                     ext_labels = ['Data1', 'Data2', 'Data3'],
                     pie_labels = ['%.1f%%' % (100*d/data.sum()) for d in data],
                     ext_labels_distance=1.2,
                     explodes=0.05*np.ones(len(data)),
                     center_circle=0.2,
                     #pie_args=dict(rotate=90), # e.g. for rotation
                     legend=None)  # set legend={} to have it appearing
    plt.show()

# %%
