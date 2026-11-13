# %%
import numpy as np
import matplotlib.pylab as plt
from matplotlib.cm import viridis
from .legend import bar_legend
    
def twoD_plot(x, y, z,
              ax=None, acb=None, fig=None,
              diverging=False,
              colormap=viridis,
              alpha=1.,
              vmin=None,
              vmax=None,
              scale='',
              # axes props
              axes_args=dict(xlim_enhancement=0., 
                             ylim_enhancement=0., 
                             tck_outward=0),
              fig_args=dict(figsize=(.9,1), right=6.),
              xlabel=None, ylabel=None, title=None,
              # bar legend
              bar_legend_args=None,
              aspect='auto',
              interpolation='none'):
    """
    surface plots for x, y and z 1 dimensional data

    switch to bar_legend=None to remove the bar legend
    """
    
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(2.1, 1.5))
        plt.subplots_adjust(bottom=0.3, left=0.25, right=0.6)
    else:
        fig = plt.gcf()
        
    if diverging and (colormap==viridis):
        colormap = cm.PiYG # we switch to a diverging colormap
        
    x, y = np.array(x), np.array(y)
    Z = np.ones((len(np.unique(y)), len(np.unique(x))))*np.nan
    for i, yy in enumerate(np.unique(y)):
        cond1 = y==yy
        for j, xx in enumerate(np.unique(x[cond1])):
            try:
                Z[i,j] = z[cond1][x[cond1]==xx]
            except ValueError:
                print('multiple values for the same (x=', xx, ', y=', yy, ') couple: ', z[cond1][x[cond1]==xx])
                Z[i,j] = np.mean(z[cond1][x[cond1]==xx])
    # z1 = np.array(Z).reshape(len(np.unique(y)), len(np.unique(x)))
    z1 = Z
    
    if vmin is None:
        if diverging:
            vmin = -np.max(np.abs(z))
        else:
            vmin = np.min(z)
    if vmax is None:
        if diverging:
            vmax = np.max(np.abs(z))
        else:
            vmax = np.max(z)
            
    ac = ax.imshow(z1,
                   interpolation=interpolation,
                   extent = (x.min(), x.max(), y.min(), y.max()),
                   vmin = vmin,
                   vmax = vmax,
                   alpha=alpha,
                   cmap=colormap,
                   origin='lower',
                   aspect=aspect)


    if bar_legend_args is not None:

        if 'bounds' not in bar_legend_args:
            bar_legend_args['bounds'] = [vmin, vmax]
        if 'colormap' not in bar_legend_args:
            bar_legend_args['colormap'] = colormap
        acb = bar_legend(fig, **bar_legend_args)
        
    else:
        
        acb = None
        
    return fig, ax, acb
    

def matrix(z,
           x=None, y=None, 
           ax=None, acb=None, fig=None,
           diverging=False,
           colormap=viridis,
           alpha=1.,
           vmin=None, vmax=None,
           aspect='equal', # switch to 'auto' if needed
           origin='lower',
           # axes props
           axes_args=dict(xlim_enhancement=0., 
                          ylim_enhancement=0., 
                          tck_outward=0),
           # bar legend
           bar_legend_args=None,
           # other args
           interpolation='none'):

    if (x is None) and (y is None):
        x, y = np.meshgrid(np.arange(z.shape[0]), 
                           np.arange(z.shape[1]), 
                           indexing='ij')
        
    if (ax is None) and (acb is None):
        fig, ax = plt.subplots(1, figsize=(2.1, 1.5))
        plt.subplots_adjust(right=0.7)
    else:
        fig = plt.gcf()
        
    if diverging and (colormap==viridis):
        colormap = cm.PiYG # we switch to a diverging colormap

    if vmin is None:
        vmin=z.min()
    if vmax is None:
        vmax=z.max()

    im = ax.imshow(z.T,
                   interpolation=interpolation,
                   extent = (x.min(), x.max(), y.min(), y.max()),
                   vmin = vmin,
                   vmax = vmax,
                   alpha=alpha,
                   cmap=colormap,
                   origin=origin,
                   aspect=aspect)
    
    if bar_legend_args is not None:

        if 'bounds' not in bar_legend_args:
            bar_legend_args['bounds'] = [vmin, vmax]
        if 'colormap' not in bar_legend_args:
            bar_legend_args['colormap'] = colormap
        acb = bar_legend(fig, **bar_legend_args)
        
    else:
        
        acb = None
        
    return fig, ax, acb


if __name__=='__main__':
    pass
    # %%
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    import plot_tools as pt


    x, y = np.meshgrid(np.arange(1, 11), np.arange(1, 20), indexing='ij')
    z = y*x*x

    fig1, ax, _ = matrix(z, aspect='equal', bar_legend_args={})

    x, y, z = np.array(x).flatten(),\
              np.array(y).flatten(),\
              np.array(z).flatten()*np.random.randn(len(z.flatten()))
    index = (x<15) & (y<x)
    # np.random.shuffle(index)
    x, y, z = x[index], y[index], z[index]

    fig2, ax, acb = twoD_plot(x, y, z,
                              vmin=-7, vmax=7,
                              bar_legend_args={'label':'color',
                                               'ticks':[-7, 0, 7],
                                               'ticks_labels':['-7', '0', '>7'],
                                               'color_discretization':20})

    pt.set_plot(ax, xlabel='x-label (X)', ylabel='y-label (Y)')
    
    plt.show()

# %%
