import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def single_curve(ax, x, y, sy,
                 sy1=None, sy2=None,
                 color=None,
                 lw=1, ms=0, ls='-', m='',
                 label=None, alpha=1.,
                 alpha_std=0.3):

    if color is None:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    # we print a single curve
    ax.plot(x, y, color=color, lw=lw, label=label, linestyle=ls, marker=m, ms=ms, alpha=alpha)
    # then errorbars if needed:
    if (sy1 is not None) and (sy2 is not None) :
        ax.fill_between(x, 
                        np.array(y)-np.array(sy1), 
                        np.array(y)+np.array(sy2),
                        color=color, lw=0, alpha=alpha_std)
    elif (sy is not None):
        ax.fill_between(x, 
                        np.array(y)-np.array(sy), 
                        np.array(y)+np.array(sy),
                        color=color, lw=0, alpha=alpha_std)

def multicolored_line(x, y, norm_color_value,
                      ax=None, alpha=1.,
                      cmap='cool', lw=2):

    if ax is None:
        fig, ax = plt.subplots(1)
    else:
        fig = None
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(norm_color_value.min(), norm_color_value.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    # Set the values used for colormapping
    lc.set_array(norm_color_value)
    lc.set_linewidth(lw)    

    line = ax.add_collection(lc)
    return fig, ax, line

def multiple_curves(ax, X, Y, sY, COLORS, LABELS,
                    sY1=None, sY2=None,
                    lw=1, ms=0, ls='-', m='', alpha=1.,
                    alpha_std=0.3, colormap=plt.cm.viridis):
    
    # meaning we have to plot several curves
    if COLORS is None:
        COLORS = [colormap(i/(len(Y)-1)) for i in range(len(Y))]
    if (LABELS is None):
        LABELS = ['Y'+str(i+1) for i in range(len(Y))]
    for x, y, l, c in zip(X, Y, LABELS, COLORS):
        ax.plot(x, y,
                color=c, linestyle=ls,
                lw=lw, marker=m, ms=ms, label=l, alpha=alpha)

    # then errorbars if needed:
    if (sY1 is not None) and (sY2 is not None) :
        for x, y, sy1, sy2, c in zip(X, Y, sY1, sY2, COLORS):
            ax.fill_between(x, 
                            np.array(y)-np.array(sy1), 
                            np.array(y)+np.array(sy2),
                            color=c, lw=0, alpha=alpha_std)
    elif (sY is not None):
        for x, y, sy, c in zip(X, Y, sY, COLORS):
            ax.fill_between(x,
                            np.array(y)-np.array(sy), 
                            np.array(y)+np.array(sy),
                            color=c, lw=0, alpha=alpha_std)

def plot(x=None, y=None, sy=None, sy1=None, sy2=None, color=None,
         X=None, Y=None, sY=None, sY1=None, sY2=None,
         COLORS=None, colormap=plt.cm.viridis,
         fig = None, ax=None,
         lw=1, alpha_std=0.3, ms=0, m='', ls='-',alpha=1.,
         xlabel='', ylabel='', bar_label='', title='',
         label=None, LABELS=None,
         fig_args={},
         axes_args={},
         bar_scale_args=None,
         bar_legend_args=None,
         legend_args=None, no_set=False):
    
    """    
    return fig, ax
    """
    # getting or creating the axis
    if ax is None:
        fig, ax = plt.subplots(1)
        
    if (y is None) and (Y is None):
        y = x
        x = np.arange(len(y))

    if (Y is not None):
        if (X is None) and (x is not None):
            X = [x for i in range(len(Y))]
        elif (X is None):
            X = [np.arange(len(y)) for y in Y]

        multiple_curves(ax, X, Y, sY, COLORS, LABELS,
                        sY1=sY1, sY2=sY2,
                        alpha_std=alpha_std,
                        colormap=colormap,
                        lw=lw, ls=ls, m=m, ms=ms,
                        alpha=alpha)
    else:
        single_curve(ax, x, y, sy,
                     sy1=sy1, sy2=sy2,
                     color=color,
                     alpha_std=alpha_std,
                     lw=lw, label=label, ls=ls, m=m, ms=ms,
                     alpha=alpha)

    # if bar_legend_args is not None:
        # self.bar_legend(ax, **bar_legend_args)

    # if (label is not None) or (LABELS is not None):
        # if legend_args is not None:
            # self.legend(ax, **legend_args)
        # else:
            # self.legend(ax, frameon=False, size='small', loc='best')

    # if bar_scale_args is not None:
        # self.draw_bar_scales(ax, **bar_scale_args)
        # self.set_plot(ax, [], **axes_args)
    # elif not no_set:
        # self.set_plot(ax, **self.compute_axes_args(axes_args,
                                                   # xlabel=xlabel,
                                                   # ylabel=ylabel,
                                                   # title=title))
            
    return fig, ax

if __name__=='__main__':

    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    import plot_tools as pt
 
    plot(Y=3*np.random.randn(4,10),
         sY=np.random.randn(4,10),
         ls=':', m='o', ms=0.1, lw=0.4,
         xlabel='x-label (X)', ylabel='y-label (Y)')

    tstop, dt = 10, 1e-2
    t = np.arange(int(tstop/dt))*dt
    x = np.random.randn(len(t))*10.-70.
    for tt in np.cumsum(np.random.exponential(tstop/10., 10)):
        x[np.argmin(np.abs(tt-t))] = 10.

    plt.show()
