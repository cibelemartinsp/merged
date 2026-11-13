import numpy as np
import matplotlib.pylab as plt

def violin(data,
           x=0, 
           ax=None,
           color='tab:blue',
           show_mean=True,
           show_extrema=False,
           show_median=False,
           quantiles=[0.25, 0.5, 0.75],
           points=100,
           ms=1,
           figsize=(1.5, 1),
           legend=None):

    """
    return fig, ax
    """

    data = np.array(data)

    # getting or creating the axis
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig = plt.gcf()

    if show_mean:
        ax.plot([x], [data.mean()], 'o', ms=ms, color=color)

    violin_parts = ax.violinplot(data,
                                 positions=[x],
                                 quantiles=[quantiles],
                                 showextrema=show_extrema,
                                 showmedians=show_median,
                                 points=points)

    for pc, color in zip(violin_parts['bodies'], [color]):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')

    violin_parts['cquantiles'].set_linewidth(0.5)

    return fig, ax


if __name__=='__main__':

    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    import plot_tools as pt

    #plotting
    fig, ax = pt.figure(figsize=(.8,1))
    violin(np.random.randn(100), x=0, color=plt.cm.tab10(1), ax=ax, points=1000)
    violin(np.random.randn(100)+1, x=1, color=plt.cm.tab10(2), ax=ax, points=1000)
    pt.set_plot(ax, ['left'])
    plt.show()
