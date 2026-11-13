# %%
import sys

sys.path.append('..')
import plot_tools as pt
pt.set_style('dark')
import matplotlib.pylab as plt
import numpy as np

# %%

def gen_fig():

    fig, AX = pt.figure((2,1))

    pt.scatter(np.random.randn(100), ax=AX[0], ms=5)
    pt.scatter(np.random.randn(100), ax=AX[0], ms=5)
    pt.set_plot(AX[0], xlabel='obs #', 
                ylabel='value')
    pt.annotate(AX[0], 'A', (-0.4, 1.05), 
                fontsize=8, bold=True)
    AX[1].hist(np.random.randn(100))
    AX[1].hist(np.random.randn(100))
    pt.set_plot(AX[1], ylabel='hist', xlabel='value')
    pt.annotate(AX[1], 'some\nlabel', (1,1), 
                va='top', ha='right', fontsize=6)
    pt.annotate(AX[1], 'B', (-0.25, 1.05), 
                fontsize=8, bold=True)
    return fig

fig = gen_fig()
pt.save(fig, on='temp', fig_name='fig.png')
    
# %%
# CHECK THAT THE FIGURE NICELY FITS
import tempfile
from IPython.display import Image
Image(os.path.join(tempfile.tempdir, 'fig.png'))

# %%
# now the same with the manuscript style
pt.set_style('manuscript')
fig = gen_fig()
pt.save(fig, on='Desktop', fig_name='fig.png', dpi=300)
Image(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.png'))

# %%
fig, ax = pt.figure()
pt.pie([0.4, 0.6], ax=ax,
       pie_labels=['40%', '60%'],
       COLORS=['tab:green', 'tab:red'])


# %%
##################################################
####    multipanel figures #######################
####        1) single column    ###################
##################################################
# ALL DIMENSIONS in CM
panels = {
    'g1': {'left': 1.5, 'bottom': 4, 'width': 2., 'height': 1.4},
    'g2': {'left': 5.5, 'bottom': 4, 'width': 2., 'height': 1.4},
    'g3': {'left': 1.5, 'bottom': 1.1, 'width': 6., 'height': 1.4},
}
labels = {
    'A': {'left': 1.1, 'bottom': 5.8},
    'B': {'left': 5.1, 'bottom': 5.8},
    'C': {'left': 1.1, 'bottom': 2.9},
}


fig = pt.multipanel_figure(panels, 
                           figsize=('1-column', 6))
pt.add_labels(fig, labels)

# first panel
ax = panels['g1']['ax']
pt.scatter(np.random.randn(100), ax=ax, ms=5)
pt.scatter(np.random.randn(100), ax=ax, ms=5)
pt.set_plot(ax, xlabel='xlabel', ylabel='ylabel',
            title='sample 1')

# second panel
ax, label = panels['g2']['ax'], 'B'
pt.scatter(np.random.randn(100), ax=ax, ms=5)
pt.scatter(np.random.randn(100), ax=ax, ms=5)
pt.set_plot(ax, xlabel='xlabel', ylabel='ylabel',
            title='sample 2')

# third panel
ax, label = panels['g3']['ax'], 'C'
data = np.log(np.abs(np.random.randn(10000))+1e-5)
ax.hist(data, bins=300)
data = np.log(np.abs(np.random.randn(10000))+1e-5)
ax.hist(data, bins=300)
pt.set_plot(ax, xlabel='log-x', ylabel='count',
            yscale='log')

pt.save(fig)
# %%
##################################################
####    multipanel figures #######################
####        2) double column    ###################
##################################################
# ALL DIMENSIONS in CM
panels = {
    'g1': {'left': 1.5, 'bottom': 4, 'width': 2., 'height': 1.4},
    'g2': {'left': 5.5, 'bottom': 4, 'width': 2., 'height': 1.4},
    'g3': {'left': 9.5, 'bottom': 4, 'width': 2., 'height': 1.4},
    'g4': {'left': 13.5, 'bottom': 4, 'width': 2., 'height': 1.4},
    'g5': {'left': 1.5, 'bottom': 1.1, 'width': 15., 'height': 1.4},
}
labels = {
    'A': {'left': 1.1, 'bottom': 5.8},
    'B': {'left': 5.1, 'bottom': 5.8},
    'C': {'left': 9.1, 'bottom': 5.8},
    'D': {'left': 13.1, 'bottom': 5.8},
    'E': {'left': 1.1, 'bottom': 2.9},
}

fig = pt.multipanel_figure(panels, 
                           figsize=('2-columns', 6))
pt.add_labels(fig, labels)

# %%
