import os, tempfile
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle

def mm2inch(x):
    return x/25.4
def inch2mm(x):
    return x*25.4

# DEFAULT AXES PROPS
AX_PROPS = {
    'ax_size':(20., 14.), # mm
    'hspace_size':9., # mm
    'wspace_size':10., # mm
    'left_size':13., # mm
    'right_size':4., # mm
    'top_size':5., # mm
    'bottom_size':10., # mm
}

def from_grid_to_subplots_args(Nx, Ny, ax_scale,
                       left, right,
                       bottom, top,
                       wspace, hspace):
    """
    calculate the dimension quantities required by *matplotlib* plt.figure object
    """
    dimension = {}

    # horizontal
    dimension['full_width'] = left*ax_scale[0]*AX_PROPS['left_size']+\
        right*ax_scale[0]*AX_PROPS['right_size']+\
        Nx*ax_scale[0]*ax_scale[0]*AX_PROPS['ax_size'][0]+\
        wspace*(Nx-1)*ax_scale[0]*AX_PROPS['wspace_size']
    dimension['left'] = left*ax_scale[0]*AX_PROPS['left_size']/dimension['full_width']
    dimension['right'] = right*ax_scale[0]*AX_PROPS['right_size']/dimension['full_width']
    dimension['wspace'] = wspace*ax_scale[0]*AX_PROPS['wspace_size']/ax_scale[0]/AX_PROPS['ax_size'][0]

    # vertical
    dimension['full_height'] = bottom*ax_scale[1]*AX_PROPS['bottom_size']+\
        top*ax_scale[1]*AX_PROPS['top_size']+\
        Ny*ax_scale[1]*AX_PROPS['ax_size'][1]+\
        hspace*ax_scale[1]*(Ny-1)*AX_PROPS['hspace_size']
    dimension['bottom'] = bottom*ax_scale[1]*AX_PROPS['bottom_size']/dimension['full_height']
    dimension['top'] = top*ax_scale[1]*AX_PROPS['top_size']/dimension['full_height']
    dimension['hspace'] = hspace*ax_scale[1]*AX_PROPS['hspace_size']/ax_scale[1]/AX_PROPS['ax_size'][1]

    return dimension


def from_figsize_to_subplots_args(figsize, Nx, Ny,
                       left, right,
                       bottom, top,
                       wspace, hspace):
    """
    calculate the dimension quantities required by *matplotlib* plt.figure object
    """
    dimension = {}

    # horizontal
    dimension['full_width'] = inch2mm(figsize[0])
    dimension['left'] = left*AX_PROPS['left_size']/dimension['full_width']
    dimension['right'] = right*AX_PROPS['right_size']/dimension['full_width']
    dimension['wspace'] = wspace*AX_PROPS['wspace_size']/AX_PROPS['ax_size'][0]

    # vertical
    dimension['full_height'] = inch2mm(figsize[1]) 
    dimension['bottom'] = bottom*AX_PROPS['bottom_size']/dimension['full_height']
    dimension['top'] = top*AX_PROPS['top_size']/dimension['full_height']
    dimension['hspace'] = hspace*AX_PROPS['hspace_size']/AX_PROPS['ax_size'][1]

    return dimension

def figure(axes = (1,1),
           axes_extents=None,
           grid=None,
           figsize=(1.,1.),
           ax_scale=(1.,1.),
           page=None, 
           left=1., right=1.,
           bottom=1., top=1.,
           wspace=1., hspace=1.,
           dpi=None,
           reshape_axes=True):

    """
    build a figure of axes with constant dimensions (see AX_PROPS above)

    the wspace, hspace, ... values are factor that modulates the default values
    -> then use >1 to make bigger, and <1 to make smaller...

    Subplots are build with this convention for the geometry:
    (X,Y)
    ------ X -------------->
    |                 |     |
    |      (3,1)      |(1,1)|
    |                 |     |
    |-----------------------|
    Y     |           |     |
    |(1,1)|   (2,1)   |(1,1)|
    |     |           |     |
    |------------------------
    |
    v

    TO PRODUCE THIS, RUN:
    figure(axes_extents=[\
                         [[3,1], [1,1] ],
                         [[1,1], [2,1], [1,1] ] ] )
    show()


    OTHERWISE, you can use the "grid" arguments that corresponds to "subplot2grid"
    TO PRODUCE THIS, RUN:
    figure(grid=[(0,0,1,4),
                 (x,y,dx,dy)])

    """

    AX = []

    if grid is not None:
        Nx = np.max([g[0]+g[2] for g in grid])
        Ny = np.max([g[1]+g[3] for g in grid])

        dim =  from_grid_to_subplots_args(Nx, Ny, ax_scale,
                            left, right, bottom, top, wspace, hspace)

    else:
        if axes_extents is not None:
            if type(axes_extents) is tuple:
                axes_extents = [[axes_extents]]
        else:
            axes_extents = [[[1,1] for j in range(axes[0])]\
                            for i in range(axes[1])]

        Nx = np.sum([axes_extents[0][j][0] \
                          for j in range(len(axes_extents[0]))])
        Ny = np.sum([axes_extents[i][0][1] \
                          for i in range(len(axes_extents))])

        dim =  from_grid_to_subplots_args(Nx, Ny, ax_scale,
                            left, right, bottom, top, wspace, hspace)

    if page=='A4':

        fig = plt.figure(figsize=(8.27, 11.69), dpi=dpi)
        plt.subplots_adjust(left=left*fig.subplotpars.left,
                            bottom=bottom*fig.subplotpars.bottom,
                            top=1-(top*(1-fig.subplotpars.top)),
                            right=1-(right*(1-fig.subplotpars.right)),
                            wspace=wspace*fig.subplotpars.wspace,
                            hspace=hspace*fig.subplotpars.hspace)

        fig = plt.figure(figsize=figsize,
                         dpi=dpi)

    else:

        fig = plt.figure(figsize=(mm2inch(dim['full_width']),
                                  mm2inch(dim['full_height'])),
                         dpi=dpi)

    if grid is not None:
        for g in grid:
            ax = plt.subplot2grid((Ny, Nx),
                                  (g[1], g[0]),
                                  colspan=g[2],
                                  rowspan=g[3])
            AX.append(ax)
    else:

        j0_row = 0
        for j in range(len(axes_extents)):
            AX_line = []
            i0_line = 0
            for i in range(len(axes_extents[j])):
                AX_line.append(plt.subplot2grid(\
                                                (Ny, Nx),
                                                (j0_row, i0_line),\
                                                colspan=axes_extents[j][i][0],
                                                rowspan=axes_extents[j][i][1]))
                i0_line += axes_extents[j][i][0]
            j0_row += axes_extents[j][i][1]
            AX.append(AX_line)

    if dim['left']>=(1-dim['right']):
        print('left=%.2f and right=%.2f leads to a too large space' % (dim['left'], dim['right']),
              'set to 0.2, & 0.95 respectively')
        dim['left'], dim['right'] = 0.2, 0.95
    if dim['bottom']>=(1-dim['top']):
        print('left=%.2f and right=%.2f leads to a too large space' % (dim['bottom'], dim['top']),
              'set to 0.2, & 0.95 respectively')
        dim['bottom'], dim['top'] = 0.2, 0.95


    # # Subplots placements adjustements
    plt.subplots_adjust(left=dim['left'],
                        bottom=dim['bottom'],
                        top=1.-dim['top'],
                        right=1.-dim['right'],
                        wspace=dim['wspace'],
                        hspace=dim['hspace'])

    if (grid is not None) or (reshape_axes is False):
        return fig, AX
    elif len(AX)==1 and (len(AX[0])==1):
        return fig, AX[0][0]
    elif (len(AX[0])==1) and (len(AX[-1])==1):
        return fig, [AX[i][0] for i in range(len(AX))]
    elif len(AX)==1:
        return fig, AX[0]
    else:
        return fig, AX

def get_width(size='1-column'):
    """
    some standard sizes for manuscript figures
    return in cm  
    taking the Cell Press size requirements:
    https://www.cell.com/information-for-authors/figure-guidelines
    """
    if size == '1-column':
        return 8.5
    elif size == '1.5-column':
        return 11.4
    elif '2-column' in size: # handle "2-columns" also
        return 17.4
    else:
        print()
        print('  Figure Size not Recognized !!!')
        print('         possibilities are:')
        print('           "1-column", "1.5-column", "2-columns" ')
        print()



def multipanel_figure(panels, 
                      figsize=('1-column', 5),
                      borders=True):
    """
    Create a multi-panel figure using physical dimensions (cm).

    Parameters
    ----------
    panels : dict
        Each entry should be a dict containing:
            - 'left'   : float (cm from left)
            - 'bottom' : float (cm from bottom)
            - 'width'  : float (cm)
            - 'height' : float (cm)


    figsize_cm : tuple
        Figure size in cm (width, height)

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict or list
    """

    if type(figsize[0])==str:
        figsize = (get_width(figsize[0]), # mm to cm
                   figsize[1])

    # Convert cm → inches
    cm_to_inch = 1 / 2.54
    figsize_inch = (figsize[0] * cm_to_inch,
                    figsize[1] * cm_to_inch)

    fig = plt.figure(figsize=figsize_inch)

    fig_w_cm, fig_h_cm = figsize

    for key, panel in panels.items():
        # Convert cm → normalized [0,1]
        left   = panel['left']   / fig_w_cm
        bottom = panel['bottom'] / fig_h_cm
        width  = panel['width']  / fig_w_cm
        height = panel['height'] / fig_h_cm

        ax = fig.add_axes([left, bottom, width, height])

        panel['ax'] = ax

    if borders:
        # draw a rectangle around the figure
        rect = Rectangle(
            (0, 0), 1, 1,
            transform=fig.transFigure,
            fill=False,
            edgecolor='black',
            linewidth=0.8  # thin line
        )
        fig.patches.append(rect)

    return fig

def save(fig, 
         on='Desktop', 
         fig_name='fig.svg',
         transparent=True,
         dpi='figure'):
    """
    - on:
        default from $HOME directory, e.g. on='Desktop/temp', saves as '~/Desktop/temp/fig.svg'
        otherwise relative path, e.g. on='./temp', saves as './temp/fig.svg'
    """
    
    if on=='temp':
        path = os.path.join(tempfile.tempdir, fig_name)
    elif on[0]=='.':
        path = os.path.join(*os.path.split(on), fig_name)
    else:
        path = os.path.join(os.path.expanduser('~'), 
                            *os.path.split(on), fig_name)
  
    fig.savefig(path, transparent=transparent, dpi=dpi)

def show():
    plt.show()

if __name__=='__main__':

    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    import plot_tools as pt

    pt.set_style('dark-notebook')

    fig, AX = pt.figure(axes=(4,10), 
                        hspace=0.2, wspace=0.2, page='A4')

    pt.show()
    
    # from datavyz import graph_env_manuscript as ge
    # import itertools, string

    # from datavyz.main import graph_env
    # ge = graph_env('manuscript')

    # # fig, ax = ge.figure()
    # fig1, AX1 = ge.figure(axes=(2,2))
    # for l, ax in zip(list(string.ascii_lowercase), itertools.chain(*AX1)):
        # ge.top_left_letter(ax, l+'     ')
        # ge.set_plot(ax, xlabel='xlabel (xunit)', ylabel='ylabel (yunit)', grid=True)
    # fig1.savefig('fig1.svg')

    # # fig2, AX2 = ge.figure(axes_extents=[\
    # #                                     [[1,1], [1,1], [1,1]],
    # #                                     [[2,2], [1,2]]])

    # fig2, AX2 = ge.figure(axes_extents=[\
                                        # [[1,1], [3,1]],
                                        # [[4,1]],
                                        # [[1,1], [2,1], [1,1] ] ],
                          # figsize=[.95,.95])

    # t = np.linspace(0, 10, 1e3)
    # y = np.cos(5*t)+np.random.randn(len(t))

    # # leave first axis empty for drawing
    # AX2[0][0].axis('off') # space for docs/schematic.svg

    # # time series plot
    # AX2[0][1].plot(t, y)
    # ge.set_plot(AX2[0][1], xlabel='xlabel (xunit)', ylabel='ylabel (yunit)')

    # # more time series plot
    # AX2[1][0].plot(t[t>9], y[t>9], label='raw')
    # AX2[1][0].plot(t[t>9][1:], np.diff(y[t>9]), label='deriv.')
    # AX2[1][0].plot(t[t>9][1:-1], np.diff(np.diff(y[t>9])), label='2nd deriv.')
    # ge.set_plot(AX2[1][0], xlabel='xlabel (xunit)', ylabel='ylabel (yunit)')

    # # histogram
    # ge.scatter(t[::10], t[::10]+np.random.randn(100),
               # ax=AX2[2][0], xlabel='ylabel (yunit)')


    # # histogram
    # ge.bar(np.random.randn(8),
           # COLORS=[ge.viridis(i/7) for i in range(8)],
            # ax=AX2[2][1], xlabel='ylabel (yunit)')

    # # pie plot
    # ge.pie([0.25,0.4,0.35], ax=AX2[2][2], ext_labels=['Set 1', 'Set 2', 'Set 3'])


    # # looping on all plots to add the top left letter:
    # for i, fig, AX in zip(range(3), [fig1, fig2], [AX1, AX2]):
        # for l, ax in zip(list(string.ascii_lowercase), itertools.chain(*AX)):
            # ge.top_left_letter(ax, l+'     ')

    # # saving the figure with all plots
    # fig2.savefig('fig2.svg')

    # # generating the figure with the addition of the drawing
    # from datavyz.plot_export import put_list_of_figs_to_svg_fig
    # put_list_of_figs_to_svg_fig(['docs/schematic.svg', fig],
                                # fig_name='fig.svg',
                                # Props={'XCOORD':[0,0], 'YCOORD':[0,0]})

    # ge.show()

