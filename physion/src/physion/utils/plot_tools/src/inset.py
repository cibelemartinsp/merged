"""

"""
import numpy as np
import matplotlib as mpl

def inset(stuff,
          rect=[.5,.5,.5,.4],
          facecolor=None):
    """
    the argument can be either a figure or an axes instance
    """

    if type(stuff)==mpl.figure.Figure: # if figure, no choice --> relative coordinates
        subax = stuff.add_axes(rect,facecolor=facecolor)
    else:

        fig = mpl.pyplot.gcf()
        box = stuff.get_position()
        width = box.width
        height = box.height
        inax_position  = stuff.transAxes.transform([rect[0], rect[1]])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)    
        x = infig_position[0]
        y = infig_position[1]
        width *= rect[2]
        height *= rect[3]
        subax = fig.add_axes([x,y,width,height],facecolor=facecolor)
        
    return subax

if __name__=='__main__':

    import matplotlib.pylab as plt

    fig, ax = plt.subplots(1)

    y = np.exp(np.random.randn(100))
    
    sax = inset(ax, [.5, .8, .5, .4])
    sax2 = inset(fig, [0.1, 0.1, .3, .2])
    sax.hist(y, bins=10)
    sax2.hist(y, bins=10)
    plt.show()
