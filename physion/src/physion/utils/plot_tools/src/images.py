import numpy as np
from skimage import color, io
from matplotlib.cm import binary_r
import matplotlib.image as mpimg
from .figure import figure


# # image plot
def image(X, 
          cmap=binary_r, 
          alpha=1., 
          ax=None, 
          fig_args=dict(ax_scale=(2,3), 
                        left=0.1, bottom=0.1, top=0.2, right=0.2),
          title=''):

    if ax is None:
        fig, ax = figure(**fig_args)
    else:
        fig = None 

    if type(X) in [str]:
        X = load_image(X)

    ax.imshow(X.T, 
            cmap=cmap, 
            alpha=alpha,
            interpolation=None,
            origin='lower',
            aspect='equal')

    ax.axis('off')

    if title!='':
        ax.set_title(title)

    return fig, ax

def load_image(image_path):

    img = color.rgb2gray(io.imread(image_path))
    
    return np.rot90(np.array(img), k=3) # needs rotation


if __name__=='__main__':

    import matplotlib.pylab as plt
    import sys
    image(sys.argv[-1])
    plt.show()

