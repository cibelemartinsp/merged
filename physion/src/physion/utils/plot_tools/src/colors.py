from matplotlib.cm import *

import matplotlib.colors as mpl_colors

def get_linear_colormap(color1='blue', color2='red'):
    return mpl_colors.LinearSegmentedColormap.from_list('mycolors',[color1, color2])

