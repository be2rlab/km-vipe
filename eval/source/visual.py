import random

import numpy as np

from matplotlib.colors import hex2color, rgb_to_hsv, CSS4_COLORS


def get_semseg_palette(num_colors, seed=100):
    semseg_colors = []
    for hex_color in CSS4_COLORS.values():
        rgb = hex2color(hex_color)
        hsv = rgb_to_hsv(rgb)
        
        if hsv[1] > 0.3 and hsv[2] > 0.3:
            semseg_colors.append(rgb)
        
    random.Random(seed).shuffle(semseg_colors)
            
    semseg_colors += semseg_colors * int(np.ceil(num_colors / len(semseg_colors)) - 1)
    
    return np.array(semseg_colors[:num_colors])