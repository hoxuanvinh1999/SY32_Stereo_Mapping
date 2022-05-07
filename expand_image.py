from random import gauss
import matplotlib.pylab as plt
import numpy as np
from skimage import data
from scipy.signal import convolve2d
from skimage import io
from PIL import Image
from urllib3 import Retry

def expand(img: tuple) -> tuple:
    """_summary_

    Args:
        img (tuple): the image source

    Returns:
        tuple: expanded image
    """    
    height, width, plane = img.shape
    img_expanded = []
    for i in range(height):
        new_line = []
        for j in range(width):
            new_line.append(img[i, j])
            new_line.append(img[i, j])
        img_expanded.append(new_line)
        img_expanded.append(new_line)
    return np.asarray(img_expanded)