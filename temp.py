import matplotlib.pylab as plt
import numpy as np

from skimage import data
from scipy.signal import convolve2d

def disp_fmt_pyr(pyr):
    """
    Visualize the Gaussian pyramid
    """
    num_levels = len(pyr)

    H, W = pyr[0].shape

    img_heights = [ H * 2.0**(-i) for i in np.arange(num_levels) ]
    H = int(np.sum( img_heights ))

    out = np.zeros((H, W))

    for i in np.arange(num_levels):
        rstart = int(np.sum(img_heights[:i]))
        rend = int(rstart + img_heights[i])

        out[rstart:rend, :int(img_heights[i])] = pyr[i]

    return out


def gauss_pyr(img, levels=6):
    """
    Compute the Gaussian pyramid
    Inputs:
    - img: Input image of size (N,M)
    - levels: Number of stages for the Gaussian pyramid

    Returns:
    A tuple of levels images
    """

    # approximate length 5 Gaussian filter using binomial filter
    a = 0.4
    b = 1./4
    c = 1./4 - a/2

    filt =  np.array([[c, b, a, b, c]])

    # approximate 2D Gaussian
    # filt = convolve2d(filt, filt.T)

    pyr = [img]

    for _ in np.arange(levels):
        # zero pad the previous image for convolution
        # boarder of 2 since filter is of length 5
        p_0 = np.pad( pyr[-1], (2,), mode='constant' )

        # convolve in the x and y directions to construct p_1
        p_1 = convolve2d( p_0, filt, 'valid' )
        p_1 = convolve2d( p_1, filt.T, 'valid' )

        # DoG approximation of LoG
        pyr.append( p_1[::2,::2] )

    return pyr


camera = data.camera()
pyr = gauss_pyr(camera)
img = disp_fmt_pyr(pyr)

plt.imshow(img)
plt.savefig('gauss_pyr.png', bbox_inches="tight")

