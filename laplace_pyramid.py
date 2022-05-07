from random import gauss
import matplotlib.pylab as plt
import numpy as np
from skimage import data
from scipy.signal import convolve2d
from skimage import io
from PIL import Image
from urllib3 import Retry


def disp_fmt_pyr(pyr, laplace=True) -> np.ndarray:
    """
    Visualize the Laplacian pyramid
    """
    num_levels = len(pyr)
    num_stages = num_levels / 2

    H, W = pyr[0].shape

    img_heights = [H * 2.0 ** (-i) for i in np.arange(num_stages)]
    H = np.int(np.sum(img_heights))

    out = np.zeros((H, W * 2))

    for i in np.arange(num_stages):
        rstart = np.int(np.sum(img_heights[:i]))
        rend = np.int(rstart + img_heights[i])

        out[rstart:rend, : np.int(img_heights[i] * 2)] = np.hstack(
            (pyr[i * 2], pyr[i * 2 + 1])
        )

    return out


def laplace_pyr(img, stages) -> tuple:
    """
    Compute the Laplacian pyramid

    Inputs:
    - img: Input image of size (N,M)
    - stages: Number of stages for the Laplacian pyramid

    Returns:
    A tuple of stages*2 images
    """

    # approximate length 5 Gaussian filter using binomial filter
    filt = 1.0 / 16 * np.array([[1, 4, 6, 4, 1]])
    filt2 = np.pad(filt, ((0, 0), (2, 2)), mode="constant")
    filt2 = convolve2d(filt2, filt, "valid")

    # approximate 2D Gaussian
    # filt = convolve2d(filt, filt.T)

    pyr = []

    # default stages = 3

    old_img = img
    for i in np.arange(stages):
        # zero pad the previous image for convolution
        # boarder of 2 since filter is of length 5
        p_0 = np.pad(old_img, (2,), mode="constant")

        # convolve in the x and y directions to construct p_1
        p_1 = convolve2d(p_0, filt, "valid")
        p_1 = convolve2d(p_1, filt.T, "valid")

        # DoG approximation of LoG
        pyr.append(p_1 - p_0[2:-2, 2:-2])

        # convolve with scaled gaussian \sigma_2 = \sqrt(2)\sigma_1
        # this is implemented by cascaded convolution
        p_1 = np.pad(p_1, (2,), mode="constant")
        p_2 = convolve2d(p_1, filt2, "valid")
        p_2 = convolve2d(p_2, filt2.T, "valid")

        # DoG approximation of LoG
        pyr.append(p_2 - p_1[2:-2, 2:-2])

        # subsample p_2 for next stage
        old_img = p_2[::2, ::2]

    return pyr[-1]


# if __name__ == "__main__":
# io_img = io.imread("photo/cones/im2.png")
# img_expanded = expand(io_img)
# print(io_img.shape, img_expanded.shape)
# io.imsave("temp.png", img_expanded)
