from random import gauss
import matplotlib.pylab as plt
import numpy as np
from skimage import data
from scipy.signal import convolve2d
from skimage import io
from PIL import Image
import cv2

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

    #default stages = 3

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

def upscaling(img,x,y,row,col) :

    # here image is of class 'uint8', the range of values  
    # that each colour component can have is [0 - 255]

    # create a zero matrix of order of x,y times
    # of previous image of 3-dimensions
    upscaling_img = np.zeros((x*row,y*col),np.uint8)

    i, m = 0, 0

    while m < row :

        j, n = 0, 0
        while n < col:

            # We assign pixel value from original image matrix to the
            # new upscaling image matrix in alternate rows and columns
            upscaling_img[i, j] = img[m, n]

            # increment j by y times
            j += y

            # increment n by one
            n += 1

        # increment m by one
        m += 1

        # increment i by x times
        i += x

    return upscaling_img


if __name__ == "__main__" :
    
    # read an image using imread() function of cv2
    # we have to  pass only the path of the image
    img = cv2.imread("photo/cones/im6.png")

    # displaying the image using imshow() function of cv2
    # In this : 1st argument is name of the frame
    # 2nd argument is the image matrix
    cv2.imshow('original image',img)

    # assigning number of rows, coulmns and
    # planes to the respective variables
    row,col,plane = img.shape

    # assign Blue plane of the BGR image
    # to the blue_plane variable
    blue_plane = img[:,:,0]

    # assign Green plane of the BGR image
    # to the green_plane variable
    green_plane = img[:,:,1]

    # assign Red plane of the BGR image
    # to the red_plane variable
    red_plane = img[:,:,2]

    # Upscaling the image x,y times along row and column
    x,y = 2, 2

    # here image is of class 'uint8', the range of values  
    # that each colour component can have is [0 - 255]

    # create a zero matrix of order of x,y times
    # of previous image of 3-dimensions
    upscale_img = np.zeros((x*row,y*col,plane),np.uint8)
    
    upscale_img[:,:,0] = upscaling(blue_plane, x,y,row,col)
    upscale_img[:,:,1] = upscaling(green_plane, x,y,row,col)
    upscale_img[:,:,2] = upscaling(red_plane, x,y,row,col)

    cv2.imshow('Upscale image',upscale_img)
    cv2.waitKey(0) & 0xFF