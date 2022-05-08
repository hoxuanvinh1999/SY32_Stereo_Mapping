"""
This module contains methods to optimize the block matching algorithm
"""

from typing import Callable

import numpy as np
from scipy.signal import convolve2d


def almost_equal(num_1: float, num_2: float, threshold: float = 0.0001) -> bool:
    """Check if x is approximately equal to y

    Args:
        num_1 (float): First number
        num_2 (float): second number
        threshold (float, optional): Acceptable difference of 2 numbers. Defaults to 0.0001.

    Returns:
        bool: True if first number is approximately equal to second one
    """

    return abs(num_1 - num_2) < threshold


def sub_pixel_estimation(
    disparity_and_measure_1: tuple,
    disparity_and_measure_2: tuple,
    disparity_and_measure_3: tuple,
) -> tuple:
    """Calculate sub_pixel estimation

    Args:
        disparity_and_measure_1 (tuple): first couple of disparity value and measure value
        disparity_and_measure_2 (tuple): second couple of disparity value and measure value
        disparity_and_measure_3 (tuple): third couple of disparity value and measure value

    Returns:
        tuple: sought couple of value disparity having measure minimum
    """
    disparity_1, measure_1 = disparity_and_measure_1
    disparity_2, measure_2 = disparity_and_measure_2
    disparity_3, measure_3 = disparity_and_measure_3

    if measure_1 == 0:
        return (disparity_1, measure_1)
    if measure_2 == 0:
        return (disparity_2, measure_2)
    if measure_3 == 0:
        return (disparity_3, measure_3)

    # Now if one of our disparity = 0, add epsilon into it
    # so that the division by 0 could be avoided
    epsilon = 0.001

    if disparity_1 == 0:
        disparity_1 += epsilon
    if disparity_2 == 0:
        disparity_2 += epsilon
    if disparity_3 == 0:
        disparity_3 += epsilon

    # At beginning the system of equations is
    # ax1^2 + bx1 + c = y1
    # ax2^2 + bx2 + c = y2
    # ax3^2 + bx3 + c = y3
    # Transform it into:
    # a = f(b,c)
    ## bA1 + cA2 = A3
    # bB1 + cB2 = B3
    # and calculate A1 A2 A3 B1 B2 B3

    A1 = disparity_2 - disparity_2**2 / disparity_1
    A2 = 1 - disparity_2**2 / disparity_1**2
    A3 = -measure_1 * disparity_2**2 / disparity_1**2 + measure_2
    B1 = disparity_3 - disparity_3**2 / disparity_1
    B2 = 1 - disparity_3**2 / disparity_1**2
    B3 = -measure_1 * disparity_3**2 / disparity_1**2 + measure_3

    # Calculation of a,b and c

    c = (A3 / A1 - B3 / B1) / (A2 / A1 - B2 / B1)
    b = (A3 - c * A2) / A1
    a = (measure_1 - b * disparity_1 - c) / disparity_1**2

    # Now with function y = f(x) = ax^2 + bx + c with y is the measure and x is disparity
    # calculating the extreme point of this equation

    disparity_min = -b / (2 * a)
    measure_min = c - b**2 / (4 * a)
    return (disparity_min, measure_min)


def gauss_pyramid_down(image: np.ndarray, levels=1):
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
    b = 1.0 / 4
    c = 1.0 / 4 - a / 2
    filter_arr = np.array([[c, b, a, b, c]])

    # approximate 2D Gaussian
    # filt = convolve2d(filt, filt.T)

    pyramids = [image]

    for _ in np.arange(levels):
        # zero pad the previous image for convolution
        # boarder of 2 since filter is of length 5
        p_0 = np.pad(pyramids[-1], (2,), mode="constant")

        # convolve in the x and y directions to construct p_1
        p_1 = convolve2d(p_0, filter_arr, "valid")
        p_1 = convolve2d(p_1, filter_arr.T, "valid")

        pyramids.append(p_1[::2, ::2])
        # DoG approximation of LoG

    return pyramids[-1]


def gauss_pyramid_rgb(image: np.ndarray, pyramid_method: Callable) -> np.ndarray:
    """Gauss Pyramid for RGB image

    Args:
        image (np.ndarray): input image.

    Returns:
        np.ndarray: pyramid image
    """

    width, height, depth = (
        (image.shape[0] + 1) // 2,
        (image.shape[1] + 1) // 2,
        image.shape[2],
    )
    pyramid = np.zeros((width, height, depth), np.uint8)
    for i in range(3):
        pyramid[:, :, i] = pyramid_method(image[:, :, i])
    return pyramid


def expand_image(image: np.ndarray) -> np.ndarray:
    """_summary_
    Args:
        img (tuple): the image source
    Returns:
        tuple: expanded image
    """

    n_rows, n_cols, _ = image.shape
    img_expanded = np.zeros(shape=(n_rows * 2, n_cols * 2, 3), dtype=np.uint8)
    for row in range(n_rows):
        img_expanded[row * 2 : row * 2 + 2, ::2] = image[row, :]
        img_expanded[row * 2 : row * 2 + 2, 1::2] = image[row, :]
    return img_expanded
