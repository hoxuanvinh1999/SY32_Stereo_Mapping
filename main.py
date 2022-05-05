#!/usr/bin/env python3

"""
Stereovision projet
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm

# from scipy.ndimage import filters

#%% Load images
cones_left = io.imread("photo/cones/im2.png", as_gray=True)
cones_right = io.imread("photo/cones/im6.png", as_gray=True)
print(f"cones_left: {cones_left.shape}")
print(f"cones_right: {cones_right.shape}")
# plt.subplot(1, 2, 1)
# plt.imshow(image_left, cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(image_right, cmap="gray")
# plt.show()

#%% Filter ?

# h = np.full((9, 9), 1) / (9**2)
# i_g = filters.convolve(i_g, h)
# i_d = filters.convolve(i_d, h)

SEARCH_BLOCK_RANGE = 20  # In pixels
BLOCK_SIZE = 7


def sum_of_abs_diff(pixel_vals_1: np.ndarray, pixel_vals_2: np.ndarray) -> float:
    """Calculate SAD value of 2 blocks
    Args:
        pixel_vals_1 (numpy.ndarray): pixel block from the left image
        pixel_vals_2 (numpy.ndarray): pixel block from the right image

    Returns:
        float: Sum of absolute difference between individual pixels
    """

    if pixel_vals_1.shape != pixel_vals_2.shape:
        raise AssertionError("Dimensions of two blocks are not the same!")

    return np.sum(np.abs(pixel_vals_1 - pixel_vals_2))


def compare_blocks(
    pixel_coord: tuple,
    block_left: np.ndarray,
    image_right: np.ndarray,
    block_size: int = BLOCK_SIZE,
    search_block_range: int = SEARCH_BLOCK_RANGE,
) -> tuple:
    """
    Compare left block of pixels with multiple blocks from the right image using SEARCH_BLOCK_SIZE
    to constrain the search in the right image.

    Args:
        pixel_coord(tuple): coordinate of the pixel
        block_left (numpy.ndarray): containing pixel values within the block selected from the
                                    left image
        image_right (numpy.ndarray]): containing pixel values for the entrire right image
        block_size (int, optional): Block of pixels width and height. Default to BLOCK_SIZE.
        search_block_range (int, optional): Number of pixels to search.
                                            Default to SEARCH_BLOCK_RANGE.

    Returns:
        tuple: (row_index, col_index) row and column index of the best matching block
                                      in the right image.
    """

    row_index, col_index = pixel_coord

    # Get search range for the right image
    col_min = max(0, col_index - search_block_range)
    col_max = min(image_right.shape[1], col_index + search_block_range)

    # print(f'search bounding box: ({y, x_min}, ({y, x_max}))')
    min_sad, sought_coord = None, None
    for col_index in range(col_min, col_max):
        block_right = image_right[
            row_index : row_index + block_size, col_index : col_index + block_size
        ]
        sad = sum_of_abs_diff(block_left, block_right)
        # print(f'sad: {sad}, {y, x}')
        if not sought_coord or sad < min_sad:
            min_sad = sad
            sought_coord = (row_index, col_index)

    return sought_coord


#%% Disparity computation


def disparity_block_matching(
    image_left: np.ndarray,
    image_right: np.ndarray,
    block_size: int = BLOCK_SIZE,
    search_block_range: int = SEARCH_BLOCK_RANGE,
) -> np.ndarray:
    """Calculate the disparity map of two images

    Args:
        pixel_vals_1 (np.ndarray): Pixel block from the first image
        pixel_vals_2 (np.ndarray): Pixel block from the second image
        block_size (int, optional): Size of the matched block. Defaults to BLOCK_SIZE.
        search_block_range (int, optional): Number of pixels to search.
                                            Defaults to SEARCH_BLOCK_RANGE.

    Returns:
        np.ndarray: The disparity map after applying the block matching algorithm
    """
    cols, rows = image_left.shape
    disparity_map = np.zeros((rows, cols))
    for row in tqdm(range(block_size, rows - block_size)):
        for col in range(block_size, cols - block_size):
            block_left = image_left[row : row + block_size, col : col + block_size]
            min_index = compare_blocks(
                (row, col), block_left, image_right, block_size, search_block_range
            )
            disparity_map[row, col] = abs(min_index[1] - col)

    return disparity_map


test = disparity_block_matching(cones_right, cones_left, BLOCK_SIZE, SEARCH_BLOCK_RANGE)
# test2 = disparity(i_g, i_d, d_est, nn)
# plt.subplot(1, 2, 1)
plt.imshow(test[:, SEARCH_BLOCK_RANGE:], cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(test2[:, d_est:], cmap='gray')
plt.title(f"N = {BLOCK_SIZE}")
plt.show()
