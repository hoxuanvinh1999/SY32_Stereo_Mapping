#!/usr/bin/env python3

"""
Stereovision projet
"""

#%% Imports

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from optimize_methods import (
    almost_equal,
    expand_image,
    gauss_pyramid_down,
    sub_pixel_estimation,
    gauss_pyramid_rgb,
)

# from scipy.ndimage import filters

#%% Filter ?

# h = np.full((9, 9), 1) / (9**2)
# i_g = filters.convolve(i_g, h)
# i_d = filters.convolve(i_d, h)

DISPARITY_RANGE = 32  # In pixels
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
        raise AssertionError(
            (
                "Dimensions of two blocks are not the same!"
                f"{pixel_vals_1.shape} != {pixel_vals_2.shape}."
            )
        )

    return np.sum(np.square(pixel_vals_1 - pixel_vals_2))


def compare_blocks(
    pixel: tuple,
    block_left: np.ndarray,
    image_right: np.ndarray,
    block_size: int = BLOCK_SIZE,
    search_block_range: int = DISPARITY_RANGE,
) -> tuple:
    """
    Compare left block of pixels with multiple blocks from the right image using SEARCH_BLOCK_SIZE
    to constrain the search in the right image.

    Args:
        pixel(tuple): coordinate of the pixel
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

    n_rows, n_cols = pixel
    search_block_range_adjust = 255 / search_block_range

    # Get search range for the right image
    col_min = max(0, n_cols - search_block_range)
    col_max = min(image_right.shape[1] - block_size, n_cols + search_block_range)

    # TODO: Might need to add edge cases for current pixel
    min_sad = float("inf")
    # list_sad = [0] * image_right.shape[1]
    col_sought_pixel = None
    for n_cols in range(col_min, col_max):
        # right_row_index = max(0, min(image_right.shape[1], row_index + block_size))
        # right_col_index = max(0, min(image_right.shape[0], col_index + block_size))

        block_right = image_right[
            n_rows : n_rows + block_size, n_cols : n_cols + block_size
        ].astype(int)
        # print(f"{row} {col} {block_right.shape}")
        sad = sum_of_abs_diff(block_left, block_right)
        # list_sad[n_cols] = sad
        if sad < min_sad:
            col_sought_pixel = n_cols
            min_sad = sad

    # if (
    #     almost_equal(min_sad, 0)
    #     or col_sought_pixel == 0
    #     or col_sought_pixel == image_right.shape[1] - 1
    # ):
    #     return abs(col_sought_pixel - pixel[1])

    # couples = (
    #     (col_sought_pixel + i - pixel[1], list_sad[col_sought_pixel + i])
    #     for i in [-1, 0, 1]
    # )
    # left_couple = (sought_coord[1] - 1 - col_index, list_sad[sought_coord[1] - 1])
    # current_couple = (sought_coord[1] - col_index, list_sad[sought_coord[1]])
    # right_couple = (sought_coord[1] + 1 - col_index, list_sad[sought_coord[1] + 1])
    # return abs(np.round(sub_pixel_estimation(*couples)[0]))
    # return abs(col_sought_pixel - 1 / 2 * (C[2] - C[0]) / (C[0] - 2 * C[1] + C[2]))

    return (col_sought_pixel - pixel[1]) * search_block_range_adjust


#%% Disparity computation


def disparity_block_matching(
    image_left: np.ndarray,
    image_right: np.ndarray,
    block_size: int = BLOCK_SIZE,
    disparity_range: int = DISPARITY_RANGE,
) -> np.ndarray:

    """Calculate the disparity map of two images

    Args:
        pixel_vals_1 (np.ndarray): Pixel block from the first image
        pixel_vals_2 (np.ndarray): Pixel block from the second image
        block_size (int, optional): Size of the matched block. Defaults to BLOCK_SIZE.
        disparity_range (int, optional): Number of pixels to search (should be power of 2).
                                            Defaults to DISPARITY_RANGE.

    Returns:
        np.ndarray: The disparity map after applying the block matching algorithm
    """
    # image_left = gauss_pyramid_rgb(
    #     np.asarray(Image.open(image_left)), gauss_pyramid_down
    # )
    # image_right = gauss_pyramid_rgb(
    #     np.asarray(Image.open(image_right)), gauss_pyramid_down
    # )
    # image_left = np.asarray(Image.open(image_left_path).convert("L"))
    # image_right = np.asarray(Image.open(image_right_path).convert("L"))
    print(f"cones_left: {image_left.shape}")
    print(f"cones_right: {image_right.shape}")

    n_rows, n_cols = image_left.shape
    disparity_map = np.zeros(image_left.shape, dtype=np.uint8)
    disparity_range_adjust = 255 / disparity_range

    # dp_image_left = dynamic_programming(image_left, block_size, np.square)
    # dp_image_right = dynamic_programming(image_right, block_size, np.square)

    for row in tqdm(range(n_rows - block_size)):
        for col in range(n_cols - block_size):
            block_left = image_left[
                row : row + block_size, col : col + block_size
            ].astype(int)

            sought_disparity = disparity_range
            min_sad = 10**12
            list_sad = np.zeros(shape=(disparity_range + 1))
            for disparity in range(disparity_range):
                if col + block_size - disparity >= 0 and col - disparity >= 0:
                    block_right = image_right[
                        row : row + block_size,
                        col - disparity : (col + block_size - disparity)
                    ].astype(int)

                    # sad = np.sum(
                    #     np.sqrt(
                    #         np.sum(np.square(block_left - block_right), axis=(2, 1))
                    #     )
                    # )
                    sad = np.sum(np.square(block_left - block_right))
                    list_sad[disparity] = sad
                    # print(f"sad: {sad}; min_sad: {min_sad}, disparity: {disparity}")
                    if sad < min_sad:
                        min_sad = sad
                        sought_disparity = disparity

            # print(f"sought_disparity: {sought_disparity}")
            if not (
                almost_equal(min_sad, 0)
                or sought_disparity == 0
                or sought_disparity == disparity_range - 1
            ):
                couples = (
                    (sought_disparity + i, list_sad[sought_disparity + i])
                    for i in [-1, 0, 1]
                )
                sought_disparity = sub_pixel_estimation(*couples)[0]

            disparity_map[row, col] = sought_disparity * disparity_range_adjust

    return disparity_map


def disparity_block_matching_rgb(
    image_left_path: str,
    image_right_path: str,
    block_size: int = BLOCK_SIZE,
    disparity_range: int = DISPARITY_RANGE,
) -> np.ndarray:
    """_summary_

    Args:
        image_left_path (str): _description_
        image_right_path (str): _description_
        block_size (int, optional): _description_. Defaults to BLOCK_SIZE.
        disparity_range (int, optional): _description_. Defaults to DISPARITY_RANGE.

    Returns:
        np.ndarray: _description_
    """
    image_left = gauss_pyramid_rgb(
        np.asarray(Image.open(image_left_path)), gauss_pyramid_down
    )
    image_right = gauss_pyramid_rgb(
        np.asarray(Image.open(image_right_path)), gauss_pyramid_down
    )

    disparity_map = np.full(image_left.shape, 255, dtype=np.uint8)

    for i in range(3):
        disparity_map[:, :, i] = disparity_block_matching(
            image_left[:, :, i], image_right[:, :, i], block_size, disparity_range
        )
    return disparity_map


if __name__ == "__main__":
    # depth = disparity_block_matching("photo/cones/im2.png", "photo/cones/im6.png")
    depth = disparity_block_matching_rgb("photo/cones/im2.png", "photo/cones/im6.png")
    # depth = gauss_pyramid_up(depth)
    depth = expand_image(depth)
    plt.imsave("photo/cones/depth.png", depth)
    depth = np.asarray(Image.open("photo/cones/depth.png").convert("L"))
    depth = depth[:-1, :]
    print(depth.shape)
    Image.fromarray(depth).save("photo/cones/depth_gr.png")
    # Image.fromarray(depth).save("photo/cones/depth.png")
