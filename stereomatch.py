#!/usr/bin/env python3

"""
Stereovision projet
"""

#%% Imports

from shutil import ExecError
import sys
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


DISPARITY_RANGE = 32  # In pixels
BLOCK_SIZE = 7


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

    print(f"cones_left: {image_left.shape}")
    print(f"cones_right: {image_right.shape}")

    n_rows, n_cols = image_left.shape
    disparity_map = np.zeros(image_left.shape, dtype=np.uint8)
    disparity_range_adjust = 255 / disparity_range

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
                        col - disparity : (col + block_size - disparity),
                    ].astype(int)

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

    for i in tqdm(range(3)):
        disparity_map[:, :, i] = disparity_block_matching(
            image_left[:, :, i], image_right[:, :, i], block_size, disparity_range
        )
    return disparity_map


if __name__ == "__main__":
    if len(sys.argv) == 3:
        output_image = "depth.png"
    elif len(sys.argv) == 4:
        output_image = sys.argv[3]
    else:
        raise ExecError("Inserter 2 ou 3 arguments")

    left_image, right_image = sys.argv[1], sys.argv[2]

    depth = disparity_block_matching_rgb(left_image, right_image)
    depth = expand_image(depth)
    plt.imsave(output_image, depth)
    depth = np.asarray(Image.open(output_image).convert("L"))
    depth = depth[:-1, :]
    Image.fromarray(depth).save(output_image)
