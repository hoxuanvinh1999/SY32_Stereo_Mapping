#!/usr/bin/env python3

"""
Stereovision projet
"""

import sys
import time
import numpy as np
from PIL import Image
from evaldisp import *


def stereo_match(
    left_img: str, right_img: str, kernel: int, max_disparity: int, formula
) -> np.ndarray:
    """Block matching algorithm

    Args:
        left_img (str): Left image
        right_img (str): Right image
        kernel (int): Kernel size
        max_disparity (int): disparity range
        formula (_type_): Measure to calculate cost of 2 blocks (SAD, SSD, etc)

    Returns:
        np.ndarray: Disparity map
    """

    # Load in both images, assumed to be RGBA 8bit per channel images
    left_img = Image.open(left_img).convert("L")
    left = np.asarray(left_img)
    right_img = Image.open(right_img).convert("L")
    right = np.asarray(right_img)
    width, height = left_img.size  # assume that both images are same size

    # Depth (or disparity) map
    depth_map = np.zeros((height, width), np.uint8)
    kernel_half = kernel // 2
    disparity_adjust = 255 / max_disparity

    for y in range(height - kernel):
        print(
            f"\rProcessing.. {(y / (height - kernel_half) * 100):.0f}% complete",
            end="",
            flush=True,
        )

        for x in range(width - kernel):
            best_disparity = 0
            prev_mes = 65534

            for disparity in range(max_disparity):
                m_left = left[y : y + kernel, x : x + kernel].astype(int)

                if x + kernel - disparity >= 0 and x - disparity >= 0:
                    m_right = right[
                        y : y + kernel, x - disparity : (x + kernel - disparity)
                    ].astype(int)

                mes = np.sum(np.square(m_left - m_right))

                if mes < prev_mes:
                    prev_mes = mes
                    best_disparity = disparity

            # set depth output for this x,y location to the best match
            depth_map[y, x] = best_disparity * disparity_adjust

    return depth_map


if __name__ == "__main__":
    start_time = time.time()
    depth = stereo_match(
        sys.argv[1], sys.argv[2], 11, 64, "ssd"
    )  # 11x11 local search kernel, 64 pixel search range

    print(f"--- {time.time() - start_time} seconds ---")

    if len(sys.argv) == 3:
        Image.fromarray(depth).save("disparites_estimees_ssd_cones.png")
        main("cones/disp2.png", "cones/occl.png", "disparites_estimees_ssd_cones.png")
    elif len(sys.argv) == 4:
        # Convert to PIL and save it
        Image.fromarray(depth).save(sys.argv[3])
