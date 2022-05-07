"""
Testing module
"""

from optimize_methods import almost_equal, sub_pixel_estimation, dynamic_programming
import numpy as np


def test_spe():
    """
    Testing sub_pixel_estimation method
    """
    # y = x^2
    couple = sub_pixel_estimation((1, 1), (2, 4), (3, 9))
    assert couple == (0, 0)

    # y = x^2 + x + 1
    couple = sub_pixel_estimation((1, 3), (2, 7), (3, 13))
    assert almost_equal(couple[0], -0.5)
    assert almost_equal(couple[1], 0.75)


def test_dp():
    """Testing dynamic programming"""
    image = np.arange(1, 13).reshape((3, 4))
    image_abs = np.array([[14, 18, 22, 0], [30, 34, 38, 0], [0, 0, 0, 0]], np.uint32)
    image_square = np.array([[66, 98, 138, 0], [242, 306, 378, 0], [0, 0, 0, 0]], np.uint32)
    assert np.array_equal(dynamic_programming(image, 2, np.abs), image_abs)
    assert np.array_equal(dynamic_programming(image, 2, np.square), image_square)
