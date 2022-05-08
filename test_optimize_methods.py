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
