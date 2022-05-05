from optimize_methods import sub_pixel_estimation


def almost_equal(x, y, threshold=0.0001):
    return abs(x - y) < threshold


def test_simple_cases():
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