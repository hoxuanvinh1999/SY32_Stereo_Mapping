def sub_pixel_estimation(
    disparity_and_sad_1: tuple, disparity_and_sad_2: tuple, disparity_and_sad_3: tuple
) -> tuple:
    """Calculate sub_pixel estimation

    Args:
        disparity_and_sad_1 (tuple): a couple of disparity value and SAD value
        disparity_and_sad_2 (tuple): a couple of disparity value and SAD value
        disparity_and_sad_3 (tuple): a couple of disparity value and SAD value

    Returns:
        tuple: a couple of value disparity minimum and value SAD minimum
    """
    disparity_1, sad_1 = disparity_and_sad_1
    disparity_2, sad_2 = disparity_and_sad_2
    disparity_3, sad_3 = disparity_and_sad_3

    # The function returns sad if sad = 0

    if sad_1 == 0:
        return (disparity_1, sad_1)
    elif sad_2 == 0:
        return (disparity_2, sad_2)
    elif sad_3 == 0:
        return (disparity_3, sad_3)

    # Now if one of our disparity = 0, add a little value into it
    # so that the division by 0 could be avoided

    if disparity_1 == 0:
        disparity_1 += 0.01
    if disparity_2 == 0:
        disparity_2 += 0.01
    if disparity_3 == 0:
        disparity_3 += 0.01

    # At beginning the system of equations is
    # ax1^2 + bx1 + c = y1
    # ax2^2 + bx2 + c = y2
    # ax3^2 + bx3 + c = y3
    # Transform it into:
    # a = f(b,c)
    ## bA1 + cA2 = A3
    # bB1 + cB2 = B3
    # and calculate A1 A2 A3 B1 B2 B3

    A1 = disparity_2 - disparity_2 * disparity_2 / disparity_1
    A2 = 1 - (disparity_2 * disparity_2) / (disparity_1 * disparity_1)
    A3 = -sad_1 * (disparity_2 * disparity_2) / (disparity_1 * disparity_1) + sad_2
    B1 = disparity_3 - disparity_3 * disparity_3 / disparity_1
    B2 = 1 - (disparity_3 * disparity_3) / (disparity_1 * disparity_1)
    B3 = -sad_1 * (disparity_3 * disparity_3) / (disparity_1 * disparity_1) + sad_3

    # Calculation of a,b and c

    c = (A3 / A1 - B3 / B1) / (A2 / A1 - B2 / B1)
    b = (A3 - c * A2) / A1
    a = (sad_1 - b * disparity_1 - c) / (disparity_1 * disparity_1)

    # Now with function y = f(x) = ax^2 + bx + c with y is sad and x is disparity
    # calculating the extreme point of this equation

    disparity_min = -b / (2 * a)
    sad_min = c - b * b / (4 * a)
    return (disparity_min, sad_min)
