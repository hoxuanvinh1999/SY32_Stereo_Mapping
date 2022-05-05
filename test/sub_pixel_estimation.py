def sub_pixel_estimation(disparity_1,disparity_2,disparity_3,sad_1,sad_2,sad_3):
    #First of all, if we have a sad = 0, we will return it
    if(sad_1 == 0):
        return(disparity_1,sad_1)
    elif(sad_2 == 0):
        return(disparity_2,sad_2)
    elif(sad_3 == 0):
        return(disparity_3,sad_3)
    #Now if one of our disparity = 0, we will add a little value into it
    #so that we can avoid the division by 0
    if(disparity_1 == 0):
        disparity_1 += 0.01
    if(disparity_2 == 0):
        disparity_2 += 0.01
    if(disparity_3 == 0):
        disparity_3 += 0.01
    # First our system of equations is
    # ax1^2 + bx1 + c = y1
    # ax2^2 + bx2 + c = y2
    # ax3^2 + bx3 + c = y3
    # I will transform it into:
    # a = f(b,c)
    # bA1 + cA2 = A3
    # bB1 + cB2 = B3
    # and calculate A1 A2 A3 B1 B2 B3 first
    A1 = disparity_2 - disparity_2*disparity_2/disparity_1
    A2 = 1 - (disparity_2*disparity_2)/(disparity_1*disparity_1)
    A3 = -sad_1*(disparity_2*disparity_2)/(disparity_1*disparity_1) + sad_2
    B1 = disparity_3 - disparity_3*disparity_3/disparity_1
    B2 = 1 - (disparity_3*disparity_3)/(disparity_1*disparity_1)
    B3 = -sad_1*(disparity_3*disparity_3)/(disparity_1*disparity_1) + sad_3
    # Now I will calculate b and c
    c = (A3/A1 - B3/B1)/(A2/A1-B2/B1)
    b = (A3 - c*A2)/A1
    # And finally I will get a
    a = (sad_1-b*disparity_1-c)/(disparity_1*disparity_1)
    # Now we have a function y = f(x) = ax^2 + bx + c with y is sad and x is disparity
    # We will calculate the extreme point of this equation
    disparity_min = -b/(2*a)
    sad_min = c - b*b/(4*a)
    return (disparity_min,sad_min)
