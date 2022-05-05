import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import filters
from skimage import color
from skimage import util
import cv2 as cv

# read left and right images
imgR = cv.imread("photo/cones/im2.png", 0)
imgL = cv.imread("photo/cones/im6.png", 0)
start_time = time.time()
# creates StereoBm object
stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)

# computes disparity
disparity = stereo.compute(imgL, imgR)

# displays image as grayscale and plotted
plt.imshow(disparity, "gray")
plt.show()
print(f"--- {time.time() - start_time} seconds ---")
