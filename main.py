# Stéréovision

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import filters
from skimage import color
from skimage import util
#%% Load images

i_g = io.imread('photo/im0.png', as_gray=True)
i_d = io.imread('photo/im1.png', as_gray=True)
plt.subplot(1, 2, 1)
plt.imshow(i_g, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(i_d, cmap='gray')
plt.show()
d_est = 20  # In pixels

#%% Filter ?

h = np.full((9, 9), 1) / (9**2)
i_g = filters.convolve(i_g, h)
i_d = filters.convolve(i_d, h)


#%% Disparity computation

def disparity(i1: np.ndarray, i2: np.ndarray, max_disp: int, N: int = 5):
    h, w = i1.shape
    xx, yy = np.meshgrid(np.arange(max_disp, w - N, step=N), np.arange(h - N, step=1))

    i_out = np.zeros(i1.shape)

    def diff(bx: int, by: int) -> np.ndarray:
        diffs = np.zeros(max_disp)
        wbx, wby = np.meshgrid(np.arange(bx, bx + N), np.arange(by, by + N))
        for i in range(0, max_disp):
            wx, wy = np.meshgrid(np.arange(bx - i, bx - i + N), np.arange(by, by + N))
            # print(wx, '\n', wy)
            diffs[i] = np.fabs(i1[wy, wx] - i2[wby, wbx]).sum()

        return diffs

    for x, y in zip(xx.flatten(), yy.flatten()):
        sad = diff(x, y)
        diss = sad.argmin()
        i_out[y, x:x+N] = diss

    return i_out


nn = 7
test = disparity(i_d, i_g, d_est, nn)
#test2 = disparity(i_g, i_d, d_est, nn)
#plt.subplot(1, 2, 1)
plt.imshow(test[:, d_est:], cmap='gray')
#plt.subplot(1, 2, 2)
#plt.imshow(test2[:, d_est:], cmap='gray')
plt.title('N = {}'.format(nn))
plt.show()