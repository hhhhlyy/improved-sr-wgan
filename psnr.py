import numpy as np
import math

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def rgb2gray(rgb):
    a =  np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    return a

