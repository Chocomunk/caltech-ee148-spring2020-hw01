import numpy as np
import matplotlib.pyplot as plt

from skimage.exposure import equalize_adapthist

from util import hsv2rgb, rgb2hsv, histogram_equalization, CLAHE, correlate, \
    black_tophat


def threshold_strategy(I, r_b=85, r_t=145, g_b=0, g_t=0, b_b=75, b_t=130):
    img = black_tophat(I, 11)
    img = rgb2hsv(img)
    img = black_tophat(img, 11)
    return (r_b <= img[:,:,0]) & (img[:,:,0] <= r_t) & \
           (g_b <= img[:,:,1]) & (img[:,:,1] <= g_t) & \
           (b_b <= img[:,:,2]) & (img[:,:,2] <= b_t)


def correlate_strategy(I, filters):
    # Using scipy's equalize_adapthist is preferred for better balancing
    img = rgb2hsv(I)
    # img[:,:,2] = CLAHE(img[:,:,2], tile_size=(16,16))
    # img[:,:,2] = equalize_adapthist(img[:,:,2], clip_limit=0.03) * 255
    img[:,:,2] = histogram_equalization(img[:,:,2])
    img = hsv2rgb(img)

    # Find cossim against original image
    output = np.zeros(img.shape[:2], dtype=np.float32)
    for filt in filters:
        corr = correlate(img, filt, step=2)
        output = np.maximum(output, corr)
    return output, img