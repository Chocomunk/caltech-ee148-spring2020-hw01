import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.exposure import equalize_adapthist
from PIL import Image

from util import load_filters, get_boxes, draw_boxes, binary_dilation, gray2rgb
from strategies import correlate_strategy, threshold_strategy


# RGB thresholds

# red light
# r_b = 180
# r_t = 255
# g_b = 135
# g_t = 255
# b_b = 0
# b_t = 180

# r_b = 20
# r_t = 45
# g_b = 30
# g_t = 255
# b_b = 165
# b_t = 255

# Darks
# r_b = 0
# r_t = 255
# g_b = 60
# g_t = 255
# b_b = 30
# b_t = 255

# --- Find red lights from black tophat conversion ----
r_b = 85
r_t = 145
g_b = 0
g_t = 0
b_b = 75
b_t = 130


CMAP = plt.cm.get_cmap('tab20')


def get_image(path, fname):
    I = np.asarray(Image.open(os.path.join(path,fname)))
    r = I.shape[0]
    return I[:2*r//3]


def gen_image_threshold(I):
    orig = np.copy(I)
    mask = threshold_strategy(I, r_b, r_t, g_b, g_t, b_b, b_t)
    dilate = binary_dilation(mask, iterations=5)
    boxes = get_boxes(dilate)
    orig = draw_boxes(orig, boxes)

    mask = np.repeat(mask[:,:,np.newaxis], 3, 2)
    dilate = np.repeat(dilate[:,:,np.newaxis], 3, 2)

    left = np.concatenate((orig, mask * 255), 0).astype(np.uint8)
    right = np.concatenate((I, dilate * orig), 0).astype(np.uint8)
    return np.concatenate((left, right), 1)


def gen_image_correlate(I):
    corr, proc_img = correlate_strategy(I, filters)
    print(np.max(corr))
    map_out = CMAP(corr)[:,:,:3] * 255
    output = binary_dilation((corr > 0.83), iterations=5)
    boxes = get_boxes(output)
    I = draw_boxes(I, boxes)

    left = np.concatenate((I, gray2rgb(output) * I), 0)
    right = np.concatenate((proc_img.astype(np.uint8), 
                            map_out.astype(np.uint8)), 0)
    return np.concatenate((left, right), 1)


def gen_image(I):
    """ Simple QOL function to select between the two algorithms above """
    # im = np.concatenate((I, wtph(I, 11), white_tophat(I, 5)), 0)
    im = gen_image_correlate(I)
    # im = gen_image_threshold(I)

    height_diff = im.shape[0] - compound_filter.shape[0]
    filter_image = np.pad(compound_filter, [(0,height_diff), (0,0), (0,0)], mode='constant')
    return np.concatenate((im, filter_image), 1)

def press(event):
    global r_b, r_t, g_b, g_t, b_b, b_t, i, I, looping
    # Control commands
    if event.key == 'g':        # Stop program completely
        looping = False
        plt.close()
    else:                       # Updates image after keypress
        if event.key == 'n':      # Next image
            i = min(n-1, i+1)
            I = get_image(data_path, file_names[i])
        if event.key == 'b':      # Previous image
            i = max(0, i-1)
            I = get_image(data_path, file_names[i])

        # Bottom boundaries
        # ______R___G___B_
        # incr |q   w   e
        # decr |a   s   d
        elif event.key == 'q':
            r_b = min(r_t, r_b+1)
        elif event.key == 'a':
            r_b = max(0, r_b-1)
        elif event.key == 'w':
            g_b = min(g_t, g_b+1)
        elif event.key == 's':
            g_b = max(0, g_b-1)
        elif event.key == 'e':
            b_b = min(b_t, b_b+1)
        elif event.key == 'd':
            b_b = max(0, b_b-1)

        # Top boundaries
        # ______R___G___B_
        # incr |u   i   o
        # decr |j   k   l
        elif event.key == 'u':
            r_t = min(255, r_t+1)
        elif event.key == 'j':
            r_t = max(r_b, r_t-1)
        elif event.key == 'i':
            g_t = min(255, g_t+1)
        elif event.key == 'k':
            g_t = max(g_b, g_t-1)
        elif event.key == 'o':
            b_t = min(255, b_t+1)
        elif event.key == 'l':
            b_t = max(b_b, b_t-1)

        # Update image
        im.set_array(gen_image(I))
        ax.set_ylabel('Upper Bounds: {0}, {1}, {2}'.format(r_t, g_t, b_t))
        ax.set_xlabel('Lower Bounds: {0}, {1}, {2}'.format(r_b, g_b, b_b))
        fig.canvas.draw()


if __name__=="__main__":
    # Data images setup
    data_path = 'data/RedLights2011_Medium'
    # data_path = 'red-lights/balance'
    file_names = sorted(os.listdir(data_path)) 
    file_names = [f for f in file_names if '.jpg' in f] 
    n = len(file_names)

    # Red-light image setup
    filters, compound_filter = load_filters('red-lights/balance')

    # Loop sentinels
    i = 0
    looping = True

    # Figure management
    fig, ax = plt.subplots()
    I = get_image(data_path, file_names[i])
    im = ax.imshow(gen_image(I))
    ax.set_ylabel('Upper Bounds: {0}, {1}, {2}'.format(r_t, g_t, b_t))
    ax.set_xlabel('Lower Bounds: {0}, {1}, {2}'.format(r_b, g_b, b_b))
    ax.set_title("Press 'g' to quit, 'n' for next image, 'b' for prev image")

    # Bind keypress controller
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.mpl_connect('key_press_event', press)

    # Main loop
    plt.show()
