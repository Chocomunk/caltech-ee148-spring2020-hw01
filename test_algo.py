import os

import numpy as np

from PIL import Image
from run_predictions import detect_red_light


def draw_boxes(I, bounding_boxes):
    """ Finds and draws red-light bounding boxes, returns the new image """
    # Top, Left, Bottom, Right coords
    for t, l, b, r in bounding_boxes:
        # Clear red and green (add value for brightness)
        I[t:b,l,0:2] = 90      # left wall
        I[t:b,r,0:2] = 90      # right wall
        I[t,l:r,0:2] = 90      # top wall
        I[b,l:r,0:2] = 90      # bottom wall

        # Color in blue
        I[t:b,l,2] = 255      # left wall
        I[t:b,r,2] = 255      # right wall
        I[t,l:r,2] = 255      # top wall
        I[b,l:r,2] = 255      # bottom wall
    return I


if __name__=="__main__":
    # set the path to the downloaded data: 
    data_path = 'data/RedLights2011_Medium'

    # get sorted list of files: 
    file_names = sorted(os.listdir(data_path)) 

    # remove any non-JPEG files: 
    file_names = [f for f in file_names if '.jpg' in f] 

    i = 0
    looping = True
    while i < len(file_names) and looping:
        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names[i]))
        
        # convert to numpy array:
        I = np.asarray(I)
        I.flags.writeable = True
        
        Image.fromarray(draw_boxes(I, detect_red_light(I))).show()

        i += 1
        looping = False     # Stop immediately for testing
