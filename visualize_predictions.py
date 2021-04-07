
import os
import json
import numpy as np
from PIL import Image

from util import draw_boxes, printProgressBar


if __name__=='__main__':
    # Directory for output visualizations
    out_path = 'data/out'
    os.makedirs(out_path,exist_ok=True)     # create directory if needed 

    # Load input images
    data_path = 'data/RedLights2011_Medium'
    file_names = sorted(os.listdir(data_path)) 
    file_names = [f for f in file_names if '.jpg' in f] 

    # Load predictions
    preds_path = 'data/hw01_preds' 
    data = None
    with open(os.path.join(preds_path,'preds_50.json'),'r') as f:
        data = json.load(f)
    assert data is not None

    # Draw and save bounding boxes for each file
    n = len(data)
    preds = {}
    printProgressBar(0, n, prefix='Progress:', suffix='Complete', length=50)
    for i in range(n):
        # Load boxes
        boxes = data[file_names[i]]
        
        # Read image using PIL
        I = np.asarray(Image.open(os.path.join(data_path,file_names[i])))
        result = draw_boxes(I, boxes)

        # Save new image
        Image.fromarray(result).save(os.path.join(out_path,file_names[i]))

        printProgressBar(i, n, prefix='Progress:', suffix='Complete', length=50)
        
