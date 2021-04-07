# Homework 1: Red Light Detection

## Usage

Run `run_predictions.py` to generate the bounding-box predictions at `data/hw01_preds`. Note, intermediate predictions will be generated every 50 images for safety.

```
python run_predictions.py
```

Next, run `visualize_predictions.py` to draw the predicted bounding boxes on each of the input images, and save these new images to `data/out`

```
python visualize_predictions.py
```