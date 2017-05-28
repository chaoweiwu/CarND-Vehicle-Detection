from itertools import chain

import numpy as np
from scipy.ndimage.measurements import label

import train_classifier
from sliding_window import sliding_window_gen, search_windows
from train_classifier import load_model
from util import draw_boxes, draw_labeled_bboxes

HEATMAP_THRESHOLD = 2


class VehicleDetection:
    def __init__(self):
        model_dict = load_model()
        self.model = model_dict[train_classifier.MODEL_KEY]
        self.scaler = model_dict[train_classifier.SCALER_KEY]

    def candidate_detections(self, img):
        hot_windows = search_windows(img, all_windows(img.shape), self.model, self.scaler)
        return draw_boxes(np.copy(img), hot_windows)

    def detect(self, img):
        hot_windows = search_windows(img, all_windows(img.shape), self.model, self.scaler)
        heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
        heatmap = add_heat(heatmap, hot_windows)
        thresholded = apply_threshold(heatmap, HEATMAP_THRESHOLD)
        heatmap = np.clip(thresholded, 0, 255)
        labels = label(heatmap)
        return draw_labeled_bboxes(np.copy(img), labels)


TOP = 360
HEIGHT = 256 + 128
# WINDOW_SIZES = [96, 128, 192, 256]
WINDOW_SIZES = [96, 128, 192]
OVERLAP = .7


def all_windows(shape):
    return list(chain.from_iterable([sliding_window_gen(shape,
                                                        y_start=TOP, y_stop=TOP + HEIGHT,
                                                        x_start=0,
                                                        window_width=ws, window_height=ws,
                                                        y_overlap=OVERLAP, x_overlap=OVERLAP) for ws in WINDOW_SIZES]))


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


if __name__ == '__main__':
    pass
