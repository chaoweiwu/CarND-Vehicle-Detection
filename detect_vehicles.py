from itertools import chain

import numpy as np
from scipy.ndimage.measurements import label
import cv2

import train_classifier
from sliding_window import sliding_window_gen, search_windows
from train_classifier import load_model


class VehicleDetection:
    def __init__(self):
        model_dict = load_model()
        self.model = model_dict[train_classifier.MODEL_KEY]
        self.scaler = model_dict[train_classifier.SCALER_KEY]

    def detect(self, img):
        hot_windows = search_windows(img, all_windows_gen(img.shape), self.model, self.scaler)
        heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
        heatmap = add_heat(heatmap, hot_windows)
        thresholded = apply_threshold(heatmap, 1)
        heatmap = np.clip(thresholded, 0, 255)
        labels = label(heatmap)
        return draw_labeled_bboxes(np.copy(img), labels)


def all_windows_gen(img_shape):
    """ All candidate windows for cars to appear in. """
    big_windows = sliding_window_gen(img_shape, y_start=500,
                                     window_width=280, window_height=200,
                                     y_overlap=.8, x_overlap=.8)

    medium_windows = sliding_window_gen(img_shape,
                                        y_start=500, y_stop=600,
                                        x_start=100, x_stop=1200,
                                        window_width=100, window_height=100,
                                        y_overlap=.8, x_overlap=.8)
    small_windows = sliding_window_gen(img_shape,
                                       y_start=500, y_stop=600,
                                       x_start=200, x_stop=1000,
                                       window_width=70, window_height=70,
                                       y_overlap=.7, x_overlap=.7)

    tiny_windows = sliding_window_gen(img_shape,
                                      y_start=510, y_stop=580,
                                      x_start=400, x_stop=800,
                                      window_width=30, window_height=30,
                                      y_overlap=.7, x_overlap=.7)

    return chain(tiny_windows, small_windows, medium_windows, big_windows)


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


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


if __name__ == '__main__':
    detect_vehicles()
