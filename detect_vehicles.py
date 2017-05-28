from collections import deque
from itertools import chain

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage.measurements import label

import train_classifier
from features import extract_features_from_windows
from heatmap import combine_heat_maps, add_heat, windows_to_heat_map
from sliding_window import sliding_window_gen
from train_classifier import load_model
from util import draw_boxes, draw_labeled_bboxes, apply_threshold

HEATMAP_THRESHOLD = 4


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
        heat_map = windows_to_heat_map(img.shape[:2], hot_windows)
        thresholded = apply_threshold(heat_map, HEATMAP_THRESHOLD)
        clipped = np.clip(thresholded, 0, 255)
        labels = label(clipped)
        return draw_labeled_bboxes(np.copy(img), labels)


class MultiFrameVehicleDetection(VehicleDetection):
    def __init__(self, buffer_len):
        super().__init__()
        self.heat_maps = deque(maxlen=buffer_len)

    def detect(self, img):
        hot_windows = search_windows(img, all_windows(img.shape), self.model, self.scaler)
        new_frame_heat_map = windows_to_heat_map(img.shape[:2], hot_windows)
        self.heat_maps.appendleft(new_frame_heat_map)

        thresholded = apply_threshold(combine_heat_maps(self.heat_maps), len(self.heat_maps) + HEATMAP_THRESHOLD)
        clipped = np.clip(thresholded, 0, 255)
        labels = label(clipped)
        return draw_labeled_bboxes(np.copy(img), labels)


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    """ Returns the windows which have positive results from the classifier in them. """

    X = extract_features_from_windows(img, windows, color_space=color_space,
                                      spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range)
    X = scaler.transform(X)
    predictions = clf.predict(X)
    return [windows[positive_ii] for positive_ii in predictions.nonzero()[0]]


def all_windows(shape):
    top = 360
    height = 256 + 128
    # window_sizes = [96, 128, 192, 256]
    window_sizes = [64, 96, 128]
    overlap = .6

    return list(chain.from_iterable([
                                        sliding_window_gen(shape,
                                                           y_start=top, y_stop=top + 2 * ws,
                                                           x_start=200,
                                                           window_width=ws, window_height=ws,
                                                           y_overlap=overlap, x_overlap=overlap) for ws in window_sizes
                                        ]))


def run_classifier_on_clip():
    detector = MultiFrameVehicleDetection(5)
    bad_sub_test_output = 'output_videos/project_video_sub.mp4'
    bad_sub = VideoFileClip("test_videos/project_video.mp4").subclip(17, 19)
    # bad_sub = VideoFileClip("test_videos/project_video.mp4").subclip(8, 10)
    bad_sub_test = bad_sub.fl_image(detector.detect)
    bad_sub_test.write_videofile(bad_sub_test_output, audio=False)


if __name__ == '__main__':
    run_classifier_on_clip()
