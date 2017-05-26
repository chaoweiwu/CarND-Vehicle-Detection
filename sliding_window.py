from collections import namedtuple
import numpy as np
import cv2

# Window = namedtuple('Window', 'x_start x_stop y_start y_stop')
from features import extract_features

Window = namedtuple('Window', 'start_xy stop_xy')


def sliding_window_gen(img_shape, x_start=0, x_stop=None, y_start=0, y_stop=None,
                       window_height=64, window_width=64, x_overlap=0.5, y_overlap=0.5):
    x_stop = img_shape[1] if x_stop is None else x_stop
    y_stop = img_shape[0] if y_stop is None else y_stop
    x_span = x_stop - x_start
    y_span = y_stop - y_start
    x_stepsize = np.int(window_width * (1 - x_overlap))
    y_stepsize = np.int(window_height * (1 - y_overlap))
    windows_per_row = int((x_span / x_stepsize) - 1)
    windows_per_col = int((y_span / y_stepsize) - 1)

    for y_cur in range(windows_per_col):
        for x_cur in range(windows_per_row):
            start_x = x_cur * x_stepsize + x_start
            start_y = y_cur * y_stepsize + y_start
            yield Window(
                (start_x, start_y),
                (start_x + window_width, start_y + window_height)
            )


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    """ Returns the windows which have positive results from the classifier in them. """

    positive_windows = []
    for window in windows:
        test_img = grab_inner_image(img, window)
        features = extract_features(test_img, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range)

        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            positive_windows.append(window)
    return positive_windows


def grab_inner_image(outer_img: np.ndarray, window: Window, output_size=(64, 64)) -> np.ndarray:
    start_x, start_y = window.start_xy
    stop_x, stop_y = window.stop_xy
    return cv2.resize(outer_img[start_y:stop_y, start_x:stop_x], output_size)


