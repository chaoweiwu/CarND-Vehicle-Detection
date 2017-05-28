from collections import namedtuple

import numpy as np

# Window = namedtuple('Window', 'x_start x_stop y_start y_stop')

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


