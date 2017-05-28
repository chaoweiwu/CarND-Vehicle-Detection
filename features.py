from typing import Tuple

import cv2
from skimage.feature import hog
import numpy as np

from util import grab_inner_image

ORIENTATIONS = 6
PIXELS_PER_CELL = 8
CELLS_PER_BLOCK = 2
HOG_CHANNEL = 0  # ALL


def extract_features_many(imgs, color_space='RGB', spatial_bins=(32, 32), hist_bins=32, hist_range=(0, 256)):
    """ Convert an iterable of images to a list of feature vectors. """
    features = [extract_features(img, color_space, spatial_bins, hist_bins, hist_range) for img in imgs]
    return features


def extract_features_from_windows(img, windows, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                                  hist_range=(0, 256)):
    """ Convert an image and set of windows to a feature matrix. """

    all_window_features = []
    for window in windows:
        window_image = grab_inner_image(img, window)
        window_features = extract_features(window_image, color_space, spatial_size, hist_bins, hist_range)
        all_window_features.append(window_features)
    return np.vstack(all_window_features)


def extract_features(img: np.ndarray, color_space: str = 'RGB', spatial_size: Tuple = (32, 32),
                     hist_bins: int = 32, hist_range: Tuple = (0, 256)):
    """ Convert an image to a vector of features. """
    img = convert_color_space(img, color_space)
    binned = bin_spatial(img, spatial_size)
    colored = color_hist(img, hist_bins, hist_range)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hog_features = hog(gray, ORIENTATIONS, (PIXELS_PER_CELL, PIXELS_PER_CELL),
                       (CELLS_PER_BLOCK, CELLS_PER_BLOCK))
    return np.concatenate([binned, colored, hog_features])


# class Hog:
#     def __init__(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         self.nx_blocks = img.shape[1] // PIXELS_PER_CELL - 1
#         self.ny_blocks = img.shape[0] // PIXELS_PER_CELL - 1
#         n_feat
#         hog_features = hog(gray, ORIENTATIONS, (PIXELS_PER_CELL, PIXELS_PER_CELL),
#                            (CELLS_PER_BLOCK, CELLS_PER_BLOCK))
#         
#     def 


def convert_color_space(img, color_space):
    if color_space == 'RGB':
        return img.copy()
    elif color_space == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    else:
        raise ValueError("color space: " + color_space + " is not valid.")


def bin_spatial(img, size=(32, 32)):
    """ Compute binned color features """
    return cv2.resize(img, size).ravel()


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """ Compute color histogram features """
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# def get_hog_features(img, orientations, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
#     ret = hog(img,
#               orientations=orienta,
#               pixels_per_cell=(pix_per_cell, pix_per_cell),
#               cells_per_block=(cell_per_block, cell_per_block),
#               visualise=vis, feature_vector=feature_vec)
#     if vis:
#         return ret[0], ret[1]
#     else:
#         return ret
