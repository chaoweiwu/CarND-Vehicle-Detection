from typing import Tuple

import cv2
from skimage.feature import hog
import numpy as np

from util import paths_to_images_gen


def extract_features_many(imgs, color_space='RGB', spatial_bins=(32, 32), hist_bins=32, hist_range=(0, 256)):
    """ Convert an iterable of images to a list of feature vectors. """
    return [extract_features(img, color_space, spatial_bins, hist_bins, hist_range) for img in imgs]


def extract_features(img: np.ndarray, color_space: str = 'RGB', spatial_size: Tuple = (32, 32),
                     hist_bins: int = 32, hist_range: Tuple = (0, 256)):
    """ Convert an image to a vector of features. """
    img = convert_color_space(img, color_space)
    binned = bin_spatial(img, spatial_size)
    colored = color_hist(img, hist_bins, hist_range)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hog_features = hog(gray)
    return np.concatenate([binned, colored, hog_features])


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
