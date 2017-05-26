from typing import Iterable, Sequence, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2


def plot3d(pixels, colors_rgb, axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


def show_images(titled_images: Sequence[Tuple[str, np.ndarray]], figsize=(20, 10), rows=1, cmap="gray"):
    """ Show a bunch of images in a grid """
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(rows + 1, int(len(titled_images) / rows))

    for (ii, (name, img)) in enumerate(titled_images):
        ax = plt.subplot(gs[ii])
        ax.set_title(name)
        ax.axis('off')
        ax.imshow(img, cmap=cmap)


def show_images_from_paths(paths: Sequence[str], figsize=(20, 10), rows=1, cmap="gray"):
    """ Display a bunch of images from a list of paths. """
    titled_images = zip(paths, list(paths_to_images_gen(paths)))
    show_images(list(titled_images))


def paths_to_images_gen(image_paths: Iterable[str]):
    for path in image_paths:
        yield mpimg.imread(path)


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    draw_img = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    return draw_img
