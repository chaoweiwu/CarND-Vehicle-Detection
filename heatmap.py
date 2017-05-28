import numpy as np


def windows_to_heat_map(shape, windows):
    heatmap = np.zeros(shape).astype(np.float)
    return add_heat(heatmap, windows)


def combine_heat_maps(heatmaps):
    agg = np.zeros_like(heatmaps[0])
    for heatmap in heatmaps:
        agg += heatmap
    return agg


def add_heat(heat_map, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heat_map  # Iterate through list of bboxes