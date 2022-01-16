import numpy as np


def stretching(img):
    height = len(img)
    width = len(img[0])
    for k in range(0, 3):
        max_channel = np.max(img[:, :, k])
        min_channel = np.min(img[:, :, k])
        for i in range(height):
            for j in range(width):
                img[i, j, k] = (img[i, j, k] - min_channel) * (255 - 0) / (max_channel - min_channel) + 0
    return img


def global_stretching(img_L, height, width):
    i_min = np.min(img_L)
    i_max = np.max(img_L)
    i_mean = np.mean(img_L)

    array_Global_histogram_stretching_L = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            p_out = (img_L[i][j] - i_min) * (1 / (i_max - i_min))
            array_Global_histogram_stretching_L[i][j] = p_out

    return array_Global_histogram_stretching_L
