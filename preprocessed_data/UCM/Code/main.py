""" UCM preprocessing for Underwater Colour Image Enhancement
This model is based on the repo:
https://github.com/wangyanckxx/Single-Underwater-Image-Enhancement-and-Color-Restoration
"""

import os
import numpy as np
import cv2
import natsort
import datetime

from color_equalisation import RGB_equalisation
from global_histogram_stretching import stretching
from hsvStretching import HSVStretching
from sceneRadiance import sceneRadianceRGB


np.seterr(over='ignore')
if __name__ == '__main__':
    pass

folder = "../.."

path = folder + "/data/train_val/images"
out_path = folder + ""
files = os.listdir(path)
files =  natsort.natsorted(files)

for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        img = cv2.imread(folder + '/data/train_val/images/' + file)
        sceneRadiance = RGB_equalisation(img)
        sceneRadiance = stretching(sceneRadiance)
        sceneRadiance = HSVStretching(sceneRadiance)
        sceneRadiance = sceneRadianceRGB(sceneRadiance)
        cv2.imwrite('../../Preprocessed Data/train_val/images/' + prefix + 'UCM.jpg', sceneRadiance)

# path = folder + "/data/TEST/images"
# out_path = folder + ""
# files = os.listdir(path)
# files =  natsort.natsorted(files)

# for i in range(len(files)):
#     file = files[i]
#     filepath = path + "/" + file
#     prefix = file.split('.')[0]
#     if os.path.isfile(filepath):
#         img = cv2.imread(folder + '/data/TEST/images/' + file)
#         sceneRadiance = RGB_equalisation(img)
#         sceneRadiance = stretching(sceneRadiance)
#         sceneRadiance = HSVStretching(sceneRadiance)
#         sceneRadiance = sceneRadianceRGB(sceneRadiance)
#         cv2.imwrite('../../Preprocessed Data/TEST/images/' + prefix + 'UCM.jpg', sceneRadiance)