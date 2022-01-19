
# encoding=utf-8
import os
import numpy as np
import cv2
import natsort

from LabStretching import LABStretching
from color_equalisation import RGB_equalisation
from global_stretching_RGB import stretching
from relativeglobalhistogramstretching import RelativeGHstretching

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

folder = "../.."

path = folder + "/data/TEST/images"
files = os.listdir(path)
files =  natsort.natsorted(files)

for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        img = cv2.imread(folder +'/data/TEST/images/' + file)
        height = len(img)
        width = len(img[0])

        sceneRadiance = img

        sceneRadiance = stretching(sceneRadiance)

        sceneRadiance = LABStretching(sceneRadiance)

        cv2.imwrite('../TEST/images/' + prefix + '_RGHS.jpg', sceneRadiance)

path = folder + "/data/train_val/images"
files = os.listdir(path)
files =  natsort.natsorted(files)

for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        img = cv2.imread(folder +'/data/train_val/images/' + file)
        height = len(img)
        width = len(img[0])

        sceneRadiance = img

        sceneRadiance = stretching(sceneRadiance)

        sceneRadiance = LABStretching(sceneRadiance)

        cv2.imwrite('../train_val/images/' + prefix + '_RGHS.jpg', sceneRadiance)
