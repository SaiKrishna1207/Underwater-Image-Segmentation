import os
import numpy as np
import cv2
import natsort
import xlwt
from global_histogram_stretching import stretching
from hsvStretching import HSVStretching
from sceneRadiance import sceneRadianceRGB


np.seterr(over='ignore')
if __name__ == '__main__':
    pass

# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/NonPhysical/ICM"
folder = "../../.."

path = folder + "/data/train_val/images"
files = os.listdir(path)
files = natsort.natsorted(files)

for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        print('********    file   ********', file)
        img = cv2.imread(folder + '/data/train_val/images/' + file)
        img = stretching(img)
        sceneRadiance = sceneRadianceRGB(img)
        sceneRadiance = HSVStretching(sceneRadiance)
        sceneRadiance = sceneRadianceRGB(sceneRadiance)
        cv2.imwrite('../train_val/images/' + prefix + '_ICM.jpg', sceneRadiance)

# path = folder + "/data/TEST/images"
# files = os.listdir(path)
# files = natsort.natsorted(files)
#
# for i in range(len(files)):
#     file = files[i]
#     filepath = path + "/" + file
#     prefix = file.split('.')[0]
#     if os.path.isfile(filepath):
#         print('********    file   ********', file)
#         img = cv2.imread(folder + '/data/TEST/images/' + file)
#         img = stretching(img)
#         sceneRadiance = sceneRadianceRGB(img)
#         sceneRadiance = HSVStretching(sceneRadiance)
#         sceneRadiance = sceneRadianceRGB(sceneRadiance)
#         cv2.imwrite('../TEST/images/' + prefix + '_ICM.jpg', sceneRadiance)