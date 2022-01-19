import numpy as np
import os
import cv2 as cv
from matplotlib import pyplot as plt

"""
We use 2 different methods to perform contrast enhancement
The 1st method is normal histogram equalization using a 
global contrast measure.

The 2nd method is called contrast limited adaptive 
histogram equalization(CLAHE), which not only uses local 
contrast measures to get gridwise contrast values, but
also clips the contrast value in case it gets too high.

For RGB images, we convert the image into YUV/LAB format 
to perform contrast enhancement without losing the original
colors of the image.
"""

""" Normal histogram equalization method """


def contrast_enhancement_normal(dir_path):
    # Read all files in the directory
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)

        # If the path corresponds to a file and not a folder
        if os.path.isfile(filepath):
            img = cv.imread(str(filepath))

            # Convert the image to YUV format
            img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)

            # Equalize the histogram of the Y channel
            img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])

            # Convert the YUV image back to RGB format
            img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

            # Write the image into its directory
            new_filepath = './HE/TEST/images/' + filename
            cv.imwrite(new_filepath, img_output)
            print("Done 1")


""" CLAHE Method """


def contrast_enhancement_clahe(dir_path):
    # Read all files in the directory
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)

        # If the path corresponds to a file and not a folder
        if os.path.isfile(filepath):
            gridsize = 8
            img = cv.imread(str(filepath))

            # Convert image to LAB format
            lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
            lab_planes = cv.split(lab)
            lab_planes_list = list(lab_planes)

            # Apply CLAHE to the L(lightness) channel only
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
            lab_planes_list[0] = clahe.apply(lab_planes_list[0])
            lab = cv.merge(lab_planes_list)

            # Convert back to RGB
            cl_img = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

            # Write the image into its directory
            new_filepath = './preprocessed_data/TEST/images/' + filename
            cv.imwrite(new_filepath, cl_img)
            print("Done 1")


def preprocessing_driver():
    dir_path = "../data/TEST/images/"
    contrast_enhancement_normal(dir_path)
    print("Done all")

if __name__ == "__main__":
    preprocessing_driver()
