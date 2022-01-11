"""
Driver file to run the base UNet model on the SUIM dataset
"""
from __future__ import print_function, division
import os
from os.path import join, exists
from keras import callbacks

from models.unet import UNet_base
from utils.util_data import trainDataGenerator
from preprocessing import contrast_enhancement_clahe

# Preprocessing - Perform contrast enhancement
img_dir = "./data/train_val/images/"
contrast_enhancement_clahe(img_dir)

# Dataset directory
dataset_name = "suim"
train_dir = "./data/train_val/"

# Checkpoint directory
ckpt_dir = "ckpt/"
im_res_ = (320, 240, 3)
ckpt_name = "unet_rgb.hdf5"
model_ckpt_name = join(ckpt_dir, ckpt_name)
if not exists(ckpt_dir): os.makedirs(ckpt_dir)

# Initialize model
model = UNet_base(input_size=(im_res_[1], im_res_[0], 3), no_of_class=5)
print (model.summary())

# Load saved model
# model.load_weights(join("ckpt/saved/", "***.hdf5"))

""" Run options """
batch_size = 2
num_epochs = 1

# Setting up the parameters for the data augmentation
data_aug_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

model_checkpoint = callbacks.ModelCheckpoint(model_ckpt_name,
                                   monitor = 'loss',
                                   verbose = 1, mode= 'auto',
                                   save_weights_only = True,
                                   save_best_only = True)

# Data augmentation
train_gen = trainDataGenerator(batch_size, # batch_size
                              train_dir,# train-data dir
                              "images", # image_folder
                              "masks", # mask_folder
                              data_aug_args, # aug_dict
                              image_color_mode="rgb",
                              mask_color_mode="rgb",
                              target_size = (im_res_[1], im_res_[0]))

# Fit model
model.fit(train_gen,
          steps_per_epoch = 100,
          epochs = num_epochs,
          callbacks = [model_checkpoint])