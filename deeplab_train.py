"""
Driver file to run the base UNet model on the SUIM dataset
"""
from __future__ import print_function, division
import os
from os.path import join, exists
from keras import callbacks
import matplotlib.pyplot as plt

from models.deeplabv3plus import DeeplabV3Plus
from utils.util_data import trainDataGenerator

# Preprocessing - Perform contrast enhancement
img_dir = "./Preprocessed Data - UCM/train_val/images/"
# contrast_enhancement_clahe(img_dir)

## Dataset directory
dataset_name = "suim"
train_dir = "./Preprocessed Data - UCM/train_val/"

## Checkpoint directory
ckpt_dir = "ckpt/"
im_res_ = (320, 320, 3)
ckpt_name = "deeplabv3plus.hdf5"
model_ckpt_name = join(ckpt_dir, ckpt_name)
if not exists(ckpt_dir): os.makedirs(ckpt_dir)

## Initialize model
model = DeeplabV3Plus(image_size=320, num_classes=5)
print (model.summary())
## load saved model
#model.load_weights(join("ckpt/saved/", "***.hdf5"))

""" Run options """
batch_size = 2
num_epochs = 25

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

## Fit model
history = model.fit(train_gen, 
          steps_per_epoch = 50,
          epochs = num_epochs,
          callbacks = [model_checkpoint])

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.show()