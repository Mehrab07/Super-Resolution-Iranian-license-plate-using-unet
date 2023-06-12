import os  # pkg to read filepaths from the dataset folder
import tensorflow as tf  # tensorflow
import tensorflow.keras as K  # keras API
import numpy as np  # numpy to work with tensors (with tf)
import matplotlib.pyplot as plt  # function to show images
import cv2
from tensorflow.keras.applications.vgg19 import VGG19

from lrPlateGenerator import *
from model import *
from loss import *
from processes import *




x_val = np.stack([preprocess_image(cv2.imread(os.path.join('dataset_grey_val', "x", fname))) for fname in os.listdir('dataset_grey_val/x')])    
y_val = np.stack([preprocess_image(cv2.imread(os.path.join('dataset_grey_val', "y", fname))) for fname in os.listdir('dataset_grey_val/y')])    

plate_data = load_images_from_folder('plate_train')
generator = ImageGenerator(plate_data)


model = RUNet()


chk = tf.keras.callbacks.ModelCheckpoint("RUnet_checkpoint/{epoch:02d}-{val_loss:.2f}.h5", monitor='val_loss', save_best_only=True) 
callbacks_list = [chk]

opt=K.optimizers.Adam(learning_rate=0.001) # Adam optimizer
model.compile(optimizer=opt,loss=perceptual_loss,metrics=[psnr,ssim,K.losses.mean_squared_error])

model.fit_generator(generator.getBatches(),validation_data=(x_val, y_val),epochs=1000, steps_per_epoch=20, callbacks=callbacks_list)

model.save("RUnet_Model_grey.h5")

