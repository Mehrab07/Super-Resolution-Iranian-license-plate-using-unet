import os  # pkg to read filepaths from the dataset folder
import tensorflow as tf  # tensorflow
import tensorflow.keras as K  # keras API
import numpy as np  # numpy to work with tensors (with tf)
import matplotlib.pyplot as plt  # function to show images
import cv2


def preprocess_image(image):
    
    return np.array(image / 255)

# def scale_resize_image(image):
#     image = tf.image.convert_image_dtype(image, tf.float32)
#     image = tf.image.resize(image, (64, 256))
#     # image = cv2.resize(image, (256, 64))
#     return image


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 64))
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.asarray(image, dtype="uint8")
        if image is not None:
            images.append(image)
    return images