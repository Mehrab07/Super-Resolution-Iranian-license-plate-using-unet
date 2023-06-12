from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras as K  # keras API
import tensorflow as tf  # tensorflow
import numpy as np



vgg = VGG19(include_top=False, weights='imagenet', input_shape=(None,None,3))
vgg_layer = K.Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
# make the net not trainable
for l in vgg_layer.layers: l.trainable=False 
# print(vgg_layer.summary())

def gray_3channel(image):
    image = tf.image.grayscale_to_rgb(image)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.asarray(image, dtype="uint8")
    return image

def perceptual_loss(y_true,y_pred):
    '''This function computes the perceptual loss using an already trained VGG layer'''
    y_t=vgg_layer(y_true)
    y_p=vgg_layer(y_pred)
    loss=K.losses.mean_squared_error(y_t,y_p)
    return loss

def psnr(y_true,y_pred):
    return tf.image.psnr(y_true,y_pred,1.0)
def ssim(y_true,y_pred):
    return tf.image.ssim(y_true,y_pred,1.0)
