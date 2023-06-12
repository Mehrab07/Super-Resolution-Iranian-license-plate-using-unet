import random
import numpy as np
import cv2
import os
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt

from processes import *


def motion_blur_horizontal(image):
    kernel_size = 20
    angle = random.randint(-10, 10)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[(kernel_size - 1) // 2, :] = np.ones(kernel_size, dtype=np.float32)
    kernel = cv2.warpAffine(kernel,
                            cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1.0),
                            (kernel_size, kernel_size))
    kernel = kernel * (1.0 / np.sum(kernel))
    return cv2.filter2D(image, -1, kernel)


def motion_blur_vertical(image):
    kernel_size = 20
    angle = random.randint(-10, 10)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[:, (kernel_size - 1) // 2] = np.ones(kernel_size, dtype=np.float32)
    kernel = cv2.warpAffine(kernel,
                            cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1.0),
                            (kernel_size, kernel_size))
    kernel = kernel * (1.0 / np.sum(kernel))
    return cv2.filter2D(image, -1, kernel)


def getRandomPointsForSkew(cols, rows):
    return np.float32(
        [
            [cols * random.uniform(0.001, 0.1), rows * random.uniform(0.9, 0.99)],
            [cols * random.uniform(0.9, 0.99), rows * random.uniform(0.9, 0.99)],
            [cols * random.uniform(0.05, 0.1), rows * random.uniform(0.05, 0.1)],
            [cols * random.uniform(0.9, 0.99), rows * random.uniform(0.05, 0.1)]]
    )


def skew_image(image):
    rows, cols = image.shape[0], image.shape[1]
    pts1 = getRandomPointsForSkew(cols, rows)
    pts2 = getRandomPointsForSkew(cols, rows)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (cols, rows),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=[0, 0, 0])


def rotate_image(image):
    theta = random.randint(-5, 5)
    num_rows, num_cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), theta, 1)
    img_rotation = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))
    return img_rotation


def shift_image(image):
    px, py = np.random.normal(scale=3), np.random.normal(scale=3)
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, px], [0, 1, py]])
    dst = cv2.warpAffine(image, M, (cols, rows))
    return dst


def zoom_image(image):
    scale = random.uniform(0.8, 1.2)
    orgX, orgY = image.shape[1], image.shape[0]
    resized = cv2.resize(image, (int(orgX * scale), int(orgY * scale)))
    if scale >= 1:
        centerY, centerX = resized.shape[0] // 2, resized.shape[1] // 2
        fromX, fromY = centerX - (orgX // 2), centerY - (orgY // 2)
        return resized[fromY:fromY + orgY, fromX:fromX + orgX, ...]
    else:
        difX, difY = image.shape[1] - resized.shape[1], image.shape[0] - resized.shape[0]
        top, bot, left, right = difY // 2, difY // 2, difX // 2, difX // 2
        if (difY % 2) > 0: top += 1
        if (difX % 2) > 0: left += 1
        return cv2.copyMakeBorder(resized, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def change_light(image):
    return cv2.convertScaleAbs(image, alpha=random.uniform(0.5, 1.5), beta=random.randint(-10, 10))

def reduce_qulity(image):
    # image = np.array(image)
    for i in range(random.randint(5,19)):
        image = cv2.resize(image, (128, 32), interpolation = cv2.INTER_AREA)
        image = cv2.resize(image, (512, 128), interpolation = cv2.INTER_AREA)
        image = cv2.resize(image, (256, 64), interpolation = cv2.INTER_AREA)
 
    return image

def median_blur(image):
    image = np.asarray(image, dtype="uint8")
    medBlur = cv2.medianBlur(image, 7)
    return medBlur


class ImageGenerator:

    def __init__(self, data):
        self.data = data
        self.batch_size = 128
        self.image_height = 64
        self.image_width = 256
        self.image_channels = 3
        self.AUGMENT_OPS =[lambda x:x, rotate_image, shift_image, zoom_image, skew_image, motion_blur_horizontal] + [median_blur, reduce_qulity, motion_blur_vertical ] * 40



    def augmentImage(self, image):
        for i in range(random.randint(0,4)):
            image = np.random.choice(self.AUGMENT_OPS, replace=False)(image)
        return image

    def yieldData(self):
        while True:
            image = random.choice(self.data)
            hq_image = preprocess_image(image)
            lq_image = preprocess_image(self.augmentImage(image))
            yield lq_image, hq_image

    def emptyBatch(self):
        return (
            np.zeros([self.batch_size, self.image_height, self.image_width, self.image_channels]),
            np.zeros([self.batch_size, self.image_height, self.image_width, self.image_channels]), 0)

    def getBatches(self):
        X, Y, i = self.emptyBatch()
        for lq_image, hq_image in self.yieldData():
            X[i, ...] = lq_image
            Y[i, ...] = hq_image
            i += 1
            if i == self.batch_size:
                yield X, Y
                X, Y, i = self.emptyBatch()
