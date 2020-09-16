import os
import cv2
import numpy as np
import tensorflow as tf

from config import image_shape, train_dir, train_label_dir, test_dir

train_list_dir = os.listdir(train_dir)
train_list_dir.sort()
train_label_list_dir = os.listdir(train_label_dir)
train_label_list_dir.sort()

train_filenames = [train_dir + filename for filename in train_list_dir]
train_label_filenames = [train_label_dir +
                         filename for filename in train_label_list_dir]

test_list_dir = os.listdir(test_dir)
test_list_dir.sort()
test_filenames = [test_dir + filename for filename in test_list_dir]


def train_generator():
    for train_file_name, train_label_filename in zip(train_filenames, train_label_filenames):
        image, label = handle_data(train_file_name, train_label_filename)

        yield tf.convert_to_tensor(image), tf.convert_to_tensor(label)


def test_generator():
    for test_filename in test_filenames:
        image = handle_data(test_filename)

        yield tf.convert_to_tensor(image)


def handle_data(train_filenames, train_label_filenames=None):
    image = cv2.resize(cv2.imread(train_filenames), image_shape)
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

    if train_label_filenames is not None:
        gt_image = cv2.resize(cv2.imread(train_label_filenames), image_shape)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        background_color = np.array([255, 0, 0])
        gt_bg = np.all(gt_image == background_color, axis=2)
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

        return np.array(image), gt_image
    else:
        return np.array(image)
