import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from config import weight_path, test_dir, result_path
from dataload import handle_data, train_label_filenames
from model import new_my_model
from train import activate_growth

COLORMAP = [[0, 0, 255], [0, 255, 0]]
cm = np.array(COLORMAP).astype(np.uint8)
cm2 = np.array([[0, 0, 0], [255, 255, 255]]).astype(np.uint8)


def addweight(pred, test_img):
    # 标签添加透明通道，叠加在原图上
    pred = Image.fromarray(pred.astype('uint8')).convert('RGBA')
    test_img = test_img[0]
    out = np.zeros(test_img.shape, test_img.dtype)
    cv2.normalize(test_img, out, 0,
                  255, cv2.NORM_MINMAX)
    image = Image.fromarray(out.astype('uint8')).convert('RGBA')
    image = Image.blend(image, pred, 0.3)
    return image


def write_pred(image, pred):
    pred = pred[0]  # pred维度为[h, w, n_class]
    pred = np.argmax(pred, axis=2)  # 获取通道的最大值的指数，比如模型输出某点的像素值为[0.1,0.5]，则该点的argmax为1.
    # pred = cm[pred]  # 将预测结果的像素值改为cm定义的值，这是语义分割常用方法。这一步是为了将上一步的1转换为cm的第二个值，即[0,255,0]
    pred = cm2[pred]
    # weighted_pred = addweight(pred, image)
    # weighted_pred.save(os.path.join(result_path, filename.split("/")[-1]))
    cv2.imwrite(os.path.join(result_path, filename.split("/")[-1]), pred)
    print(filename.split("/")[-1] + " finished")


def load_model():
    model = new_my_model(n_class=2)
    model.load_weights(weight_path + 'fcn_20191021.ckpt')
    return model


test_list_dir = os.listdir(test_dir)
test_list_dir.sort()
test_filenames = [test_dir + filename for filename in test_list_dir]

activate_growth()
model = load_model()
for i, filename in enumerate(test_filenames):
    image, gt_label = handle_data(train_filenames=filename, train_label_filenames=train_label_filenames[i])
    image = image[np.newaxis, :, :, :].astype("float32")
    with tf.device('/gpu:0'):
        out = model.predict(image)  # out的维度为[batch, h, w, n_class]
        write_pred(image, out)
