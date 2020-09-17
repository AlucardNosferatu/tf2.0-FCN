import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from DL.config import weight_path, test_dir, result_path, image_shape
from DL.dataload import handle_data, train_label_filenames
from DL.model import new_my_model
from DL.train import activate_growth
from FillConcave import fill_array
from FitQuad import fit_array

COLORMAP = [[0, 0, 255], [0, 255, 0]]
cm = np.array(COLORMAP).astype(np.uint8)
cm2 = np.array([[0, 0, 0], [255, 255, 255]]).astype(np.uint8)

test_list_dir = os.listdir(test_dir)
test_list_dir.sort()
test_filenames = [test_dir + filename for filename in test_list_dir]


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


def transform_out(pred):
    pred = pred[0]  # pred维度为[h, w, n_class]
    pred = np.argmax(pred, axis=2)  # 获取通道的最大值的指数，比如模型输出某点的像素值为[0.1,0.5]，则该点的argmax为1.
    pred = cm2[pred]
    return pred


def write_pred(pred, filename):
    pred = transform_out(pred)
    cv2.imwrite(os.path.join(result_path, filename.split("/")[-1]), pred)
    print(filename.split("/")[-1] + " finished")


def load_model():
    model = new_my_model(n_class=2)
    model.load_weights(weight_path + 'fcn_20191021.ckpt')
    return model


def test_array(model, image):
    image = image[np.newaxis, :, :, :].astype("float32")
    with tf.device('/gpu:0'):
        out = model.predict(image)  # out的维度为[batch, h, w, n_class]
        out = transform_out(out)
        out = fill_array(out)
        out = fit_array(out)
        image = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
        img = (out * 0.3 + image * 0.7).astype(np.uint8)
        return img


def test_file():
    activate_growth()
    model = load_model()
    for i, filename in enumerate(test_filenames):
        image = handle_data(train_filenames=filename)
        img = test_array(model, image)
        cv2.imshow('img', img)
        cv2.waitKey()
        cv2.waitKey(1)
        # write_pred(out, filename)


def test_cam():
    activate_growth()
    model = load_model()
    sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    while sample.isOpened():
        ret, img = sample.read()
        if img is not None:
            img = cv2.resize(img, image_shape)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            image = test_array(model, img)
            cv2.imshow('img', image)
            # cv2.waitKey()
            cv2.waitKey(1)


if __name__ == '__main__':
    test_cam()
