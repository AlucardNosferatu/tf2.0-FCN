import cv2
import numpy as np
from dataload import train_label_filenames


def correct_single_file(file_path):
    image = cv2.imread(file_path)
    height = image.shape[0]  # 高度
    width = image.shape[1]  # 宽度
    for i in range(0, width):  # 遍历所有长度的点
        for j in range(0, height):  # 遍历所有宽度的点
            data = image[j, i, :]
            if data.tolist() in [[0, 0, 254],[0, 0, 255]]:
                image[j, i, :] = np.array([0, 0, 255]).astype(np.uint8)
            else:
                image[j, i, :] = np.array([0, 0, 0]).astype(np.uint8)
    cv2.imwrite(file_path, image)
    print(file_path, 'Done')


def batch_correction():
    for file_name in train_label_filenames:
        correct_single_file(file_name)


if __name__ == '__main__':
    batch_correction()
