import os
import cv2
import numpy as np


def fill_concave(path):
    image = cv2.imread(path)
    thresh = cv2.Canny(image, 200, 100)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        # length = len(hull)
        # for i in range(length):
        #     image = cv2.line(
        #         image,
        #         tuple(hull[i][0]),
        #         tuple(hull[(i + 1) % length][0]),
        #         (0, 0, 255),
        #         2
        #     )
        image = cv2.fillPoly(image, [hull], (255, 255, 255))
    cv2.imwrite(path.replace('result/LCD', 'result/LCD_Filled'), image)


def batch_fill():
    file_list = os.listdir('result/LCD')
    for file in file_list:
        file = os.path.join('result/LCD', file)
        fill_concave(file)
        print(file, 'Done')


def check_result(mask_path='result/LCD_Filled'):
    path1 = 'data/LCD/train/img'
    path2 = mask_path
    file_list = os.listdir(path2)
    for file in file_list:
        img2 = cv2.imread(os.path.join(path2, file))
        img1 = cv2.imread(os.path.join(path1, file))
        img1 = cv2.resize(img1, (1024, 768))
        img = (img2 * 0.3 + img1 * 0.7).astype(np.uint8)
        cv2.imshow('blended', img)
        cv2.waitKey()


if __name__ == '__main__':
    batch_fill()
    check_result()