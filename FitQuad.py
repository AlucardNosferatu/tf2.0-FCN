import os

import cv2
import numpy as np

from FillConcave import check_result
from Points2Area import Point, GetAreaOfPolyGon

path = 'result/LCD_Filled/LCD (1).png'


def fit_quad(path):
    image = cv2.imread(path)
    thresh = cv2.Canny(image, 200, 100)
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for i in range(len(contours)):
        eps = 12
        temp_cnt = contours[i].copy()
        while len(temp_cnt) > 4:
            eps += 1
            temp_cnt = cv2.approxPolyDP(contours[i], eps, True)
        contours[i] = temp_cnt
        if len(contours[i]) != 4:
            areas.append(0)
        else:
            points = []
            for index in range(len(contours[i])):
                x = contours[i][index, 0, 0]
                y = contours[i][index, 0, 1]
                points.append(Point(x, y))
            area = GetAreaOfPolyGon(points)
            areas.append(area)
    cnt = contours[areas.index(max(areas))]
    image = np.zeros_like(image).astype(np.uint8)
    # image = cv2.drawContours(image, [cnt], -1, (255, 255, 255), 1)
    image = cv2.fillPoly(image, [cnt], (255, 255, 255))
    new_path = path.replace('LCD_Filled', 'LCD_Fitted')
    cv2.imwrite(new_path, image)


def batch_fit():
    file_list = os.listdir('result/LCD_Filled')
    for file in file_list:
        file = os.path.join('result/LCD_Filled', file)
        fit_quad(file)
        print(file, 'Done')


if __name__ == '__main__':
    batch_fit()
    check_result(mask_path='result/LCD_Fitted')
