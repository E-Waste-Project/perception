#!/usr/bin/env python

import cv2
import numpy as np
from bisect import bisect_right, bisect_left
from perception.laptop_perception_helpers import read_and_resize, enclosing_rect_area

data_dir = "/home/ubuntu/data/laptop_motherboard/"
dset_sz = 10
image_id = 1
original_img = read_and_resize(data_dir, image_id, compression='')
img = original_img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#############################
# Tunning Bars              #
#############################

cv2.namedWindow("image_window")
cv2.namedWindow("output_window")

# Parmaeters to tune
color = 'g'
if(color == 'b'):
    low_H = 93
    low_S = 73
    low_V = 63
else:
    low_H = 44
    low_S = 26
    low_V = 80
high_H = 100
high_S = 255
high_V = 255
min_len = 70000
max_len = 200000
# min_circ = 76
use_canny = False
thresh1 = 34
thresh2 = 0
morph_kernel = 0
k = 49
c = 2

cv2.createTrackbar('low_H', 'image_window', low_H, 255, lambda x: None)
cv2.createTrackbar('low_S', 'image_window', low_S, 255, lambda x: None)
cv2.createTrackbar('low_V', 'image_window', low_V, 255, lambda x: None)
cv2.createTrackbar('high_H', 'image_window', high_H, 255, lambda x: None)
cv2.createTrackbar('high_S', 'image_window', high_S, 255, lambda x: None)
cv2.createTrackbar('high_V', 'image_window', high_V, 255, lambda x: None)

if use_canny:
    cv2.createTrackbar('thresh1', 'output_window', thresh1, 1000, lambda x: None)
    cv2.createTrackbar('thresh2', 'output_window', thresh2, 1000, lambda x: None)


cv2.createTrackbar('k', 'output_window', int(0.5*k+0.5), 25, lambda x: None)
cv2.createTrackbar('c', 'output_window', c, 25, lambda x: None)

cv2.createTrackbar('morph_kernel', 'output_window', morph_kernel, 100, lambda x: None)
# cv2.createTrackbar('dil_iter', 'output_window', 1, 100, nothing)
# cv2.createTrackbar('close_iter', 'output_window', 1, 100, nothing)

cv2.createTrackbar('top_len', 'output_window', max_len, 1000000, lambda x: None)
cv2.createTrackbar('min_len', 'output_window', min_len, 1000000, lambda x: None)
# cv2.createTrackbar('min_circ', 'output_window', min_circ, 100, nothing)

# ========================================================================== #

while True:

    output = gray.copy()
    img = original_img.copy()

    ###########################################
    # Retrieve Tunning Bars Values            #
    ###########################################

    if use_canny:
        thresh1 = cv2.getTrackbarPos('thresh1', 'output_window')
        thresh2 = cv2.getTrackbarPos('thresh2', 'output_window')
    
    low_H = cv2.getTrackbarPos('low_H', 'image_window')
    low_S = cv2.getTrackbarPos('low_S', 'image_window')
    low_V = cv2.getTrackbarPos('low_V', 'image_window')
    high_H = cv2.getTrackbarPos('high_H', 'image_window')
    high_S = cv2.getTrackbarPos('high_S', 'image_window')
    high_V = cv2.getTrackbarPos('high_V', 'image_window')


    morph_kernel = max((2 * cv2.getTrackbarPos('morph_kernel', 'output_window') - 1), 1)
    # dil_iter = max(cv2.getTrackbarPos('dil_iter', 'output_window'), 1)
    # close_iter = max(cv2.getTrackbarPos('close_iter', 'output_window'), 1)

    max_len = cv2.getTrackbarPos('top_len', 'output_window')
    min_len = cv2.getTrackbarPos('min_len', 'output_window')
    # min_circ = (cv2.getTrackbarPos('min_circ', 'output_window') / 100.0)
    
    k = max((2 * cv2.getTrackbarPos('k', 'output_window') - 1), 1)
    c = max(cv2.getTrackbarPos('c', 'output_window'), 1)

    # ========================================================= #

    #############################
    # Pre-Processing            #
    #############################

    output = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # create a CLAHE object (Arguments are optional).
    # clahe = cv2.createCLAHE(clipLimit=c, tileGridSize=(2, 2))
    # output = clahe.apply(output)

    output = cv2.GaussianBlur(output, (k, k), 0)

    # ret, output = cv2.threshold(output, thresh, 255, cv2.THRESH_BINARY_INV)
    # output = cv2.adaptiveThreshold(output, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                            #    cv2.THRESH_BINARY_INV, 11, 2)
    
    output = cv2.inRange(output, (low_H, low_S, low_V), (high_H, high_S, high_V))

    if use_canny:
        output = cv2.Canny(output, thresh1, thresh2)

    # kernel = np.ones((2, 2), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    # Use erosion and dilation combination to eliminate false positives.
    # output = cv2.dilate(output, kernel, iterations=1)
    # output = cv2.erode(output, kernel, iterations=1)

    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel, iterations=1)

    # ========================================================= #

    #############################
    # Find and Filter contours  #
    #############################

    # output2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img = img.reshape((-1, 3))
    # img[output.flatten() == 0] = [0, 0, 0]
    # img = img.reshape(original_img.shape)

    contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=enclosing_rect_area)
    contours_per = list(map(enclosing_rect_area, contours))
    start = bisect_left(contours_per, min_len)
    end = bisect_right(contours_per, max_len)
    contours = contours[start:end+1]

    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    # ========================================================= #

    cv2.imshow("output_window", output)
    # cv2.imshow("thresh_window", thresh)
    cv2.imshow("image_window", img)

    key = cv2.waitKey(20) & 0xFF

    if key == ord('e'):
        break
    elif key == ord('c'):
        image_id += 1
        if image_id <= dset_sz:
            original_img = read_and_resize(data_dir, image_id, compression='')
            img = original_img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            break

# close all open windows
cv2.destroyAllWindows()
