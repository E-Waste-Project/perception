#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
from bisect import bisect_right, bisect_left
from perception.laptop_perception_helpers import read_and_resize, enclosing_rect_area
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import ros_numpy
from math import atan2, cos, sin, sqrt, pi


rospy.init_node("laptop_detection_tuning")
rospy.sleep(1)
laptop_data_publisher = rospy.Publisher("/laptop_data", Float32MultiArray)

read_img = False

if read_img:
    data_dir = "/home/ubuntu/data/laptop_base/"
    dset_sz = 34
    image_id = 1
    original_img = read_and_resize(data_dir, image_id)
    img = original_img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#############################
# Tunning Bars              #
#############################

cv2.namedWindow("image_window")
cv2.namedWindow("output_window")

# Parmaeters to tune
min_len = 200000
# min_len = 0
max_len = 500000
# min_circ = 76
use_canny = False
thresh1 = 34
thresh2 = 0
# morph_kernel = 3
morph_kernel = 0
# k = 15
k = 23
c = 2
eps = 0
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
cv2.createTrackbar('epsilon', 'output_window', eps, 1000, lambda x: None)
# cv2.createTrackbar('min_circ', 'output_window', min_circ, 100, nothing)

# ========================================================================== #

while True:
    if not read_img:
        msg = rospy.wait_for_message("/img", Image)
        original_img = ros_numpy.numpify(msg)
    
    if original_img.shape[0] == 3:
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        img = original_img.copy()
    else:
        gray = original_img.copy()
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    output = gray.copy()

    ###########################################
    # Retrieve Tunning Bars Values            #
    ###########################################

    if use_canny:
        thresh1 = cv2.getTrackbarPos('thresh1', 'output_window')
        thresh2 = cv2.getTrackbarPos('thresh2', 'output_window')

    morph_kernel = max((2 * cv2.getTrackbarPos('morph_kernel', 'output_window') - 1), 1)
    # dil_iter = max(cv2.getTrackbarPos('dil_iter', 'output_window'), 1)
    # close_iter = max(cv2.getTrackbarPos('close_iter', 'output_window'), 1)

    max_len = cv2.getTrackbarPos('top_len', 'output_window')
    min_len = cv2.getTrackbarPos('min_len', 'output_window')
    eps = cv2.getTrackbarPos('epsilon', 'output_window') / 1000.0
    # min_circ = (cv2.getTrackbarPos('min_circ', 'output_window') / 100.0)
    
    k = max((2 * cv2.getTrackbarPos('k', 'output_window') - 1), 1)
    c = max(cv2.getTrackbarPos('c', 'output_window'), 1)

    # ========================================================= #

    #############################
    # Pre-Processing            #
    #############################

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=c, tileGridSize=(2, 2))
    output = clahe.apply(output)

    # output = cv2.GaussianBlur(output, (k, k), 0)
    output = cv2.medianBlur(output, ksize=k)
    # ret, output = cv2.threshold(output, thresh, 255, cv2.THRESH_BINARY_INV)
    output = cv2.adaptiveThreshold(output, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv2.THRESH_BINARY_INV, 11, 2)

    if use_canny:
        output = cv2.Canny(output, thresh1, thresh2)

    # kernel = np.ones((2, 2), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    # Use erosion and dilation combination to eliminate false positives.
    # output = cv2.dilate(output, kernel, iterations=1)
    # output = cv2.erode(output, kernel, iterations=1)

    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ========================================================= #

    #############################
    # Find and Filter contours  #
    #############################

    contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=enclosing_rect_area)
    contours_per = list(map(enclosing_rect_area, contours))
    start = bisect_left(contours_per, min_len)
    end = bisect_right(contours_per, max_len)
    contours = contours[start:end+1]
    contours.reverse()

    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cnt = contours[0]
        
        # Find the convex hull object for each contour
        hull_list = [cv2.convexHull(cnt)]
        hll = hull_list[0]
        leftmost = tuple(hll[hll[:,:,0].argmin()][0])
        rightmost = tuple(hll[hll[:,:,0].argmax()][0])
        topmost = tuple(hll[hll[:,:,1].argmin()][0])
        bottommost = tuple(hll[hll[:,:,1].argmax()][0])
        cv2.circle(img, leftmost, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(img, rightmost, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(img, topmost, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(img, bottommost, 5, color=(255, 0, 0), thickness=-1)
        epsilon = eps*cv2.arcLength(hull_list[0],True)
        approx = cv2.approxPolyDP(hull_list[0],epsilon,True)
        rect = cv2.minAreaRect(hull_list[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Retrieve the key parameters of the rotated bounding box
        center = (int(rect[0][0]),int(rect[0][1])) 
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(rect[2])

        if width < height:
            angle = 90 - angle
            nw = (width // 2) * sin(angle * pi / 180)
            nh = (width // 2) * cos(angle * pi / 180)
            flip_radius = width
        else:
            angle = -angle
            nw = (height // 2) * sin(angle * pi / 180)
            nh = (height // 2) * cos(angle * pi / 180)
            flip_radius = height
        
        flip_point = (int(center[0] + nw), int(center[1] + nh))
        upper_point = (int(center[0] + nw), int(center[1] - nh))
        label = "  Rotation Angle: " + str(angle) + " degrees"
        textbox = cv2.rectangle(img, (center[0]-35, center[1]-25), 
            (center[0] + 295, center[1] + 10), (255,255,255), -1)
        cv2.putText(img, label, (center[0]-50, center[1]), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2)
        cv2.drawContours(img, [approx], 0, (0, 255, 255), 2)
        cv2.drawContours(img, [box], 0, (255, 255, 0), 2)
        cv2.circle(img, (x + w//2, y + h//2), 5, color=(0, 255, 0), thickness=2)
        cv2.circle(img, center, 10, color=(255, 0, 0), thickness=2)
        cv2.circle(img, flip_point, 20, color=(0, 255, 0), thickness=-1)
        cv2.line(img, upper_point, flip_point, color=(0, 255, 0), thickness=2)
        laptop_data_msg = Int32MultiArray()
        laptop_data_msg.data = [center[0], center[1], flip_point[0], flip_point[1], flip_radius]
        laptop_data_publisher.publish(laptop_data_msg)
    # ========================================================= #

    cv2.imshow("output_window", output)
    # cv2.imshow("thresh_window", thresh)
    cv2.imshow("image_window", img)

    key = cv2.waitKey(20) & 0xFF

    if key == ord('e'):
        break
    elif key == ord('c') and read_img:
        image_id += 1
        if image_id < dset_sz:
            original_img = read_and_resize(data_dir, image_id)
            img = original_img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            break

# close all open windows
cv2.destroyAllWindows()
