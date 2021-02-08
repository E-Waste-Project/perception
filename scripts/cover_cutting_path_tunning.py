#!/usr/bin/env python

# import sys

# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy as np
from perception.laptop_perception_helpers import plan_cover_cutting_path, read_and_resize

data_dir = "/home/ubuntu/data/laptop_base/"
dset_sz = 29
image_id = 1
original_img = read_and_resize(data_dir, image_id)
img = original_img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#############################
# Tunning Bars              #
#############################

cv2.namedWindow("image_window")

# Parmaeters to tune
tol = 30
min_hole_dist = 5

cv2.createTrackbar('tolerance', 'image_window', tol, 1000, lambda x: None)
cv2.createTrackbar('min_hole_dist', 'image_window', min_hole_dist, 1000, lambda x: None)

# ========================================================================== #

while True:

    output = gray.copy()
    img = original_img.copy()

    ###########################################
    # Retrieve Tunning Bars Values            #
    ###########################################

    tol = cv2.getTrackbarPos('tolerance', 'image_window')
    min_hole_dist = cv2.getTrackbarPos('min_hole_dist', 'image_window')

    # ========================================================= #

    ####################################################################################
    # Plan Cover Cutting Path given a gray image, tolerance, and minimum hole distance.#
    ####################################################################################

    cut_path = plan_cover_cutting_path(gray, tol, min_hole_dist, draw_on=img, method=1)
    for i in range(len(cut_path) - 1):
        cv2.line(img, tuple(cut_path[i]), tuple(cut_path[i+1]), (0, 0, 255), 2)    

    # ======================================================================== #

    cv2.imshow("image_window", img)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('e'):
        break
    elif key == ord('c'):
        print("new_img")
        image_id += 1
        if image_id < dset_sz:
            original_img = read_and_resize(data_dir, image_id)
            img = original_img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            break

# close all open windows
cv2.destroyAllWindows()
