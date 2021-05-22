#!/usr/bin/env python

import cv2
import numpy as np
from bisect import bisect_right, bisect_left
from perception.laptop_perception_helpers import read_and_resize, enclosing_rect_area
import sys
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from ros_numpy import numpify


class FindCorner:
    
    def __init__(self):
        
        rospy.init_node("find_corners")

        cv2.namedWindow("image_window")
        cv2.namedWindow("output_window")
        cv2.namedWindow("image")
        
        #############################
        # Tunning Bars              #
        #############################

        # Parmaeters to tune
        min_len = 138000
        max_len = 200000
        # min_circ = 76
        self.use_canny = False
        thresh1 = 34
        thresh2 = 0
        morph_kernel = 3
        k = 15
        c = 2

        if self.use_canny:
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
        
        # initialize the list of reference points and boolean indicating
        # whether cropping is being performed or not
        self.refPt = []
        self.cropping = False
        self.cropped = False
        
        cv2.setMouseCallback("image", self.click_and_crop)

    
    def find_corners(self):

        while True:
            # # Wait for rgb camera stream to publish a frame.
            image_msg = rospy.wait_for_message('/camera/color/image_raw', Image)

            # Convert msg to numpy image.
            image_np = numpify(image_msg)
            

            # image_np = img = cv2.imread(
            #     '/home/zaferpc/TensorFlow/workspace/training_demo/images/test/1.jpg')
            original_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            self.image = original_img
            
            while not self.cropped:
                # # Wait for rgb camera stream to publish a frame.
                image_msg = rospy.wait_for_message('/camera/color/image_raw', Image)

                # Convert msg to numpy image.
                image_np = numpify(image_msg)
                

                # image_np = img = cv2.imread(
                #     '/home/zaferpc/TensorFlow/workspace/training_demo/images/test/1.jpg')
                original_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                self.image = original_img
                cv2.imshow("image", self.image)
                cv2.waitKey(20)
                if self.cropped:
                    cv2.destroyWindow("image")
                    break
                
            self.image = self.image[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
            output = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY).copy()
            img = self.image.copy()

            ###########################################
            # Retrieve Tunning Bars Values            #
            ###########################################

            if self.use_canny:
                thresh1 = cv2.getTrackbarPos('thresh1', 'output_window')
                thresh2 = cv2.getTrackbarPos('thresh2', 'output_window')

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

            # create a CLAHE object (Arguments are optional).
            clahe = cv2.createCLAHE(clipLimit=c, tileGridSize=(2, 2))
            output = clahe.apply(output)

            output = cv2.GaussianBlur(output, (k, k), 0)

            # ret, output = cv2.threshold(output, thresh, 255, cv2.THRESH_BINARY_INV)
            output = cv2.adaptiveThreshold(output, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY_INV, 11, 2)

            if self.use_canny:
                output = cv2.Canny(output, thresh1, thresh2)

            # kernel = np.ones((2, 2), np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
            # Use erosion and dilation combination to eliminate false positives.
            output = cv2.dilate(output, kernel, iterations=1)
            # output = cv2.erode(output, kernel, iterations=1)

            output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel, iterations=1)

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

            cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

            # if len(contours) > 0:

            #     x, y, w, h = cv2.boundingRect(contours[0])
            #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # rect = cv2.minAreaRect(contours[0])
                # box = cv2.boxPoints(rect)
                # box = np.int0(box)
                # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

            # ========================================================= #

            cv2.imshow("output_window", output)
            # cv2.imshow("thresh_window", thresh)
            cv2.imshow("image_window", img)

            key = cv2.waitKey(20) & 0xFF

            if key == ord('e'):
                break
            # elif key == ord('c'):
            #     image_id += 1
            #     if image_id < dset_sz:
            #         original_img = read_and_resize(data_dir, image_id)
            #         img = original_img.copy()
            #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #     else:
            #         break

        # close all open windows
        cv2.destroyAllWindows()
        

    def click_and_crop(self, event, x, y, flags, param):
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that self.cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
            self.cropping = True
        # if event == cv2.EVENT_MOUSEMOVE:
        #     if self.cropping:
        #         # draw a rectangle around the region of interest
        #         cv2.rectangle(self.image, self.refPt[0], (x, y), (0, 255, 0), 2)
        #         cv2.imshow("image", self.image)
        #         cv2.waitKey(1000)
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the self.cropping operation is finished
            self.refPt.append((x, y))
            self.cropping = False
            self.cropped = True
            # draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", self.image)
            cv2.waitKey(0)


if __name__ == "__main__":
    fc = FindCorner()
    fc.find_corners()
    