#!/usr/bin/env python
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from copy import deepcopy
import ros_numpy
from std_msgs.msg import String, Float32MultiArray
from realsense2_camera.msg import Extrinsics
from perception.laptop_perception_helpers import detect_laptop, calculate_dist_3D,\
     transform_depth_to_color_frame, constrain_environment, get_intrinsics
    
    

class LaptopPoseDetector:
    def __init__(self):
        self.aligned_depth_img = None
        rospy.Subscriber("/get_laptop_pose", String, self.detect_pose_callback)
        self.img_publisher = rospy.Publisher("/img", Image, queue_size=1)
        self.laptop_pose_publisher = rospy.Publisher("/laptop_pose", Float32MultiArray, queue_size=1)
        self.trackbar_limits = {'min_x': 1802, 'max_x': 2212,
                                'min_y': 0   , 'max_y': 2125,
                                'min_z': 2249, 'max_z': 2349}
        self.limits = {}
        for key, val in self.trackbar_limits.items():
            self.limits[key] = 0.001* val - 2
        
    def _tune_pose_detector(self):

        win_name = 'constrained environment'
        cv2.namedWindow(win_name)
        for key, val in self.trackbar_limits.items():
            cv2.createTrackbar(key, win_name, val, 4000, lambda x: None)

        key = 0
        while not rospy.is_shutdown() and key != ord('e'):
            
            for key in self.trackbar_limits.keys():
                self.limits[key] = (0.001 * cv2.getTrackbarPos(key, win_name) - 2)
            
            # color_img_msg = rospy.wait_for_message(
            #         "/camera/color/image_raw", Image)
            # color_img = ros_numpy.numpify(color_img_msg)
            # color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            # print("received Image")
            dist_image, dist_mat = self.detect_pose()
            color_dist_image = cv2.cvtColor(dist_image, cv2.COLOR_GRAY2BGR)
            laptop_data_px = detect_laptop(dist_image, draw_on=color_dist_image)
            
            # img_msg = ros_numpy.msgify(Image, color_img, encoding='bgr8')
            img_msg = ros_numpy.msgify(Image, dist_image, encoding='mono8')

            self.img_publisher.publish(img_msg)
            # laptop_data_msg = rospy.wait_for_message("/laptop_data", Float32MultiArray) # center, flip_point
        
            # show converted depth image
            cv2.imshow(win_name, color_dist_image)
            key = cv2.waitKey(10) & 0xFF

        cv2.destroyWindow(win_name)
    
    def detect_pose(self):
        depth_img_msg = rospy.wait_for_message("/camera/depth/image_rect_raw", Image)
        depth_img = ros_numpy.numpify(depth_img_msg)
        intrinsics = get_intrinsics()
        dist_mat = calculate_dist_3D(depth_img, intrinsics)
        dist_mat = transform_depth_to_color_frame(dist_mat)
        dist_image = constrain_environment(deepcopy(dist_mat),
                                            x_lim=(self.limits['min_x'], self.limits['max_x']),
                                            y_lim=(self.limits['min_y'], self.limits['max_y']),
                                            z_lim=(self.limits['min_z'], self.limits['max_z']))
        return dist_image, dist_mat
    
    def detect_pose_callback(self, msg):
        dist_image, dist_mat = self.detect_pose()
        laptop_data_px = detect_laptop(dist_image)
        laptop_data_xyz = []
        for i in range(0, len(laptop_data_px), 2):
            x_px = laptop_data_px[i]
            y_px = laptop_data_px[i+1]
            xyz = dist_mat[:, y_px, x_px].tolist()
            laptop_data_xyz.extend(xyz)
        laptop_data_msg = Float32MultiArray()
        laptop_data_msg.data = laptop_data_xyz
        self.laptop_pose_publisher.publish(laptop_data_msg)
        


if __name__ == "__main__":
    rospy.init_node("frame_detector")
    laptop_pose_detector = LaptopPoseDetector()
    laptop_pose_detector._tune_pose_detector()
    rospy.spin()
