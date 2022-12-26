#!/usr/bin/env python
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from copy import deepcopy
import ros_numpy
from std_msgs.msg import String, Float32MultiArray
from realsense2_camera.msg import Extrinsics
from perception.laptop_perception_helpers import RealsenseHelpers, detect_laptop_pose, constrain_environment
    
    

class LaptopPoseDetector:
    def __init__(self):
        self.aligned_depth_img = None
        # rospy.Subscriber("/get_laptop_pose", String, self.detect_pose_callback)
        self.img_publisher = rospy.Publisher("/img", Image, queue_size=1)
        # self.laptop_pose_publisher = rospy.Publisher("/laptop_pose", Float32MultiArray, queue_size=1)
        self.trackbar_limits = {'x_min': 1802, 'x_max': 4000,
                                'y_min': 1763   , 'y_max': 3674,
                                'z_min': 0, 'z_max': 2258}
        limits = {'x_min': 0.46799999999999997, 'x_max': 1.157, 'y_min': -0.08899999999999997, 'y_max': 2.0, 'z_min': -0.7050000000000001, 'z_max': 0.6760000000000002}
        for key, val in limits.items():
            self.trackbar_limits[key] = int(1000 * (val + 2))
        self.limits = {}
        for key, val in self.trackbar_limits.items():
            self.limits[key] = 0.001* val - 2
            # if key in ['z_min', 'z_max']:
            #     self.limits[key] += 2
        self.cam_helpers = RealsenseHelpers()
        
    def _tune_pose_detector(self, detect_laptop=True):

        win_name = 'constrained environment'
        cv2.namedWindow(win_name)
        for key, val in self.trackbar_limits.items():
            cv2.createTrackbar(key, win_name, val, 4000, lambda x: None)

        key = 0
        while not rospy.is_shutdown() and key != ord('e'):
            
            for key in self.trackbar_limits.keys():
                self.limits[key] = (0.001 * cv2.getTrackbarPos(key, win_name) - 2)
                # if key in ['z_min', 'z_max']:
                #     self.limits[key] += 2
            print(self.limits)
                    
            # color_img_msg = rospy.wait_for_message(
            #         "/camera/color/image_raw", Image)
            # color_img = ros_numpy.numpify(color_img_msg)
            # color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            # print("received Image")
            
            # Get current distances of all pixels from the depth image.
            dist_mat = self.cam_helpers.get_dist_mat_from_cam(transform_to_color=True)
            # print(dist_mat.shape)
            # Detect the laptop pose data (laptop_center, flipping_point, upper_point) as pixels
            if detect_laptop:
                laptop_data_px , dist_image = detect_laptop_pose(dist_mat, draw=True,
                                                                    x_min=self.limits['x_min'], x_max=self.limits['x_max'],
                                                                    y_min=self.limits['y_min'], y_max=self.limits['y_max'],
                                                                    z_min=self.limits['z_min'], z_max=self.limits['z_max'])
                if laptop_data_px is not None:
                    # Deproject the pixels representing laptop pose data to xyz 3d pose data. 
                    laptop_pose_data = self.cam_helpers.px_to_xyz(laptop_data_px, dist_mat=dist_mat)
                    print(dist_image.shape)
                    img_msg = ros_numpy.msgify(Image, dist_image, encoding='bgr8')
                    # img_msg = ros_numpy.msgify(Image, dist_image, encoding='mono8')
                    
                    self.img_publisher.publish(img_msg)
                    # laptop_data_msg = rospy.wait_for_message("/laptop_data", Float32MultiArray) # center, flip_point
            else:
                dist_image = constrain_environment(
                    deepcopy(dist_mat),
                    x_lim=(self.limits['x_min'], self.limits['x_max']),
                    y_lim=(self.limits['y_min'], self.limits['z_max']),
                    z_lim=(self.limits['z_min'], self.limits['z_max']),
                    target_frame='aruco_base',
                    src_frame='camera_color_optical_frame',
                    transform_points=True
                )
            
            # show converted depth image
            cv2.imshow(win_name, dist_image)
            key = cv2.waitKey(10) & 0xFF

        cv2.destroyWindow(win_name)


if __name__ == "__main__":
    rospy.init_node("frame_detector")
    laptop_pose_detector = LaptopPoseDetector()
    laptop_pose_detector._tune_pose_detector(detect_laptop=False)
    rospy.spin()
