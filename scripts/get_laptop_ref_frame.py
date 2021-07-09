#!/usr/bin/env python
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from copy import deepcopy
import ros_numpy
from std_msgs.msg import String, Float32MultiArray
from realsense2_camera.msg import Extrinsics
from perception.laptop_perception_helpers import detect_laptop, calculate_dist_3D, transform_depth_to_color_frame, constrain_environment
    
    

class LaptopPoseDetector:
    def __init__(self):
        self.aligned_depth_img = None
        # rospy.Subscriber(
        #     "/camera/aligned_depth_to_color/camera_info", CameraInfo, self.info_callback)
        self.img_publisher = rospy.Publisher("/img", Image, queue_size=1)
        self.laptop_pose_publisher = rospy.Publisher("/laptop_pose", Float32MultiArray, queue_size=1)
        self.min_x, self.max_x = 1802, 2212
        self.min_y, self.max_y = 0, 2125
        self.min_z, self.max_z = 2249, 2349
        
    def _tune_pose_detector(self):

        win_name = 'constrained environment'
        cv2.namedWindow(win_name)
        cv2.createTrackbar('min_x', 'distance_tuning', self.min_x, 4000, lambda x: None)
        cv2.createTrackbar('max_x', 'distance_tuning', self.max_x, 4000, lambda x: None)
        cv2.createTrackbar('min_y', 'distance_tuning', self.min_y, 4000, lambda x: None)
        cv2.createTrackbar('max_y', 'distance_tuning', self.max_y, 4000, lambda x: None)
        cv2.createTrackbar('min_z', 'distance_tuning', self.min_z, 4000, lambda x: None)
        cv2.createTrackbar('max_z', 'distance_tuning', self.max_z, 4000, lambda x: None)

        key = 0
        while not rospy.is_shutdown() and key != ord('e'):
            
            min_x = (0.001 * cv2.getTrackbarPos('min_x', 'distance_tuning') - 2)
            max_x = (0.001 * cv2.getTrackbarPos('max_x', 'distance_tuning') - 2)
            min_y = (0.001 * cv2.getTrackbarPos('min_y', 'distance_tuning') - 2)
            max_y = (0.001 * cv2.getTrackbarPos('max_y', 'distance_tuning') - 2)
            min_z = (0.001 * cv2.getTrackbarPos('min_z', 'distance_tuning') - 2)
            max_z = (0.001 * cv2.getTrackbarPos('max_z', 'distance_tuning') - 2)
            
            # color_img_msg = rospy.wait_for_message(
            #         "/camera/color/image_raw", Image)
            # color_img = ros_numpy.numpify(color_img_msg)
            # color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            # print("received Image")
            depth_img_msg = rospy.wait_for_message("/camera/depth/image_rect_raw", Image)
            depth_img = ros_numpy.numpify(depth_img_msg)
            intrinsics = self.get_intrinsics()
            dist_mat = calculate_dist_3D(depth_img, intrinsics)
            dist_mat = transform_depth_to_color_frame(dist_mat)
            dist_image = constrain_environment(deepcopy(dist_mat),
                                               x_lim=(self.x_min, self.x_max),
                                               y_lim=(self.y_min, self.y_max),
                                               z_lim=(self.z_min, self.z_max))
            
            # img_msg = ros_numpy.msgify(Image, color_img, encoding='bgr8')
            img_msg = ros_numpy.msgify(Image, dist_image, encoding='mono8')

            self.img_publisher.publish(img_msg)
            laptop_data_msg = rospy.wait_for_message("/laptop_data", Float32MultiArray) # center, flip_point
        
            # show converted depth image
            cv2.imshow(win_name, dist_image)
            key = cv2.waitKey(10) & 0xFF

        if self.tune:
            cv2.destroyWindow(win_name)
    
    def get_intrinsics(self):
        intrinsics = {'fx': 0, 'fy': 0, 'px': 0, 'py': 0, 'w': 0, 'h': 0}
        msg = rospy.wait_for_message("/camera/depth/camera_info", CameraInfo)
        intrinsics['fx'] = msg.K[0]
        intrinsics['fy'] = msg.K[4]
        intrinsics['px'] = msg.K[2]
        intrinsics['py'] = msg.K[5]
        intrinsics['w'] = msg.width
        intrinsics['h'] = msg.height
        return intrinsics
    
    def detect_pose_callback(self, msg):
        dist_image = self.constrain_environment()
        laptop_data = detect_laptop(dist_image)
        laptop_data_msg = Float32MultiArray()
        laptop_data_msg.data = laptop_data
        self.laptop_pose_publisher.publish(laptop_data_msg)
        


if __name__ == "__main__":
    rospy.init_node("frame_detector")
    laptop_pose_detector = LaptopPoseDetector()
    laptop_pose_detector._tune_pose_detector()
    rospy.spin()