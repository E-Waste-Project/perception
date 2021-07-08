#!/usr/bin/env python
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from copy import deepcopy


class FrameDetector:
    def __init__(self):
        self.aligned_depth_img = None
        self.intrinsics = {'fx': 0, 'fy': 0, 'px': 0, 'py': 0, 'w': 0, 'h': 0}
        rospy.Subscriber(
            "/camera/aligned_depth_to_color/camera_info", CameraInfo, self.info_callback)

        aligned_depth_msg = rospy.wait_for_message(
            "/camera/aligned_depth_to_color/image_raw", Image)
        print("received Image")
        dist_mat = self.calculate_dist_3D(aligned_depth_msg)
        dist_image = deepcopy(dist_mat[2]) * 255 / np.max(dist_mat[2])
        dist_image = dist_image.astype(np.uint8)
        cv2.imshow("depth_image", dist_image)
        cv2.waitKey(0)
    
    def info_callback(self, msg):
        self.intrinsics['fx'] = msg.K[0]
        self.intrinsics['fy'] = msg.K[4]
        self.intrinsics['px'] = msg.K[2]
        self.intrinsics['py'] = msg.K[5]
        self.intrinsics['w'] = msg.width
        self.intrinsics['h'] = msg.height


if __name__ == "__main__":
    frame_detector = FrameDetector()
    rospy.spin()