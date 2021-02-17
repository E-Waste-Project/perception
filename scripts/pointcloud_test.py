#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
import ros_numpy


class Subscribers():
    def __init__(self):
        self.aligned_depth_img = None
        self.intrinsics = {'fx': 0, 'fy': 0, 'px': 0, 'py': 0, 'w': 0, 'h': 0}
        rospy.Subscriber(
            "/camera/aligned_depth_to_color/camera_info", CameraInfo, self.info_callback)
        rospy.Subscriber("/cutting_path", Float32MultiArray,
                         self.cutting_callback)
        self.cut_xyz_pub = rospy.Publisher("/cut_xyz", PoseArray, queue_size=1)

    def calculate_dist_3D(self, msg):
        aligned_depth_image = ros_numpy.numpify(msg) * 0.001
        index_mat = np.indices(aligned_depth_image.shape)
        print(index_mat.shape)
        dist_mat = np.zeros((3, self.intrinsics['h'], self.intrinsics['w']))
        dist_mat[0] = (index_mat[0] - self.intrinsics['py']) * \
            aligned_depth_image / self.intrinsics['fy']
        dist_mat[1] = (index_mat[1] - self.intrinsics['px']) * \
            aligned_depth_image / self.intrinsics['fx']
        dist_mat[2] = aligned_depth_image
        #print(dist_mat[:, self.intrinsics['h'] // 2, self.intrinsics['w'] // 2])
        return dist_mat

    def cutting_callback(self, msg):
        print("callback")
        aligned_depth_msg = rospy.wait_for_message(
            "/camera/aligned_depth_to_color/image_raw", Image)
        print("received Image")
        dist_mat = self.calculate_dist_3D(aligned_depth_msg)
        contour_indices = np.array(msg.data, dtype=np.uint16)
        contour_indices = contour_indices.reshape((len(msg.data)//2, 2))
        #print(contour_indices)
        contour_xyz = dist_mat[:, contour_indices[:, 0], contour_indices[:, 1]]
        pose_msg = PoseArray()
        pose_msg.header.frame_id = "camera_start"
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.poses = []
        print(contour_xyz.shape)
        print(contour_xyz)
        for point_num in range(0, contour_indices[:, 0].shape[0]):
            pose = Pose()
            pose.position.x = contour_xyz[1, point_num]
            pose.position.y = contour_xyz[0, point_num]
            pose.position.z = contour_xyz[2, point_num]
            # pose.orientation.w = 1
            pose_msg.poses.append(pose)
        self.cut_xyz_pub.publish(pose_msg)

    def info_callback(self, msg):
        self.intrinsics['fx'] = msg.K[0]
        self.intrinsics['fy'] = msg.K[4]
        self.intrinsics['px'] = msg.K[2]
        self.intrinsics['py'] = msg.K[5]
        self.intrinsics['w'] = msg.width
        self.intrinsics['h'] = msg.height
        #print(self.intrinsics)


if __name__ == "__main__":
    rospy.init_node("dist3d_calculator")
    subs = Subscribers()
    rospy.spin()
