#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import ros_numpy

class Subscribers():
	def __init__(self):
		self.aligned_depth_img = None
		self.intrinsics = {'fx':0, 'fy':0, 'px':0, 'py':0, 'w':0, 'h':0}
		self.dist_mat = None
		rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.aligned_callback)
		rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.info_callback)

	def aligned_callback(self, msg):
		aligned_depth_image = ros_numpy.numpify(msg) * 0.001
		index_mat=np.indices(aligned_depth_image.shape)
		self.dist_mat[0] = (index_mat[0] - self.intrinsics['px']) * aligned_depth_image / self.intrinsics['fx']
		self.dist_mat[1] = (index_mat[1] - self.intrinsics['py']) * aligned_depth_image / self.intrinsics['fy']
		self.dist_mat[2] = aligned_depth_image
		print(self.dist_mat[:, self.intrinsics['h'] // 2, self.intrinsics['w'] // 2])

	def info_callback(self, msg):
		self.intrinsics['fx'] = msg.K[0]
		self.intrinsics['fy'] = msg.K[4]
		self.intrinsics['px'] = msg.K[2]
		self.intrinsics['py'] = msg.K[5]
		self.intrinsics['w'] = msg.width
		self.intrinsics['h'] = msg.height
		self.dist_mat = np.zeros((3, self.intrinsics['h'], self.intrinsics['w']))
		#print(self.intrinsics)

if __name__=="__main__":
	rospy.init_node("vision_node")
	subs = Subscribers()
	rospy.spin()
