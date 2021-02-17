#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import ros_numpy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

rospy.init_node("select_node")
point_publisher = rospy.Publisher("/cutting_path",Float32MultiArray)

image_msg = rospy.wait_for_message("/camera/color/image_raw",Image)
image_mat = ros_numpy.numpify(image_msg)

cv2.imshow("Image_Window",image_mat)

pub_msg = Float32MultiArray()
pub_msg.data = []

for i in range(0,4):
	(y,x,w,h) = cv2.selectROI("Image_window",image_mat)
	pub_msg.data.append(x)
	pub_msg.data.append(y)
	print("x=",x,"y=",y)

point_publisher.publish(pub_msg)
cv2.waitKey(0)
cv2.destroyAllWindows()
