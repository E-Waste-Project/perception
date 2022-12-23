#!/usr/bin/env python
from os import read
from time import sleep
import rospy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, String
from ros_numpy import numpify
from perception.laptop_perception_helpers import RealsenseHelpers
from robot_helpers.srv import TransformPoses, CreateFrameAtPose, LookupTransform
import numpy as np
import cv2 as cv


class Calibrate:
    def __init__(self, read_img=False, publish_topic='/chess_corners'):
        self.image = None
        self.publisher = rospy.Publisher(publish_topic,
                                         Float32MultiArray,
                                         queue_size=1)
        self.rs_helpers = RealsenseHelpers()
        self.image_topic = self.rs_helpers.color_image_topic
        self.read_img = read_img
        self.cam_points = []
        self.rob_points = []
        self.joint_states = []
        self.recieve_img()

    def recieve_img(self):
        if self.read_img:
            self.image = cv.imread("/home/ubuntu/Pictures/pattern.png")
            self.image = cv.resize(self.image, (1280, 720))
        else:
            # # Wait for rgb camera stream to publish a frame.e
            image_msg = rospy.wait_for_message(self.image_topic, Image)
            # Convert msg to numpy image.
            self.image = numpify(image_msg)
    
    def detect_corners(self, size=(7, 6)):
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Arrays to store object points and image points from all the images.
        img = self.image.copy()
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # cv.imshow('img', gray)
        # cv.waitKey(0)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, size, None)
        # If found, add object points, image points (after refining them)
        # print(ret)
        if ret == True:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # print(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, size, corners2, ret)
            corner_points = []
            for i in range(corners2.shape[0]):
                # cv.circle(img, tuple(corners2[i][0]), 5, (0, 255, 0), 3)
                corner_points.append(tuple(corners2[i][0]))
            self.publish_points(corner_points)
            cv.imshow('img', img)
            if cv.waitKey(10) & 0xFF == ord('q'):
                self.add_points(corner_points)
            # cv.destroyAllWindows()

    def publish_points(self, points):
        # Publish Cutting Path.
        path_msg = Float32MultiArray()
        for x, y in points:
            path_msg.data.append(y)
            path_msg.data.append(x)
        self.publisher.publish(path_msg)
    
    def add_points(self, corners):
        # corners_np = np.array(corners)
        transformation_data = {
            "target_frame": String("tool0"),
            "source_frame": String("base_link")
        }
        response = self.service_req(
            "/lookup_transform", LookupTransform, inputs=transformation_data
        )
        ee_frame = response.frame_pose
        self.rob_points.append([ee_frame.position.x,
                                ee_frame.position.y,
                                ee_frame.position.z,])
        joints_msg = rospy.wait_for_message("/joint_states",JointState)
        self.joint_states.append(joints_msg.position)
        # x_min = int(round(min([x for x, y in corners])))
        # y_min = int(round(min([y for x, y in corners])))
        x_min = int(round(corners[-5][0]))
        y_min = int(round(corners[-5][1]))
        xyz_list = self.rs_helpers.px_to_xyz([(y_min, x_min)])
        self.cam_points.append(xyz_list)
        self.append_new_line()
        print("added point")
        
    
    def service_req(self, name, service_type, **inputs):
        _ = rospy.wait_for_service(name)
        try:
            callable_service_func = rospy.ServiceProxy(name, service_type)
            response = callable_service_func(**inputs["inputs"])
            return response
        except rospy.ServiceException as e:
            print("Service Failed : {}".format(e))
            
    
    def append_new_line(self):
        """Append given text as a new line at the end of file"""
        # Open the file in append & read mode ('a+')
        with open("/home/bass/ur5_rob_ws/src/calibration/rob.txt", "a+") as file_object:
            # Move read cursor to the start of file.
            file_object.seek(0)
            # If file is not empty then append '\n'
            data = file_object.read(100)
            if len(data) > 0:
                file_object.write("\n")
            # Append text at the end of file
            file_object.write(','.join(str(j) for j in self.rob_points[-1]))
        with open("/home/bass/ur5_rob_ws/src/calibration/cam.txt", "a+") as file_object:
            # Move read cursor to the start of file.
            file_object.seek(0)
            # If file is not empty then append '\n'
            data = file_object.read(100)
            if len(data) > 0:
                file_object.write("\n")
            # Append text at the end of file
            file_object.write(','.join(str(j) for j in self.cam_points[-1]))
        with open("/home/bass/ur5_rob_ws/src/calibration/joints.txt", "a+") as file_object:
            # Move read cursor to the start of file.
            file_object.seek(0)
            # If file is not empty then append '\n'
            data = file_object.read(100)
            if len(data) > 0:
                file_object.write("\n")
            # Append text at the end of file
            file_object.write(','.join(str(j) for j in self.joint_states[-1]))


if __name__ == '__main__':
    rospy.init_node("calibration_test")
    sleep(1)
    calib = Calibrate(read_img=False)
    while not rospy.is_shutdown():
        calib.recieve_img()
        calib.detect_corners(size=(5, 8))
        # print("Input Any Key to Redetect")
        # input()
    print(calib.rob_points)
    print(calib.cam_points)
    print(calib.joint_states)
    cv.destroyAllWindows()
