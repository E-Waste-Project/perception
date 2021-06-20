#!/usr/bin/env python
# from moveit_commander import MoveGroupCommander
# from moveit_msgs.msg import RobotTrajectory
# from moveit_msgs.msg import MoveGroupActionFeedback
from actionlib_msgs.msg import GoalStatusArray, GoalStatus

import cv2
import matplotlib.pyplot as plt
from perception.yolo_detector import Yolo
from perception.coco_datasets import convert_format
from perception.laptop_perception_helpers import plan_cover_cutting_path, interpolate_path,\
    plan_port_cutting_path, filter_boxes_from_image,\
    draw_lines, draw_boxes, box_near_by_dist
import sys
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from ros_numpy import numpify
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from copy import deepcopy
# from tf.transformations import euler_from_quaternion, quaternion_matrix, quaternion_from_matrix, quaternion_from_euler
from math import sqrt, fabs
import numpy as np
import tensorflow as tf
import time
import socket
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

user = socket.gethostname()
ws = 'ewaste_ws' if user == 'abdelrhman' else 'abb_ws'

window_index = 0
detection_index = 0
blue_button_pose = Pose()
battery_cover_pose = Pose()
red_button_pose = Pose()
board_state = None
# camera_group = MoveGroupCommander("camera_link")
# transformer_listener = tf.TransformListener()

def spiral(X, Y, center):
    x = y = 0
    dx = 0
    dy = -1
    yaxis = list()
    xaxis = list()
    spiral_result = list()
    for i in range(max(X, Y)**2):
        if (x-X/2 < x <= x+X/2) and (y-Y/2 < y <= y+Y/2):
            yaxis.append(y+center[1])
            xaxis.append(x+center[0])
            
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy
    [spiral_result.extend([yaxis[i], xaxis[i]]) for i in range((len(yaxis)))]
    # print(spiral_result)
    # plt.plot(xaxis, yaxis, marker='o')
    # plt.show()
    return spiral_result


def generate_spiral(X, Y, ref_pose, ref_frame="capture_frame", step=1):
    x = y = 0
    dx = 0
    dy = -step
    pose_array = PoseArray()
    pose_array.header.frame_id = ref_frame
    X_mm = int(X * 1000)
    Y_mm = int(Y * 1000)
    for i in range(max(X_mm, Y_mm)**2):
        if (x-X_mm/2 < x <= x+X_mm/2) and (y-Y_mm/2 < y <= y+Y_mm/2):
            pose = Pose()
            pose.position.x = ref_pose.position.x + (x / 1000.0)
            pose.position.y = ref_pose.position.y + (y / 1000.0)
            pose.position.z = ref_pose.position.z
            pose.orientation.x = ref_pose.orientation.x
            pose.orientation.y = ref_pose.orientation.y
            pose.orientation.z = ref_pose.orientation.z
            pose.orientation.w = ref_pose.orientation.w
            pose_array.poses.append(pose)
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy
    
    return pose_array


#yaw = z, pitch = Y , Roll = X
def angle_to_quaternion(yaw,pitch,roll):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return w , x , y , z

#uses battery centers and taskboard center to determine the battery orientation 
def get_battery_orientation(battery_center, task_board_center):
    battery_difference_0 = abs(task_board_center[1]-battery_center[0][1])
    battery_difference_1 = abs(task_board_center[1]-battery_center[1][1])
    
    if battery_difference_0 > battery_difference_1:
        battery_left_center = battery_center[0]
        battery_right_center = battery_center[1]
    else:
        battery_left_center = battery_center[1]
        battery_right_center = battery_center[0]
    
    return battery_left_center , battery_right_center

def send_to_pct(pixels_msg):
    global pixel_to_cartesian_pub
    global operations_pub
    global board_state
    pixel_to_cartesian_pub.publish(pixels_msg)
    rospy.sleep(1)
    operations_msg = String()
    operations_msg.data = "Competition"
    operations_pub.publish(operations_msg)
    cartesian_msg = PoseArray()
    print("wait_xyz")
    cartesian_msg = rospy.wait_for_message("/competition_xyz",PoseArray)
    print("received_xyz")
    return cartesian_msg


def capture_callback(msg):
    # Recieve an image msg from camera topic, then, return image and detections.
    global window_index
    global camera_group
    global detection_index
    global board_state
    global blue_button_pose
    global battery_cover_pose
    global red_button_pose
    window_index = window_index + 1 
    if msg.data == 'Orient':
        detection_index = 0 # Changed
        #detect and show image
        image, detections = model_orientation.recieve_and_detect()
        print(detections)
        cv2.startWindowThread()
        cv2.namedWindow("Captured Image")
        cv2.imshow('Captured Image', image)
        cv2.waitKey(7000)
        cv2.destroyAllWindows()
        
        #detect pixel values
        blue_button_px = model_orientation.get_class_detections(detections,'Blue_Button')
        key_hole_px = model_orientation.get_class_detections(detections,'Key_hole')
        task_board_px = model_orientation.get_class_detections(detections,'Task_board')
        blue_button_center = [round(0.5 * (blue_button_px[0][0]+blue_button_px[0][2])),round(0.5 * (blue_button_px[0][1]+blue_button_px[0][3]))]
        key_hole_center = [round(0.5 * (key_hole_px[0][0]+key_hole_px[0][2])),round(0.5 * (key_hole_px[0][1]+key_hole_px[0][3]))]
        task_board_center = [round(0.5 * (task_board_px[0][0]+task_board_px[0][2])),round(0.5 * (task_board_px[0][1]+task_board_px[0][3]))]
        task_board_line = [[task_board_px[0][2],task_board_px[0][1]],[task_board_px[0][2],task_board_px[0][3]]]
        image_center = [1280//2 , 720//2]
        
        #send pixel values
        pixel_to_cartesian_msg = Float32MultiArray()
        pixel_to_cartesian_msg.data = [blue_button_center[1],blue_button_center[0],
                                       key_hole_center[1],key_hole_center[0],
                                       task_board_center[1],task_board_center[0],
                                       task_board_line[0][1],task_board_line[0][0],
                                       task_board_line[1][1],task_board_line[1][0],
                                       image_center[1],image_center[0]]
        
        cartesian_msg = send_to_pct(pixel_to_cartesian_msg)
        
        #Data Retrieval
        blue_button_position = cartesian_msg.poses[0].position
        key_hole_pose = cartesian_msg.poses[1].position
        task_board_center_pose = cartesian_msg.poses[2].position
        task_board_line_pose1 = cartesian_msg.poses[3].position
        task_board_line_pose2 = cartesian_msg.poses[4].position
        image_center_pose = cartesian_msg.poses[5].position
        
        #Vector Calculations
        if key_hole_pose.y > blue_button_position.y:
            button_hole_vector = np.array([key_hole_pose.x - blue_button_position.x , key_hole_pose.y - blue_button_position.y])
            #task_board_vector = np.array([task_board_line_pose1.x - task_board_line_pose2.x  , task_board_line_pose1.y - task_board_line_pose2.y])
        elif key_hole_pose.y < blue_button_position.y:
            button_hole_vector = np.array([blue_button_position.x - key_hole_pose.x , blue_button_position.y - key_hole_pose.y])
            #task_board_vector = np.array([task_board_line_pose2.x - task_board_line_pose1.x  , task_board_line_pose2.y - task_board_line_pose1.y])
        #task_board_vector = np.array([task_board_line_pose1.x - task_board_line_pose2.x  , task_board_line_pose1.y - task_board_line_pose2.y])
        task_board_vector = np.array([ -1 , 0])
        
        print("button_hole_vector =",  button_hole_vector)
        print("task_board_vector =" , task_board_vector)
        
        #Transformation Calculation
        rotation_angle = np.arccos(np.sum(button_hole_vector*task_board_vector)/(np.linalg.norm(button_hole_vector)*np.linalg.norm(task_board_vector)))
        if(button_hole_vector[0]<0):
            rotation_angle = (((np.pi * 28.68/180) - rotation_angle) )
        else:
            rotation_angle = (((np.pi * 28.68/180) - rotation_angle) ) + np.pi
        print("Rotation Angle =",rotation_angle*(180/np.pi))
        translation_vector = np.array([task_board_center_pose.x - image_center_pose.x , task_board_center_pose.y - image_center_pose.y]) 
        
        #Get Pose with respect to calibrated frame
        camera_detect_pose = PoseStamped()
        camera_detect_pose.header.frame_id = 'calibrated_frame'
        camera_detect_pose.pose.position.x = translation_vector [0]
        camera_detect_pose.pose.position.y = translation_vector [1]
        camera_detect_pose.pose.position.z = 0.05
        w , x , y , z = angle_to_quaternion(rotation_angle,0,0)
        camera_detect_pose.pose.orientation.w = w
        camera_detect_pose.pose.orientation.x = x
        camera_detect_pose.pose.orientation.y = y
        camera_detect_pose.pose.orientation.z = z
        
        desired_pose_pub.publish(camera_detect_pose)
        print("pub_poses")
        return
        
        
    elif msg.data == 'Detect':
        #cv2.destroyWindow('Captured Image'+str(window_index-1))
        if detection_index == 0:
            detection_index = 1
            image, detections = model_detection.recieve_and_detect()
            
            # cv2.startWindowThread()
            # cv2.namedWindow("Captured Image")
            # cv2.imshow('Captured Image', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            # blue_button_px = model_detection.get_class_detections(detections,'Blue_Button')
            red_button_px = model_detection.get_class_detections(detections,'Red_Button')
            ethernet_cable_px = model_detection.get_class_detections(detections,'Ethernet_Cable')
            # key_px = model_detection.get_class_detections(detections,'Key')
            key_hole_px = model_detection.get_class_detections(detections,'Key_hole')
            battery_cover_px = model_detection.get_class_detections(detections,'Battery_Cover')
            # cover_pressure_point_px = model_detection.get_class_detections(detections,'Cover_Pressure_Point')
            # ethernet_empty_px = model_detection.get_class_detections(detections,'Ethernet_Empty')
            
            
            # blue_button_center = [round(0.5 * (blue_button_px[0][0]+blue_button_px[0][2])),round(0.5 * (blue_button_px[0][1]+blue_button_px[0][3]))]
            red_button_center = [round(0.5 * (red_button_px[0][0]+red_button_px[0][2])),round(0.5 * (red_button_px[0][1]+red_button_px[0][3]))]
            ethernet_cable_center = [round(0.5 * (ethernet_cable_px[0][0]+ethernet_cable_px[0][2])),round(0.5 * (ethernet_cable_px[0][1]+ethernet_cable_px[0][3]))]
            # key_center = [round(0.5 * (key_px[0][0]+key_px[0][2])),round(0.5 * (key_px[0][1]+key_px[0][3]))]
            key_hole_center = [round(0.5 * (key_hole_px[0][0]+key_hole_px[0][2])),round(0.5 * (key_hole_px[0][1]+key_hole_px[0][3]))]
            battery_cover_center = [round(0.5 * (battery_cover_px[0][0]+battery_cover_px[0][2])),round(0.5 * (battery_cover_px[0][1]+battery_cover_px[0][3]))]
            # cover_pressure_point_center = [round(0.5 * (cover_pressure_point_px[0][0]+cover_pressure_point_px[0][2])),round(0.5 * (cover_pressure_point_px[0][1]+cover_pressure_point_px[0][3]))]
            # ethernet_empty_center = [round(0.5 * (ethernet_empty_px[0][0]+ethernet_empty_px[0][2])),round(0.5 * (ethernet_empty_px[0][1]+ethernet_empty_px[0][3]))]
            
            cv2.circle(image, tuple(key_hole_center), 1, (0, 255, 0), thickness=1)
            # cv2.circle(image, tuple(key_center), 1, (0, 255, 0), thickness=1)
            
            print("SHOWWWWWW")
            cv2.startWindowThread()
            cv2.namedWindow("Captured Image")
            cv2.imshow("Captured Image", image)
            cv2.waitKey(7000)
            cv2.destroyAllWindows()
            print("DONEEEEEE")
            
            pixel_to_cartesian_msg = Float32MultiArray()
            pixel_to_cartesian_msg.data = [#blue_button_center[1],blue_button_center[0],
                                        red_button_center[1],red_button_center[0],
                                        ethernet_cable_center[1],ethernet_cable_center[0],
                                        # key_center[1],key_center[0],
                                        key_hole_center[1],key_hole_center[0],
                                        battery_cover_center[1],battery_cover_center[0],
                                        # cover_pressure_point_center[1],cover_pressure_point_center[0],
                                        #ethernet_empty_center[1],ethernet_empty_center[0]
                                        ]
            
            cartesian_msg = send_to_pct(pixel_to_cartesian_msg)
            
            blue_button_pose = deepcopy(cartesian_msg.poses[0])
            red_button_pose = cartesian_msg.poses[0]
            comp_x = 0.017
            comp_x = -comp_x if board_state == "Flipped" else comp_x
            blue_button_pose.position.x += comp_x
            ethernet_cable_pose = cartesian_msg.poses[1]
            # key_pose = cartesian_msg.poses[3]
            key_hole_pose = cartesian_msg.poses[2]
            battery_cover_pose = cartesian_msg.poses[3]
            # cover_pressure_point_pose = cartesian_msg.poses[5]
            # ethernet_empty_pose = cartesian_msg.poses[4]
            ethernet_empty_pose = deepcopy(key_hole_pose)
            comp_x = 0.045
            comp_x = -comp_x if board_state == "Flipped" else comp_x
            ethernet_empty_pose.position.x += comp_x
            key_pose = deepcopy(key_hole_pose)
            cover_pressure_point_pose = deepcopy(battery_cover_pose)
            
            key_hole_spiral = generate_spiral(0.04, 0.04, key_hole_pose)

            key_y_dist = 0.065
            cover_x_dist = 0.02
            if(key_hole_pose.position.y > blue_button_pose.position.y):
                board_state = 'Not Flipped'
                key_pose.position.y -= key_y_dist
                cover_pressure_point_pose.position.x += cover_x_dist
            else:
                board_state = 'Flipped'
                key_pose.position.y += key_y_dist
                cover_pressure_point_pose.position.x -= cover_x_dist
                print("Flipped~!!!!!!!!!!!!!!!!!!!!!")
            
            key_pose.position.z = battery_cover_pose.position.z + 0.01

            board_state_msg = String()
            board_state_msg.data = board_state
            board_state_pub.publish(board_state_msg)
            print ("wait for done flipping")
            success_msg = rospy.wait_for_message("/done",String)
            print ("received done flipping")
            
            #form ethernet PoseArray
            ethernet_poses = PoseArray()
            ethernet_poses.poses.append(ethernet_empty_pose)
            ethernet_poses.poses.append(ethernet_cable_pose)

            
            pb_pub.publish(blue_button_pose)
            success_msg = rospy.wait_for_message("/done",String)

            key_pub.publish(key_pose)
            success_msg = rospy.wait_for_message("/done",String)
            
            key_hole_spiral_pub.publish(key_hole_spiral)
            success_msg = rospy.wait_for_message("/done",String)

            cover_pub.publish(cover_pressure_point_pose)
            success_msg = rospy.wait_for_message("/done",String)
            
            # pb_pub.publish(red_button_pose)
            # success_msg = rospy.wait_for_message("/done",String)
            
            return
        
        elif detection_index ==1:
            image, detections = model_detection.recieve_and_detect()
            # cv2.imshow('Captured Image'+str(window_index), image)
            # cv2.waitKey()
            # rospy.sleep(2)
            
            battery_hole_px = model_detection.get_class_detections(detections,'Battery_Hole', sort=True)
            battery_px = model_detection.get_class_detections(detections,'Battery', sort=True)
            task_board_px = model_detection.get_class_detections(detections,'Task_board')
            
            ethernet_cable_px = model_detection.get_class_detections(detections,'Ethernet_Cable')
            ethernet_empty_px = model_detection.get_class_detections(detections,'Ethernet_Empty')
            
            
            
            
            battery_hole_center_0 = [round(0.5 * (battery_hole_px[0][0]+battery_hole_px[0][2])),round(0.5 * (battery_hole_px[0][1]+battery_hole_px[0][3]))]
            battery_hole_center_1 = [round(0.5 * (battery_hole_px[1][0]+battery_hole_px[1][2])),round(0.5 * (battery_hole_px[1][1]+battery_hole_px[1][3]))]
            # battery_hole_center_0 = [battery_hole_px[0][0], battery_hole_px[0][1]]
            corner_0 = [battery_hole_px[0][2], battery_hole_px[0][3]]
            # battery_hole_center_1 = [battery_hole_px[1][0], battery_hole_px[1][1]]
            corner_1 = [battery_hole_px[1][2], battery_hole_px[1][3]]
            battery_center = [[round(0.5 * (battery_px[0][0]+battery_px[0][2])),round(0.5 * (battery_px[0][1]+battery_px[0][3]))],
                              [round(0.5 * (battery_px[1][0]+battery_px[1][2])),round(0.5 * (battery_px[1][1]+battery_px[1][3]))]]
            task_board_center = [round(0.5 * (task_board_px[0][0]+task_board_px[0][2])),round(0.5 * (task_board_px[0][1]+task_board_px[0][3]))]
            ethernet_cable_center = [round(0.5 * (ethernet_cable_px[0][0]+ethernet_cable_px[0][2])),round(0.5 * (ethernet_cable_px[0][1]+ethernet_cable_px[0][3]))]
            ethernet_empty_center = [round(0.5 * (ethernet_empty_px[0][0]+ethernet_empty_px[0][2])),round(0.5 * (ethernet_empty_px[0][1]+ethernet_empty_px[0][3]))]
            
            cv2.circle(image, tuple(battery_hole_center_0), 1, (0, 255, 0), thickness=1)
            cv2.circle(image, tuple(battery_hole_center_1), 1, (0, 255, 0), thickness=1)
            cv2.circle(image, tuple(battery_center[0]), 1, (0, 255, 0), thickness=1)
            cv2.circle(image, tuple(battery_center[1]), 1, (0, 255, 0), thickness=1)
            cv2.circle(image, tuple(ethernet_cable_center), 1, (0, 255, 0), thickness=1)
            cv2.circle(image, tuple(ethernet_empty_center), 1, (0, 255, 0), thickness=1)
            
            # print("SHOWWWWWW")
            # cv2.startWindowThread()
            # cv2.namedWindow("Captured Image")
            # cv2.imshow("Captured Image", image)
            # cv2.waitKey(3000)
            # cv2.destroyAllWindows()
            # print("DONEEEEEE")
                      
            battery_left_center , battery_right_center = get_battery_orientation(battery_center, task_board_center)
            
            pixel_to_cartesian_msg = Float32MultiArray()
            pixel_to_cartesian_msg.data = [battery_left_center[1],battery_left_center[0],
                                           battery_right_center[1],battery_right_center[0],
                                           battery_hole_center_0[1],battery_hole_center_0[0],
                                           corner_0[1], corner_0[0],
                                           battery_hole_center_1[1],battery_hole_center_1[0],
                                           corner_1[1], corner_1[0],
                                           ethernet_cable_center[1],ethernet_cable_center[0],
                                           ethernet_empty_center[1],ethernet_empty_center[0]]
            
            
            cartesian_msg = send_to_pct(pixel_to_cartesian_msg)
            
            battery_left_pose = cartesian_msg.poses[0]
            battery_right_pose = cartesian_msg.poses[1]
            battery_hole_pose_0 = cartesian_msg.poses[2]
            corner_pose_0 = cartesian_msg.poses[3]
            battery_hole_pose_1 = cartesian_msg.poses[4]
            corner_pose_1 = cartesian_msg.poses[5]
            ethernet_cable_pose = cartesian_msg.poses[6]
            ethernet_empty_pose = cartesian_msg.poses[7]
            
            battery_right_pose.position.x = battery_cover_pose.position.x
            battery_left_pose.position.x = battery_cover_pose.position.x
            
            c_x_0 = (corner_pose_0.position.x + battery_hole_pose_0.position.x) / 2
            c_y_0 = (corner_pose_0.position.y + battery_hole_pose_0.position.y) / 2
            c_x_1 = (corner_pose_1.position.x + battery_hole_pose_1.position.x) / 2
            c_y_1 = (corner_pose_1.position.y + battery_hole_pose_1.position.y) / 2
    
            # battery_hole_pose_0.position.x = c_x_0
            # battery_hole_pose_0.position.y = c_y_0
            
            # battery_hole_pose_1.position.x = c_x_1
            # battery_hole_pose_1.position.y = c_y_1
            dx, dy, dx2 = -0.017, 0.023, 0
            if board_state == "Flipped":
                dx, dy, dx2 = dx*-1, dy*-1, dx2*-1
                
            battery_hole_pose_0.position.x = blue_button_pose.position.x + dx
            battery_hole_pose_0.position.y = blue_button_pose.position.y + dy
                
            battery_hole_pose_1.position.x = blue_button_pose.position.x + dx2
            battery_hole_pose_1.position.y = blue_button_pose.position.y + dy
            
            
            
            '''
            Spiral old
            '''
            # #crop battery holes for spiral search
            # battery_hole_cropped_0 = image[battery_hole_px[0][1]:battery_hole_px[0][3]+1,battery_hole_px[0][0]:battery_hole_px[0][2]+1]
            # battery_hole_cropped_1 = image[battery_hole_px[1][1]:battery_hole_px[1][3]+1,battery_hole_px[1][0]:battery_hole_px[1][2]+1]
            # ethernet_empty_cropped = image[ethernet_empty_px[0][1]:ethernet_empty_px[0][3]+1,ethernet_empty_px[0][0]:ethernet_empty_px[0][2]+1]
            
            # #return indices of battery hole spiral search
            # battery_hole_list_0 = spiral(battery_hole_cropped_0.shape[1], battery_hole_cropped_0.shape[0],battery_hole_center_0)
            # battery_hole_list_1 = spiral(battery_hole_cropped_1.shape[1] // 2, battery_hole_cropped_1.shape[0],battery_hole_center_1)
            # ethernet_empty_list = spiral(ethernet_empty_cropped.shape[1], ethernet_empty_cropped.shape[0],ethernet_empty_center)
            
            # # Convert pixels to poses
            # pixel_to_cartesian_msg = Float32MultiArray()
            # pixel_to_cartesian_msg.data = battery_hole_list_0
            # cartesian_msg = send_to_pct(pixel_to_cartesian_msg)
            # battery_hole_spiral_0 = cartesian_msg
            
            # pixel_to_cartesian_msg = Float32MultiArray()
            # pixel_to_cartesian_msg.data = battery_hole_list_1
            # cartesian_msg = send_to_pct(pixel_to_cartesian_msg)
            # battery_hole_spiral_1 = cartesian_msg
            
            # pixel_to_cartesian_msg = Float32MultiArray()
            # pixel_to_cartesian_msg.data = ethernet_empty_list
            # cartesian_msg = send_to_pct(pixel_to_cartesian_msg)
            # ethernet_empty_spiral = cartesian_msg
            '''
            Spiral new
            '''
            battery_hole_spiral_0 = generate_spiral(0.04, 0.04, battery_hole_pose_0)
            battery_hole_spiral_1 = generate_spiral(0.04, 0.04, battery_hole_pose_1)
            ethernet_empty_spiral = generate_spiral(0.04, 0.04, ethernet_empty_pose)
            
            if board_state == "Flipped":
                battery_right_pose, battery_left_pose = battery_left_pose, battery_right_pose
                print("Switched batteries")
                
            # # Go Pick the right battery.
            right_battery_pub.publish(battery_right_pose)
            success_msg = rospy.wait_for_message("/done",String)
            
            # # Put the battery in the other hole (with spiral search method).
            battery_hole_spiral_pub.publish(battery_hole_spiral_0)
            success_msg = rospy.wait_for_message("/done",String)
            
            # # Go Pick left the battery.
            left_battery_pub.publish(battery_left_pose)
            success_msg = rospy.wait_for_message("/done",String)
            
            # # Put the battery in the hole (with spiral search method).
            battery_hole_spiral_pub.publish(battery_hole_spiral_1)
            success_msg = rospy.wait_for_message("/done",String)
            
            
            # Go Pick Ethernet Cable.
            pose_array = PoseArray()
            pose_array.poses.append(ethernet_empty_pose)
            pose_array.poses.append(ethernet_cable_pose)
            ethernet_pub.publish(pose_array)
            success_msg = rospy.wait_for_message("/done",String)
            
            # Put the Ethernet Cable in its port.
            ethernet_empty_spiral_pub.publish(ethernet_empty_spiral)
            success_msg = rospy.wait_for_message("/done",String)
            
            #Press Red Button
            pb_pub.publish(red_button_pose)
            success_msg = rospy.wait_for_message("/done",String)
            
            return
                       
            
            
        # cv2.destroyAllWindows()
        # return

class Model:
    def __init__(self,
                 model_type='ssd',
                 model_path='/home/' + user + '/' + ws + '/src/perception/models/',
                 image_topic='/camera/color/image_raw',
                 cutting_plan_topic='/cutting_path',
                 imgsz=1280):

        self.model_type = model_type
        thresh = 0.2

        if model_type == 'ssd':

            PATH_TO_MODEL_DIR = model_path + 'saved_model'
            PATH_TO_LABELS = model_path + 'label_map.pbtxt'

            print('Loading model... ')
            start_time = time.time()

            # Load the save tensorflow model.
            model = tf.saved_model.load(PATH_TO_MODEL_DIR)

            def detect_fn(image_np):
                # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
                input_tensor = tf.convert_to_tensor(image_np)
                # The model expects a batch of images, so add an axis with `tf.newaxis`.
                input_tensor = input_tensor[tf.newaxis, ...]

                detections = model(input_tensor,draw=True)
                # All outputs are batches tensors.
                # Convert to numpy arrays, and take index [0] to remove the batch dimension.
                # We're only interested in the first num_detections.
                num_detections = int(detections.pop('num_detections'))
                detections = {key: threshue[0, :num_detections].numpy()
                              for key, threshue in detections.items()}
                detections['num_detections'] = num_detections
                return detections

            self.detect_fn = detect_fn

            end_time = time.time()
            elapsed_time = end_time - start_time
            print('Done! Took {} seconds'.format(elapsed_time))

            # define class thresholds, ids, and names.
            self.class_thresh = {
                'Battery':           {'thresh': 0.5, 'id': 1},
                'Connector':         {'thresh': 0.2, 'id': 2},
                'CPU':               {'thresh': 0.8, 'id': 3},
                'Fan':               {'thresh': 0.8, 'id': 4},
                'Hard Disk':         {'thresh': 0.5, 'id': 5},
                'Motherboard':       {'thresh': 0.8, 'id': 6},
                'RAM':               {'thresh': 0.7, 'id': 7},
                'Screw':             {'thresh': 0.2, 'id': 8},
                'SSD':               {'thresh': 0.8, 'id': 9},
                'WLAN':              {'thresh': 0.7, 'id': 10},
                'CD-ROM':            {'thresh': 0.5, 'id': 11},
                'Laptop_Back_Cover': {'thresh': 0.92, 'id': 12}
            }
            
        elif model_type == 'yolo_detection':
            PATH_TO_MODEL_DIR = model_path + 'Model_Weights3.pt'
            self.detect_fn = Yolo(PATH_TO_MODEL_DIR, imgsz, thresh).detect
            # define class thresholds, ids, and names.
            self.class_thresh = {
                'Blue_Button':          {'thresh': thresh, 'id': 0},
                'Red_Button':           {'thresh': thresh, 'id': 1},
                'Ethernet_Empty':       {'thresh': thresh, 'id': 2},
                'Ethernet_Cable':       {'thresh': thresh, 'id': 3},
                'Ethernet_Tip':         {'thresh': thresh, 'id': 4},
                'Key':                  {'thresh': thresh, 'id': 5},
                'Key_hole':             {'thresh': thresh, 'id': 6},
                'Microcontroller':      {'thresh': thresh, 'id': 7},
                'Battery_Cover':        {'thresh': thresh, 'id': 8},
                'Battery_Hole':         {'thresh': thresh, 'id': 9},
                'Task_board':           {'thresh': thresh, 'id': 10},
                'Cover_Pressure_Point': {'thresh': thresh, 'id': 11},
                'Battery':              {'thresh': thresh, 'id': 12}
            }
        
        elif model_type == 'yolo_orientation':
            PATH_TO_MODEL_DIR = model_path + 'Model2_Weights2.pt'
            self.detect_fn = Yolo(PATH_TO_MODEL_DIR, imgsz, thresh).detect
            # define class thresholds, ids, and names.
            self.class_thresh = {
                'Blue_Button':          {'thresh': thresh, 'id': 0},
                'Red_Button':           {'thresh': thresh, 'id': 1},
                'Ethernet_Empty':       {'thresh': thresh, 'id': 2},
                'Ethernet_Cable':       {'thresh': thresh, 'id': 3},
                'Ethernet_Tip':         {'thresh': thresh, 'id': 4},
                'Key':                  {'thresh': thresh, 'id': 5},
                'Key_hole':             {'thresh': thresh, 'id': 6},
                'Microcontroller':      {'thresh': thresh, 'id': 7},
                'Battery_Cover':        {'thresh': thresh, 'id': 8},
                'Left_Battery':         {'thresh': thresh, 'id': 9},
                'Battery_Tip':          {'thresh': thresh, 'id': 10},
                'Battery_Hole':         {'thresh': thresh, 'id': 11},
                'Task_board':           {'thresh': thresh, 'id': 12},
                'Cover_Pressure_Point': {'thresh': thresh, 'id': 13},
                'Key_Pickup_Empty':     {'thresh': thresh, 'id': 14},
                'Ethernet_Top_Empty':   {'thresh': thresh, 'id': 15},
                'Right_Battery':        {'thresh': thresh, 'id': 16}
            }

        self.image_topic = image_topic
        self.image = None
        self.imgsz = imgsz

        # dictionary to convert class id to class name
        self.cid_to_cname = {
            vals['id']: cname for cname, vals in self.class_thresh.items()}

        # dictionary to convert class name to class id
        self.cname_to_cid = {cname: vals['id']
                             for cname, vals in self.class_thresh.items()}

        # dictionary to convert class id to class threshold
        self.cid_to_cthresh = {vals['id']: vals['thresh'] for cname,
                               vals in self.class_thresh.items()}

        self.path_publisher = rospy.Publisher(cutting_plan_topic,
                                              Float32MultiArray,
                                              queue_size=1)
        self.state_publisher = rospy.Publisher(
            "/found_components", String, queue_size=1)
        

    def remove_detections(self, detections, indicies_to_remove):
        detections['detection_boxes'] = np.delete(
            detections['detection_boxes'], indicies_to_remove, axis=0)
        detections['detection_classes'] = np.delete(
            detections['detection_classes'], indicies_to_remove)
        detections['detection_scores'] = np.delete(
            detections['detection_scores'], indicies_to_remove)

    def recieve_and_detect(self, read_img=False, image_path=None):
        # Either read image from path or wait for ros message
        if image_path is not None and read_img:
            image_np = cv2.imread(image_path)
            image_np = cv2.resize(image_np, (832, 480))
        else:
            # # Wait for rgb camera stream to publish a frame.
            while True:
                try:
                    image_msg = rospy.wait_for_message(
                        self.image_topic, Image, timeout=5)
                    break
                except rospy.ROSException:
                    components_msg = String()
                    components_msg.data = "Camera Disconnected"
                    self.state_publisher.publStampedish(components_msg)
                    print("Camera Disconnected")
                    connection_msg = rospy.wait_for_message(
                        "/connection_error_handled", String)
                    continue

            # Convert msg to numpy image.2
            image_np = numpify(image_msg)

            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        self.image = image_np
        # Run forward prop of model to get the detections.
        im0, detections = self.detect_fn(image_np, draw=True)

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # Filter the detections by using the pre-defined thresholds.
        indicies_to_remove = []
        for j, score in enumerate(detections['detection_scores']):
            min_score = self.cid_to_cthresh[detections['detection_classes'][j]]
            if score < min_score:
                indicies_to_remove.append(j)

        self.remove_detections(detections, indicies_to_remove)

        # print(image_np.shape)2
        # print(detections['detection_boxes'][0])
        detections['detection_boxes'][:, 0] *= image_np.shape[0]
        detections['detection_boxes'][:, 2] *= image_np.shape[0]
        detections['detection_boxes'][:, 1] *= image_np.shape[1]
        detections['detection_boxes'][:, 3] *= image_np.shape[1]

        return im0, detections

    def get_class_detections(self, detections, class_name, min_score=0, best_only=False, get_scores=False, format=(
            'x1', 'y1', 'x2', 'y2'), sort=True):
        """
        Extract detections of a specific class from all the detections in the image.
        """
        boxes = []
        scores = []
        best_box = None
        max_score = 0
        for i, cid in enumerate(detections['detection_classes']):
            if cid == self.cname_to_cid[class_name]:
                score = detections['detection_scores'][i]
                if score < min_score or (score < max_score and best_only):
                    continue
                max_score = score
                box = convert_format(detections['detection_boxes'][i], in_format=(
                    'y1', 'x1', 'y2', 'x2'), out_format=format)
                box = [int(x) for x in box]
                best_box = box
                boxes.append(box)
                print("{} box added".format(class_name))
                scores.append(score)
        if sort:
            enumerated_boxes = sorted(list(enumerate(boxes)), key=lambda i: scores[i[0]], reverse=True)
            
            scores = [scores[enumerated_boxes[i][0]] for i in range(len(boxes))]
            boxes = [enumerated_boxes[i][1] for i in range(len(boxes))]
        if get_scores:
            return boxes, scores
        elif best_only:
            return best_box
        else:
            return boxes

    def publish_path(self, cut_path):
        # Publish Cutting Path.
        path_msg = Float32MultiArray()
        for x, y in cut_path:
            path_msg.data.append(y)
            path_msg.data.append(x)
        self.path_publisher.publish(path_msg)


if __name__ == "__main__":

    rospy.init_node("competition_detection")
    pixel_to_cartesian_pub = rospy.Publisher("/cutting_path", Float32MultiArray, queue_size=1)
    operations_pub = rospy.Publisher("/operation", String, queue_size=1)
    desired_pose_pub = rospy.Publisher("/desired_pose_topic", PoseStamped, queue_size=1)
    left_battery_pub = rospy.Publisher("/left_battery_topic", Pose, queue_size=1)
    right_battery_pub = rospy.Publisher("/right_battery_topic", Pose, queue_size=1)
    battery_hole_spiral_pub = rospy.Publisher("/battery_hole_spiral_topic", PoseArray, queue_size=1)
    battery_hole_pub = rospy.Publisher("/battery_hole_topic", Pose, queue_size=1)
    ethernet_pub = rospy.Publisher("/ethernet_topic", PoseArray, queue_size=1)
    ethernet_empty_spiral_pub = rospy.Publisher("/ethernet_empty_spiral_topic", PoseArray, queue_size=1)
    key_pub = rospy.Publisher("/key_topic", Pose, queue_size=1)
    key_hole_pub = rospy.Publisher("/key_hole_topic", Pose, queue_size=1)
    key_hole_spiral_pub = rospy.Publisher("/key_hole_spiral_topic", PoseArray, queue_size=1)
    pb_pub = rospy.Publisher("/pb_topic", Pose, queue_size=1)
    cover_pub = rospy.Publisher("/cover_topic", Pose, queue_size=1)
    board_state_pub = rospy.Publisher("/board_state_topic", String, queue_size=1)


    sys.path.insert(
        0, '/home/' + user + '/TensorFlow/workspace/yolov5')

    # Parameters that can be given from command-line / parameter-server.
    ns = '/components_detection'
    publish_flipping_plan_data = rospy.get_param(
        ns+'/publish_flipping_plan_data', True)
    publish_cut_path = rospy.get_param(ns+'/publish_cut_path', False)
    publish_screw_centers = rospy.get_param(ns+'/publish_screw_centers', False)
    use_state = True

    model_detection = Model(model_path='/home/' + user + '/' + ws + '/src/perception/models/',
                  image_topic='/camera/color/image_raw',
                  cutting_plan_topic="/cutting_path", model_type='yolo_detection', imgsz=832)

    model_orientation = Model(model_path='/home/' + user + '/' + ws + '/src/perception/models/',
                  image_topic='/camera/color/image_raw',
                  cutting_plan_topic="/cutting_path", model_type='yolo_orientation', imgsz=832)

    rospy.Subscriber("/capture_state", String, capture_callback)
    img_num = 10
    images_path = '/home/' + user + '/TensorFlow/workspace/training_demo/images/test/'
    
    if use_state:
        rospy.spin()
          