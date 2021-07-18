#!/usr/bin/env python
import cv2
from perception.yolo_detector import Yolo
from perception.coco_datasets import convert_format
from perception.laptop_perception_helpers import (
    detect_picking_point,
    plan_cover_cutting_path,
    interpolate_path,
    plan_port_cutting_path,
    filter_boxes_from_image,
    draw_lines,
    draw_boxes,
    box_near_by_dist,
    box_to_center,
    correct_circles,
    RealsenseHelpers,
    detect_laptop_pose,
    xyz_list_to_pose_array,
)
from robot_helpers.srv import TransformPoses, CreateFrameAtPose
from perception.msg import PerceptionData
import sys
import ros_numpy
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray
from ros_numpy import numpify

from threading import Thread
from copy import deepcopy
import numpy as np
import tensorflow as tf
import time
import socket
import warnings

warnings.filterwarnings("ignore")  # Suppress Matplotlib warnings

user = socket.gethostname()
ws = "ewaste_ws" if user == "abdelrhman" else "abb_ws"


from threading import Thread
import cv2, time


class ImshowThread(object):
    def __init__(self, src=0):
        # Start the thread to read frames from the video stream
        self.thread = Thread(args=())
        self.thread.daemon = True
        self.thread.start()

    def show_frame(self, frame1, frame2):
        # Display frames in main program
        cv2.imshow("frame1", frame1)
        cv2.imshow("frame2", frame2)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("e") or rospy.is_shutdown():
            cv2.destroyAllWindows()
            print("Here")
            exit(1)


class Model:
    def __init__(
        self,
        model_type="ssd",
        model_path="/home/" + user + "/" + ws + "/src/perception/models/",
        image_topic="/camera/color/image_raw",
        cutting_plan_topic="/cutting_path",
        perception_data_topic="/perception_data",
        get_laptop_pose_topic="/get_laptop_pose",
        imgsz=1280,
    ):

        self.model_type = model_type
        if model_type == "ssd":

            PATH_TO_MODEL_DIR = model_path + "saved_model"
            PATH_TO_LABELS = model_path + "label_map.pbtxt"

            print("Loading model... ")
            start_time = time.time()

            # Load the save tensorflow model.
            model = tf.saved_model.load(PATH_TO_MODEL_DIR)

            def detect_fn(image_np):
                # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
                input_tensor = tf.convert_to_tensor(image_np)
                # The model expects a batch of images, so add an axis with `tf.newaxis`.
                input_tensor = input_tensor[tf.newaxis, ...]

                detections = model(input_tensor)
                # All outputs are batches tensors.
                # Convert to numpy arrays, and take index [0] to remove the batch dimension.
                # We're only interested in the first num_detections.
                num_detections = int(detections.pop("num_detections"))
                detections = {
                    key: threshue[0, :num_detections].numpy()
                    for key, threshue in detections.items()
                }
                detections["num_detections"] = num_detections
                return detections

            self.detect_fn = detect_fn

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Done! Took {} seconds".format(elapsed_time))

            # define class thresholds, ids, and names.
            self.class_thresh = {
                "Battery": {"thresh": 0.5, "id": 1},
                "Connector": {"thresh": 0.2, "id": 2},
                "CPU": {"thresh": 0.8, "id": 3},
                "Fan": {"thresh": 0.8, "id": 4},
                "Hard Disk": {"thresh": 0.5, "id": 5},
                "Motherboard": {"thresh": 0.8, "id": 6},
                "RAM": {"thresh": 0.7, "id": 7},
                "Screw": {"thresh": 0.2, "id": 8},
                "SSD": {"thresh": 0.8, "id": 9},
                "WLAN": {"thresh": 0.7, "id": 10},
                "CD-ROM": {"thresh": 0.5, "id": 11},
                "Laptop_Back_Cover": {"thresh": 0.92, "id": 12},
            }

        elif model_type == "yolo":
            PATH_TO_MODEL_DIR = model_path + "best.pt"
            self.detect_fn = Yolo(PATH_TO_MODEL_DIR, imgsz).detect
            # define class thresholds, ids, and names.
            self.class_thresh = {
                "Connector": {"thresh": 0.2, "id": 0},
                "CPU": {"thresh": 0.5, "id": 1},
                "Fan": {"thresh": 0.8, "id": 2},
                "Hard Disk": {"thresh": 0.3, "id": 3},
                "Motherboard": {"thresh": 0.7, "id": 4},
                "RAM": {"thresh": 0.7, "id": 5},
                "Screw": {"thresh": 0.3, "id": 6},
                "SSD": {"thresh": 0.8, "id": 7},
                "Battery": {"thresh": 0.5, "id": 8},
                "WLAN": {"thresh": 0.5, "id": 9},
                "CD-ROM": {"thresh": 0.5, "id": 10},
                "Laptop_Back_Cover": {"thresh": 0.6, "id": 11},
                "Port": {"thresh": 0.2, "id": 12},
                "RTC_Battery": {"thresh": 0.5, "id": 13},
                "HD_Bay": {"thresh": 0.3, "id": 14},
                "CD_Bay": {"thresh": 0.3, "id": 15},
                "Battery_Bay": {"thresh": 0.3, "id": 16},
                "SD_Slot": {"thresh": 0.3, "id": 17},
                "front_cover": {"thresh": 0.5, "id": 18},
                "mouse_pad": {"thresh": 0.5, "id": 19},
                "keyboard_bay": {"thresh": 0.3, "id": 20},
                "keyboard": {"thresh": 0.5, "id": 21},
            }

        self.image_topic = image_topic
        self.image = None
        self.imgsz = imgsz
        self.screws_near_cpu = []
        self.screws_near_cpu_cut_path = []
        self.image_win = "image_window"
        self.depth_win = "depth_window"
        self.imshow_thread = ImshowThread()

        # dictionary to convert class id to class name
        self.cid_to_cname = {
            vals["id"]: cname for cname, vals in self.class_thresh.items()
        }

        # dictionary to convert class name to class id
        self.cname_to_cid = {
            cname: vals["id"] for cname, vals in self.class_thresh.items()
        }

        # dictionary to convert class id to class threshold
        self.cid_to_cthresh = {
            vals["id"]: vals["thresh"] for cname, vals in self.class_thresh.items()
        }

        self.path_publisher = rospy.Publisher(
            cutting_plan_topic, Float32MultiArray, queue_size=1
        )
        self.perception_data_publisher = rospy.Publisher(
            perception_data_topic, PerceptionData, queue_size=1
        )
        self.get_laptop_pose_publisher = rospy.Publisher(
            get_laptop_pose_topic, Float32MultiArray, queue_size=1
        )
        self.state_publisher = rospy.Publisher(
            "/found_components", String, queue_size=1
        )
        rospy.Subscriber("/capture_state", String, self.capture_callback)
        self.state = "capture"  # if state == "caoture cpu screws" preserve the previous cpu screws
        self.free_area_tunning_pub = rospy.Publisher(
            "/free_area_tunning", Image, queue_size=1
        )

        # Detect laptop pose parameters
        self.cam_helpers = RealsenseHelpers()
        self.trackbar_limits = {
            "x_min": 1802,
            "x_max": 2212,
            "y_min": 0,
            "y_max": 2125,
            "z_min": 249,
            "z_max": 340,
        }
        self.limits = {}
        for key, val in self.trackbar_limits.items():
            self.limits[key] = 0.001 * val - 2
            if key in ["z_min", "z_max"]:
                self.limits[key] += 2

    def capture_callback(self, msg):
        # recieve state from messege
        self.state = msg.data
        yolo_draw = False
        depth_draw = True
        draw = True
        # Recieve an image msg from camera topic, then, return image and detections.
        image, detections = self.recieve_and_detect(draw=yolo_draw)

        # Get laptop pose and flipping plan data (laptop_center, flipping_point, upper_point)
        (
            laptop_data_pose_array,
            dist_mat,
            dist_image,
        ) = self.detect_laptop_pose_data_as_pose_array(draw=depth_draw)

        # Transform laptop_center from calibrated_frame to base_link
        laptop_center_pose_array = PoseArray()
        laptop_center_pose_array.poses.append(deepcopy(laptop_data_pose_array.poses[0]))
        transformation_data = {
            "ref_frame": String("calibrated_frame"),
            "target_frame": String("base_link"),
            "poses_to_transform": laptop_center_pose_array,
        }
        response = self.service_req(
            "/transform_poses", TransformPoses, inputs=transformation_data
        )
        laptop_center_transformed = response.transformed_poses
        
        # Create/Update frame at the laptop center
        frame_pose = laptop_center_transformed.poses[0]
        frame_data = {
            "ref_frame": String("base_link"),
            "new_frame": String("laptop"),
            "frame_pose": frame_pose,
        }
        _ = self.service_req(
            "/create_frame_at_pose", CreateFrameAtPose, inputs=frame_data
        )

        # Generate the cover cutting path, and screw holes from given detections and image to visulaise on.
        cover_cut_path, screw_holes, cover_type = self.generate_cover_cutting_path(
            image,
            detections,
            min_screw_score=0,
            tol=50,
            method=1,
            min_hole_dist=10,
            hole_tol=0,  # generated path params
            return_holes_inside_cut_path=False,
            filter_screws=False,
            draw=draw,
        )
        


        # Filter screws near the cpu and add them to self.screws_near_cpu
        if self.state != "capture cpu screws":
            screw_holes = self.filter_screws_near_cpu(
                screw_holes, detections, dist_as_side_ratio=0.5, draw=draw, image=image
            )

        # Adjust screw centers
        # screw_holes = correct_circles(image, screw_holes)

        # Generate the cover cutting path, and screw holes from given detections and image to visulaise on.
        ports_cut_path = self.generate_ports_cutting_path(image, detections, draw=draw)

        # Generate the screws cut paths
        screws_cut_path = self.generate_rectangular_cutting_path(
            screw_holes, interpolate=False
        )
        
        # Generate the keyboard cut path
        keyboard = self.get_class_detections(detections, "keyboard", best_only=True)
        keyboard_cut_path = []
        if len(keyboard) > 0:
            keyboard_cut_path = self.generate_rectangular_cutting_path([keyboard], interpolate=False)[0]

        if self.state != "capture cpu screws":
            # Generate the screws_near_cpu cut paths
            screws_near_cpu_cut_path_px = self.generate_rectangular_cutting_path(
                self.screws_near_cpu, interpolate=False
            )
            # Get 3D positions of the cpu screws cut path
            for screw in screws_near_cpu_cut_path_px:
                cpu_screws_as_xyz_list = self.cam_helpers.px_to_xyz(
                    dist_mat=dist_mat, px_data=screw, px_data_format="list_of_tuples"
                )
                cpu_screws_as_pose_array = xyz_list_to_pose_array(
                    cpu_screws_as_xyz_list
                )
                # Transform cpu screws to be wrt laptop frame
                transformation_data = {
                    "ref_frame": String("calibrated_frame"),
                    "target_frame": String("laptop"),
                    "poses_to_transform": cpu_screws_as_pose_array,
                }
                response = self.service_req(
                    "/transform_poses", TransformPoses, inputs=transformation_data
                )
                cpu_screw_transformed = response.transformed_poses
                self.screws_near_cpu_cut_path.append(cpu_screw_transformed)

        # Construct perception_data msg.
        data_msg = PerceptionData()

        # Add flipping and laptop pose data to the data_msg.
        data_msg.flipping_points = laptop_data_pose_array

        if cover_type == "Laptop_Back_Cover":
            data_msg.front_cover_cut_path = self.construct_float_multi_array([])
            # Add back cover cut path
            data_msg.back_cover_cut_path = self.construct_float_multi_array(
                cover_cut_path
            )
        elif cover_type == "front_cover":
            data_msg.back_cover_cut_path = self.construct_float_multi_array([])
            # Add front cover cut path
            data_msg.front_cover_cut_path = self.construct_float_multi_array(
                cover_cut_path
            )
        
        # Add keyboard cut path
        data_msg.keyboard_cut_path = self.construct_float_multi_array(
                keyboard_cut_path
            )
        
        # Add ports cut paths
        for i in range(len(ports_cut_path)):
            path_msg = self.construct_float_multi_array(ports_cut_path[i])
            data_msg.ports_cut_path.append(path_msg)

        # Add screws centers.
        screw_boxes = [box_to_center(sb) for sb in screw_holes]
        data_msg.screws = self.construct_float_multi_array(screw_boxes)

        # Add cpu screws
        cpu_screw_boxes = [box_to_center(sb) for sb in self.screws_near_cpu]
        data_msg.screws_near_cpu = self.construct_float_multi_array(cpu_screw_boxes)

        # Add screws cut path
        for i in range(len(screws_cut_path)):
            path_msg = self.construct_float_multi_array(screws_cut_path[i])
            data_msg.screws_cut_path.append(path_msg)

        # Add cpu screws cut path
        data_msg.screws_near_cpu_cut_path = self.screws_near_cpu_cut_path
        # for i in range(len(screws_near_cpu_cut_path)):
        #     path_msg = self.construct_float_multi_array(screws_near_cpu_cut_path[i])
        #     data_msg.screws_near_cpu_cut_path.append(path_msg)

        # Add detected frontcover as mousepad center.
        data_msg.front_cover = self.get_detection_as_msg(
            detections=detections, class_name="mouse_pad", best_only=True
        )
        
        # Add detected keyboard center.
        data_msg.keyboard = self.get_detection_as_msg(
            detections=detections, class_name="keyboard", best_only=True
        )
        
        # Add detected CD-ROM.
        data_msg.cd_rom = self.get_detection_as_msg(
            detections=detections, class_name="CD-ROM", best_only=True
        )

        # Add detected Hard Disk.
        data_msg.hard_disk = self.get_detection_as_msg(
            detections=detections, class_name="Hard Disk", best_only=True
        )

        # Add detected Fan.
        data_msg.fan = self.get_detection_as_msg(
            detections=detections, class_name="Fan", best_only=True
        )

        # Add detected CPUs.
        data_msg.cpu = self.get_detection_as_msg(
            detections=detections, class_name="CPU", best_only=False
        )

        # Add detected motherboard.
        # data_msg.motherboard = self.get_detection_as_msg(detections=detections, class_name="Motherboard", best_only=True)
        mother_picking_point = self.free_areas_detection(
            detections=detections, img=image, tol=20, use_depth=True, draw=draw
        )
        data_msg.motherboard = self.construct_float_multi_array(
            path_points=mother_picking_point
        )

        key = 0
        if draw or yolo_draw or depth_draw:
            # self.imshow_thread.show_frame(image, dist_image)
            cv2.imshow(self.image_win, image)
            cv2.imshow(self.depth_win, dist_image)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
        # print("Enter 0 if wait for another callback, else publish the data")
        # decision = int(input())
        # if decision == 0:
        #     return
        print("Publishing Perception Data")
        components_msg = String()
        components_msg.data = "components"
        cut_boxes = []

        # Publish the generated flipping plan data if not empty.
        if publish_flipping_plan_data:
            flipping_plan_data = model.get_flipping_plan_data(detections)
            if len(flipping_plan_data) > 0:
                print("Publishing flipping plan data")
                cut_boxes.extend(flipping_plan_data)

        # Publish the generated cutting path if not empty.
        if len(cover_cut_path) > 0 and publish_cut_path:
            cut_boxes.extend(cover_cut_path)
            components_msg.data = "cover"

        # Publish the generated cutting path if not empty.
        if len(screw_holes) > 0 and publish_screw_centers:
            print("Publishing screw cut paths")
            # screw_centers = [(sh[0] + (sh[2] // 2), sh[1] + (sh[3] // 2)) for sh in screw_holes]
            for sh in screw_holes:
                x, y, x2, y2 = sh[0], sh[1], sh[0] + sh[2], sh[1] + sh[3]
                box_path = [(x, y), (x, y2), (x2, y2), (x2, y), (x, y)]
                cut_boxes.extend(interpolate_path(box_path))
            components_msg.data = "screws"

        if len(cut_boxes) >= 1:
            self.publish_path(cut_boxes)
        self.state_publisher.publish(components_msg)

        rospy.sleep(1)

        # Publish Perception Data
        self.perception_data_publisher.publish(data_msg)

    def get_detection_as_msg(
        self, detections, class_name, best_only=False, preprocessor=None
    ):
        preprocessor = box_to_center if preprocessor is None else preprocessor
        boxes = self.get_class_detections(detections=detections, class_name=class_name)
        processed_boxes = []
        if len(boxes) > 0:
            boxes = [boxes[0]] if best_only else boxes
            processed_boxes = [preprocessor(box) for box in boxes]
        return self.construct_float_multi_array(processed_boxes)

    def remove_detections(self, detections, indicies_to_remove):
        detections["detection_boxes"] = np.delete(
            detections["detection_boxes"], indicies_to_remove, axis=0
        )
        detections["detection_classes"] = np.delete(
            detections["detection_classes"], indicies_to_remove
        )
        detections["detection_scores"] = np.delete(
            detections["detection_scores"], indicies_to_remove
        )

    def recieve_and_detect(self, read_img=False, image_path=None, draw=False):
        # Either read image from path or wait for ros message
        if image_path is not None and read_img:
            image_np = cv2.imread(image_path)
            image_np = cv2.resize(image_np, (1280, 720))
        else:
            # # Wait for rgb camera stream to publish a frame.
            while True:
                try:
                    image_msg = rospy.wait_for_message(
                        self.image_topic, Image, timeout=5
                    )
                    break
                except rospy.ROSException:
                    components_msg = String()
                    components_msg.data = "Camera Disconnected"
                    self.state_publisher.publish(components_msg)
                    print("Camera Disconnected")
                    connection_msg = rospy.wait_for_message(
                        "/connection_error_handled", String
                    )
                    continue

            # Convert msg to numpy image.
            image_np = numpify(image_msg)

            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        self.image = image_np
        # Run forward prop of model to get the detections.
        image_np, detections = self.detect_fn(image_np, draw=draw)

        # detection_classes should be ints.
        detections["detection_classes"] = detections["detection_classes"].astype(
            np.int64
        )

        # Filter the detections by using the pre-defined thresholds.
        indicies_to_remove = []
        for j, score in enumerate(detections["detection_scores"]):
            min_score = self.cid_to_cthresh[detections["detection_classes"][j]]
            if score < min_score:
                indicies_to_remove.append(j)

        self.remove_detections(detections, indicies_to_remove)

        detections["detection_boxes"][:, 0] *= image_np.shape[0]
        detections["detection_boxes"][:, 2] *= image_np.shape[0]
        detections["detection_boxes"][:, 1] *= image_np.shape[1]
        detections["detection_boxes"][:, 3] *= image_np.shape[1]

        return image_np, detections

    def get_class_detections(
        self,
        detections,
        class_name,
        min_score=0,
        best_only=False,
        get_scores=False,
        format=("x1", "y1", "w", "h"),
        sort=True,
    ):
        """
        Extract detections of a specific class from all the detections in the image.
        """
        boxes = []
        scores = []
        best_box = []
        max_score = 0
        for i, cid in enumerate(detections["detection_classes"]):
            if cid == self.cname_to_cid[class_name]:
                score = detections["detection_scores"][i]
                if score < min_score or (score < max_score and best_only):
                    continue
                max_score = score
                box = convert_format(
                    detections["detection_boxes"][i],
                    in_format=("y1", "x1", "y2", "x2"),
                    out_format=format,
                )
                box = [int(x) for x in box]
                best_box = box
                boxes.append(box)
                print("{} box added".format(class_name))
                scores.append(score)
        if sort:
            enumerated_boxes = sorted(
                list(enumerate(boxes)), key=lambda i: scores[i[0]], reverse=True
            )
            scores = [scores[enumerated_boxes[i][0]] for i in range(len(boxes))]
            boxes = [enumerated_boxes[i][1] for i in range(len(boxes))]
        if get_scores:
            return boxes, scores
        elif best_only:
            return best_box
        else:
            return boxes

    def free_areas_detection(self, detections, img, tol=0, use_depth=False, draw=False):
        other_boxes = []
        for cname in self.cname_to_cid.keys():
            if cname == "Motherboard":
                mother_box = self.get_class_detections(
                    detections, "Motherboard", format=("x1", "y1", "x2", "y2"), best_only=True
                )
                if len(mother_box) < 1:
                    return []
            else:
                other_boxes.append(
                    self.get_class_detections(
                        detections, cname, format=("x1", "y1", "x2", "y2")
                    )
                )
        mx1, my1, mx2, my2 = mother_box[0], mother_box[1], mother_box[2], mother_box[3]
        mx1 = max(0, mx1 - tol)
        my1 = max(0, my1 - tol)
        mx2 = min(img.shape[1] - 1, mx2 + tol)
        my2 = min(img.shape[1] - 1, my2 + tol)
        free_area_img = deepcopy(img)
        free_area_img[: my1 + tol, :, :] = 0
        free_area_img[:, : mx1 + tol :] = 0
        free_area_img[my2 - tol :, :, :] = 0
        free_area_img[:, mx2 - tol :, :] = 0
        for cboxes in other_boxes:
            for box in cboxes:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                x1 = max(0, x1 - tol)
                y1 = max(0, y1 - tol)
                x2 = min(img.shape[1] - 1, x2 + tol)
                y2 = min(img.shape[0] - 1, y2 + tol)
                free_area_img[y1:y2, x1:x2, :] = 0
        # mother_img = free_area_img[my1:my2, mx1:mx2, :]
        # cv2.imshow("motherboard", free_area_img)
        # cv2.waitKey(0)
        draw_on = img if draw else None
        gray_img = cv2.cvtColor(free_area_img, cv2.COLOR_BGR2GRAY)
        depth_img = None
        if use_depth:
            depth_img = rospy.wait_for_message(
                "/camera/aligned_depth_to_color/image_raw", Image
            )
            depth_img = ros_numpy.numpify(depth_img).astype(np.float)
            dist_mat = self.cam_helpers.get_dist_mat_from_cam(
                depth_topic="/camera/aligned_depth_to_color/image_raw",
                intrinsics_topic="/camera/color/camera_info"
                )
        picking_point = detect_picking_point(
            gray_img,
            use_center=False,
            center=((my1 + my2) // 2, (mx1 + mx2) // 2),
            depth_img=deepcopy(depth_img),
            dist_mat=deepcopy(dist_mat),
            use_depth=use_depth,
            method=1,
            draw_on=draw_on
        )
        # self.free_area_tunning_pub.publish(ros_numpy.msgify(Image, mother_img, encoding='bgr8'))
        return [picking_point]

    def detect_laptop_pose_data_as_pose_array(self, draw=False):
        # Get current distances of all pixels from the depth image.
        dist_mat = self.cam_helpers.get_dist_mat_from_cam(transform_to_color=True)
        # Detect the laptop pose data (laptop_center, flipping_point, upper_point) as pixels
        laptop_data_px, dist_image = detect_laptop_pose(
            dist_mat,
            draw=draw,
            x_min=self.limits["x_min"],
            x_max=self.limits["x_max"],
            y_min=self.limits["y_min"],
            y_max=self.limits["y_max"],
            z_min=self.limits["z_min"],
            z_max=self.limits["z_max"],
        )
        # Deproject the pixels representing laptop pose data to xyz 3d pose data.
        laptop_pose_data = self.cam_helpers.px_to_xyz(laptop_data_px, dist_mat=dist_mat)

        # Put the data in a PoseArray and return it
        laptop_data_pose_array = xyz_list_to_pose_array(laptop_pose_data)

        return laptop_data_pose_array, dist_mat, dist_image

    def generate_rectangular_cutting_path(
        self, boxes, interpolate=False, npoints=20
    ):
        cut_boxes = []
        for b in boxes:
            x, y, x2, y2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
            box_path = [(x, y), (x, y2), (x2, y2), (x2, y), (x, y)]
            if interpolate:
                box_path = interpolate_path(box_path, npoints=npoints)
            cut_boxes.append(box_path)
        return cut_boxes

    def generate_ports_cutting_path(self, image, detections, draw=True):
        """Generate a cutting path and publish it."""

        image_np = image

        box_cname = "Motherboard"
        # Get detected laptop_covers.
        cover_boxes, cover_scores = self.get_class_detections(
            detections=detections, class_name=box_cname, get_scores=True
        )

        best_box = []
        if len(cover_boxes) > 0:
            # Get Motherboard with highest confidence.
            best_cover_score = 0
            for box, score in zip(cover_boxes, cover_scores):
                if score > best_cover_score:
                    best_cover_score = score
                    best_box = box

            # Visualise detected Motherboard
            if draw:
                draw_boxes(image_np, [best_box])

        else:
            return []

        port_boxes = []
        for cname in ["Port", "Connector"]:
            port_boxes.extend(
                self.get_class_detections(detections=detections, class_name=cname)
            )

        # Plan the Cutting Path.
        ports_cut_paths = []
        if len(best_box) > 0 and len(port_boxes) > 0:
            ports_cut_paths = plan_port_cutting_path(
                best_box,
                port_boxes,
                near_edge_dist=20,
                grouping_dist=40,
                cutting_dist=5,
            )

        # Visualise the cutting path.
        # Visualise detected screws/ports/connectors.
        if draw:
            draw_boxes(image_np, port_boxes)
            for ports_cut_path in ports_cut_paths:
                draw_lines(image_np, ports_cut_path)

        return ports_cut_paths

    def filter_screws_near_cpu(
        self, screws, detections, dist_as_side_ratio=0.5, image=None, draw=False
    ):
        cpu_boxes = self.get_class_detections(detections=detections, class_name="CPU")

        self.screws_near_cpu = list(
            filter(
                lambda box: box_near_by_dist(box, cpu_boxes, dist_as_side_ratio), screws
            )
        )

        if draw and image is not None:
            draw_boxes(image, cpu_boxes, draw_center=True, color=(255, 0, 0))
            draw_boxes(image, self.screws_near_cpu, draw_center=True, color=(0, 0, 255))

        screws = [box for box in screws if box not in self.screws_near_cpu]

        print("There are {} Screws near cpu".format(len(self.screws_near_cpu)))
        print("There are {} Screws not near cpu".format(len(screws)))

        return screws

    def generate_cover_cutting_path(
        self,
        image,
        detections,
        min_screw_score=0,
        tol=100,
        method=0,
        min_hole_dist=3,
        hole_tol=0,
        return_holes_inside_cut_path=False,
        filter_screws=False,
        draw=True,
    ):
        """Generate a cutting path and publish it."""

        image_np = image

        box_cname = "Laptop_Back_Cover"
        # Get detected laptop_covers.
        cover_boxes, cover_scores = self.get_class_detections(
            detections=detections, class_name=box_cname, get_scores=True
        )
        if len(cover_boxes) < 1:
            box_cname = "front_cover"
            # Get detected laptop_covers.
            cover_boxes, cover_scores = self.get_class_detections(
                detections=detections, class_name=box_cname, get_scores=True
                )

        screw_boxes = self.get_class_detections(
            detections=detections, class_name="Screw", min_score=min_screw_score
        )

        screw_boxes = [
            [
                sb[0] - hole_tol,
                sb[1] - hole_tol,
                sb[2] + 2 * hole_tol,
                sb[3] + 2 * hole_tol,
            ]
            for sb in screw_boxes
        ]

        # # Adjust screw centers
        # screw_boxes = correct_circles(image_np, screw_boxes)

        # Visualise detected screws/ports/connectors.
        if draw:
            draw_boxes(image_np, screw_boxes, draw_center=True)

        best_cover_box = []
        if len(cover_boxes) > 0:
            # Get laptop_cover with highest confidence.
            best_cover_score = 0
            for box, score in zip(cover_boxes, cover_scores):
                if score > best_cover_score:
                    best_cover_score = score
                    best_cover_box = box

            # Visualise detected laptop_cover/Motherboard
            if draw:
                draw_boxes(image_np, [best_cover_box])

        if filter_screws:
            screw_boxes = filter_boxes_from_image(
                screw_boxes, image_np, "Choose Boxes, then press 'c'"
            )
            # Visualise detected screws.
            draw_boxes(image_np, screw_boxes, color=(0, 0, 255))

        # Plan the Cutting Path.
        cut_path = []
        if len(best_cover_box) > 0 and len(screw_boxes) > 0:
            cut_path, holes_inside_cut_path = plan_cover_cutting_path(
                laptop_coords=best_cover_box,
                holes_coords=screw_boxes,
                method=method,
                interpolate=False,
                interp_step=4,
                tol=tol,
                min_hole_dist=min_hole_dist,
            )

            if return_holes_inside_cut_path and len(best_cover_box) > 0:
                screw_boxes = holes_inside_cut_path

            # Visualise the cutting path.
            if draw:
                draw_lines(image_np, cut_path)

        return cut_path, screw_boxes, box_cname

    def construct_float_multi_array(self, path_points):
        # Publish Cutting Path.
        path_msg = Float32MultiArray()
        for x, y in path_points:
            path_msg.data.append(y)
            path_msg.data.append(x)
        return path_msg

    def publish_path(self, cut_path):
        path_msg = self.construct_float_multi_array(cut_path)
        self.path_publisher.publish(path_msg)

    def service_req(self, name, service_type, **inputs):
        _ = rospy.wait_for_service(name)
        try:
            callable_service_func = rospy.ServiceProxy(name, service_type)
            response = callable_service_func(**inputs["inputs"])
            return response
        except rospy.ServiceException as e:
            print("Service Failed : {}".format(e))


if __name__ == "__main__":

    rospy.init_node("components_detection")
    sys.path.insert(0, "/home/" + user + "/TensorFlow/workspace/yolov5")

    # Parameters that can be given from command-line / parameter-server.
    ns = "/components_detection"
    publish_flipping_plan_data = rospy.get_param(
        ns + "/publish_flipping_plan_data", False
    )
    publish_cut_path = rospy.get_param(ns + "/publish_cut_path", False)
    publish_screw_centers = rospy.get_param(ns + "/publish_screw_centers", False)
    use_state = rospy.get_param(ns + "/use_state", True)

    model = Model(
        model_path="/home/" + user + "/" + ws + "/src/perception/models/",
        image_topic="/camera/color/image_raw",
        cutting_plan_topic="/cutting_path",
        model_type="yolo",
        imgsz=832,
    )

    print(use_state)
    img_num = 10
    draw = False
    yolo_draw = False
    images_path = "/home/" + user + "/TensorFlow/workspace/training_demo/images/test/"
    if use_state:
        rospy.spin()
    else:
        while not rospy.is_shutdown():
            # Recieve an image msg from camera topic, then, return image and detections.
            image, detections = model.recieve_and_detect(
                read_img=True,
                image_path=images_path + str(img_num) + ".jpg",
                draw=yolo_draw,
            )

            # Generate the cover cutting path, and screw holes from given detections and image to visulaisSe on.
            cut_path, screw_holes = model.generate_cover_cutting_path(
                image,
                detections,
                min_screw_score=0,
                # generated path params
                tol=20,
                method=0,
                min_hole_dist=10,
                hole_tol=0,
                return_holes_inside_cut_path=False,
                filter_screws=False,
                draw=draw,
            )
            model.free_areas_detection(
                detections=detections, img=image, tol=20, draw=True
            )
            cv2.imshow("image", image)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord("e"):
                continue
            # Publish the generated flipping plan data if not empty.
            if publish_flipping_plan_data:
                flipping_plan_data = model.get_flipping_plan_data(detections)
                if len(flipping_plan_data) > 0:
                    print("Publishing flipping plan data")
                    model.publish_path(flipping_plan_data)

            # Publish the generated cutting path if not empty.
            if len(screw_holes) > 0 and publish_screw_centers:
                print("Publishing screw cut paths")
                # screw_centers = [(sh[0] + (sh[2] // 2), sh[1] + (sh[3] // 2)) for sh in screw_holes]
                cut_boxes = model.generate_rectangular_cutting_path(
                    screw_holes, interpolate=False
                )
                model.publish_path(cut_boxes[0])

            # Publish the generated cutting path if not empty.
            if len(cut_path) > 0 and publish_cut_path:
                model.publish_path(cut_path)

            print("Input Any Key to Continue")
            input()
            img_num += 1
            if img_num > 60:
                img_num = 1
