#!/usr/bin/env python
from itertools import filterfalse
import cv2
from perception.yolo_detector import Yolo
from perception.coco_datasets import convert_format
from perception.laptop_perception_helpers import plan_cover_cutting_path, interpolate_path,\
                                                 plan_port_cutting_path, filter_boxes_from_image,\
                                                     draw_lines, draw_boxes, box_near_by_dist
import sys
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from ros_numpy import numpify

from math import sqrt
import numpy as np
import tensorflow as tf
import time
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings


user = 'abdelrhman'
# user = 'zaferpc'
ws = 'ewaste_ws' if user == 'abdelrhman' else 'abb_ws'


class Model:
    def __init__(self,
                 model_type='ssd',
                 model_path='/home/' + user + '/' + ws + '/src/perception/models/',
                 image_topic='/camera/color/image_raw',
                 cutting_plan_topic='/cutting_path',
                 imgsz=1280):

        self.model_type = model_type
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

                detections = model(input_tensor)
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

        elif model_type == 'yolo':
            PATH_TO_MODEL_DIR = model_path + 'best.pt'
            self.detect_fn = Yolo(PATH_TO_MODEL_DIR, imgsz).detect
            # define class thresholds, ids, and names.
            self.class_thresh = {
                'Connector':         {'thresh': 0.2, 'id': 0},
                'CPU':               {'thresh': 0.8, 'id': 1},
                'Fan':               {'thresh': 0.8, 'id': 2},
                'Hard Disk':         {'thresh': 0.5, 'id': 3},
                'Motherboard':       {'thresh': 0.7, 'id': 4},
                'RAM':               {'thresh': 0.7, 'id': 5},
                'Screw':             {'thresh': 0.3, 'id': 6},
                'SSD':               {'thresh': 0.8, 'id': 7},
                'Battery':           {'thresh': 0.5, 'id': 8},
                'WLAN':              {'thresh': 0.7, 'id': 9},
                'CD-ROM':            {'thresh': 0.5, 'id': 10},
                'Laptop_Back_Cover': {'thresh': 0.5, 'id': 11},
                'Port':              {'thresh': 0.2, 'id': 12},
                'RTC_Battery':       {'thresh': 0.5, 'id': 13},
                'HD_Bay':            {'thresh': 0.5, 'id': 14},
                'CD_Bay':            {'thresh': 0.5, 'id': 15},
                'Battery_Bay':       {'thresh': 0.5, 'id': 16},
                'SD_Slot':           {'thresh': 0.5, 'id': 17}
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

    def remove_detections(self, detections, indicies_to_remove):
        detections['detection_boxes'] = np.delete(
            detections['detection_boxes'], indicies_to_remove, axis=0)
        detections['detection_classes'] = np.delete(
            detections['detection_classes'], indicies_to_remove)
        detections['detection_scores'] = np.delete(
            detections['detection_scores'], indicies_to_remove)

    def recieve_and_detect(self,image_path=None, read_img=False):
        if image_path is not None and read_img:
            image_np = cv2.imread(image_path)
            image_np = cv2.resize(image_np, (832, 480))
        else:
            # # Wait for rgb camera stream to publish a frame.
            image_msg = rospy.wait_for_message(self.image_topic, Image)

            # Convert msg to numpy image.
            image_np = numpify(image_msg)

            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # cv2.imshow("image", image_np)
        # cv2.waitKey(0)

        # Run forward prop of model to get the detections.
        detections = self.detect_fn(image_np)

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(
            np.int64)

        # Filter the detections by using the pre-defined thresholds.
        indicies_to_remove = []
        for j, score in enumerate(detections['detection_scores']):
            min_score = self.cid_to_cthresh[detections['detection_classes'][j]]
            if score < min_score:
                indicies_to_remove.append(j)

        self.remove_detections(detections, indicies_to_remove)

        print(image_np.shape)
        print(detections['detection_boxes'][0])
        detections['detection_boxes'][:, 0] *= image_np.shape[0]
        detections['detection_boxes'][:, 2] *= image_np.shape[0]
        detections['detection_boxes'][:, 1] *= image_np.shape[1]
        detections['detection_boxes'][:, 3] *= image_np.shape[1]

        return image_np, detections

    def get_class_detections(self, detections, class_name, min_score=0, get_scores=False, format=(
                    'x1', 'y1', 'w', 'h')):
        """
        Extract detections of a specific class from all the detections in the image.
        """
        boxes = []
        scores = []
        for i, cid in enumerate(detections['detection_classes']):
            if cid == self.cname_to_cid[class_name]:
                box = convert_format(detections['detection_boxes'][i], in_format=(
                    'y1', 'x1', 'y2', 'x2'), out_format=format)
                box = [int(x) for x in box]
                score = detections['detection_scores'][i]
                if score >= min_score:
                    boxes.append(box)
                    if get_scores:
                        scores.append(score)

        if get_scores:
            return boxes, scores
        else:
            return boxes
        
        
    def generate_cover_cutting_path(self,
                                    image,
                                    detections,
                                    min_screw_score=0,
                                    tol=100,
                                    method=0,
                                    min_hole_dist=3,
                                    hole_tol=0,
                                    return_holes_inside_cut_path=False,
                                    filter_screws=False,
                                    avoid_screws_near_cpu=False):
        """Waits for an image msg from camera topic, then applies detection model
        to get the detected classes, finally generate a cutting path and publish it.

        param min_screw_score: a score threshold for detected screws confidence.
        """

        image_np = np.copy(image)
        self.image = image_np

        box_cname = 'Motherboard'
        # Get detected laptop_covers.
        cover_boxes, cover_scores = self.get_class_detections(detections=detections,
                                                              class_name=box_cname,
                                                              get_scores=True)
        if len(cover_boxes) < 1:
            box_cname = 'Laptop_Back_Cover'
            # Get detected laptop_covers.
            cover_boxes, cover_scores = self.get_class_detections(detections=detections,
                                                              class_name=box_cname,
                                                              get_scores=True)
        best_cover_box = []
        if len(cover_boxes) > 0:
            # Get laptop_cover with highest confidence.
            best_cover_score = 0
            for box, score in zip(cover_boxes, cover_scores):
                if score > best_cover_score:
                    best_cover_score = score
                    best_cover_box = box

            # Visualise detected laptop_cover/Motherboard
            draw_boxes(image_np, [best_cover_box])
        
        screw_boxes = self.get_class_detections(detections=detections,
                                                class_name='Screw',
                                                min_score=min_screw_score)
        
        port_boxes = []
        for cname in ['Port', 'Connector']:
            port_boxes.extend(self.get_class_detections(detections=detections,
                                                            class_name=cname,
                                                            min_score=min_screw_score))
        
        if avoid_screws_near_cpu:
            cpu_boxes = self.get_class_detections(detections=detections,
                                                class_name='CPU',
                                                min_score=0)

            screw_boxes = filter(lambda box: not box_near_by_dist(box, cpu_boxes, min_hole_dist), screw_boxes)


        screw_boxes = [[sb[0] - hole_tol, sb[1] - hole_tol, sb[2] +
                        2*hole_tol, sb[3] + 2*hole_tol] for sb in screw_boxes]
        
        # Visualise detected screws/ports/connectors.
        draw_boxes(image_np, screw_boxes, draw_center=True)
        draw_boxes(image_np, port_boxes)

        if filter_screws:
            screw_boxes = filter_boxes_from_image(screw_boxes, self.image, "Choose Boxes, then press 'c'")
            # Visualise detected screws.
            draw_boxes(image_np, screw_boxes)
        
        # Plan the Cutting Path.
        ports_cut_paths = []
        cut_path = []
        if len(best_cover_box) > 0 and len(port_boxes) > 0:
            if box_cname == 'Motherboard':
                ports_cut_paths = plan_port_cutting_path(
                    best_cover_box, port_boxes, near_edge_dist=20, grouping_dist=40, cutting_dist=5)
            else:
                cut_path, holes_inside_cut_path = plan_cover_cutting_path(laptop_coords=best_cover_box,
                                                                          holes_coords=screw_boxes,
                                                                          method=method,
                                                                          interpolate=False, interp_step=4,
                                                                          tol=tol, min_hole_dist=min_hole_dist)

                if return_holes_inside_cut_path and len(best_cover_box) > 0:
                    screw_boxes = holes_inside_cut_path
        
        # Visualise the cutting path.
        if box_cname == 'Motherboard':
            for ports_cut_path in ports_cut_paths:
                draw_lines(image_np, ports_cut_path)
        else:
            draw_lines(image_np, cut_path)

        cv2.imshow("image_window", image_np)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("image_window")

        if key == ord('e'):
            return [], []

        if box_cname == 'Motherboard':
            return ports_cut_path, screw_boxes
        else:
            return cut_path, screw_boxes

    def publish_cut_path(self, cut_path):
        # Publish Cutting Path.
        path_msg = Float32MultiArray()
        for x, y in cut_path:
            path_msg.data.append(y)
            path_msg.data.append(x)
        self.path_publisher.publish(path_msg)


if __name__ == "__main__":

    rospy.init_node("components_detection")

    sys.path.insert(
        0, '/home/' + user + '/TensorFlow/workspace/yolov5')

    publish_cut_path = False
    publish_screw_centers = True

    model = Model(model_path='/home/' + user + '/' + ws + '/src/perception/models/',
                  image_topic='/camera/color/image_raw',
                  cutting_plan_topic="/cutting_path", model_type='yolo', imgsz=832)
    img_num = 10

    while not rospy.is_shutdown():
        images_path = '/home/abdelrhman/TensorFlow/workspace/training_demo/images/test/'
        # Recieve an image msg from camera topic, then, return image and detections.
        image, detections = model.recieve_and_detect(image_path=images_path + str(img_num) + '.jpg',
                                                     read_img=True)

        # Generate the cover cutting path, and screw holes from given detections and image to visulaisSe on.
        cut_path, screw_holes = model.generate_cover_cutting_path(image, detections,
                                                                  min_screw_score=0,
                                                                  tol=20, method=1, min_hole_dist=10, hole_tol=0,  # generated path params
                                                                  return_holes_inside_cut_path=False,
                                                                  filter_screws=False,
                                                                  avoid_screws_near_cpu=True)

        # Publish the generated cutting path if not empty.
        if len(screw_holes) > 0 and publish_screw_centers:
            print("Publishing screw cut paths")
            # screw_centers = [(sh[0] + (sh[2] // 2), sh[1] + (sh[3] // 2)) for sh in screw_holes]
            cut_boxes = []
            for sh in screw_holes:
                x, y, x2, y2 = sh[0], sh[1], sh[0] + sh[2], sh[1] + sh[3]
                box_path = [(x, y), (x, y2), (x2, y2), (x2, y), (x, y)]
                cut_boxes.extend(interpolate_path(box_path))
            cut_boxes.extend(cut_path)
            model.publish_cut_path(cut_boxes)
        # Publish the generated cutting path if not empty.
        if len(cut_path) > 0 and publish_cut_path:
            model.publish_cut_path(cut_path)

        print("Input Any Key to Continue")
        input()
        img_num += 1
        if img_num > 17:
            img_num = 10
