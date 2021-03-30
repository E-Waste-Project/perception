#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from ros_numpy import numpify

import numpy as np
import tensorflow as tf

import time
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
import sys

from perception.laptop_perception_helpers import plan_cover_cutting_path, adjust_hole_center
from perception.coco_datasets import convert_format
from perception.yolo_detector import Yolo
import cv2


class Model:
    def __init__(self,
                 model_type='ssd',
                 model_path='/home/zaferpc/abb_ws/src/perception/models/',
                 image_topic='/camera/color/image_raw',
                 cutting_plan_topic='/cutting_path',
                 imgsz=1920):
        
        self.model_type = model_type
        if model_type == 'ssd':

            PATH_TO_MODEL_DIR = model_path + 'saved_model'
            PATH_TO_LABELS = model_path + 'label_map.pbtxt'

            print('Loading model... ', end='')
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
            PATH_TO_MODEL_DIR = model_path + 'last.pt'
            self.detect_fn = Yolo(PATH_TO_MODEL_DIR, imgsz).detect
            # define class thresholds, ids, and names.
            self.class_thresh = {
                'Connector':         {'thresh': 0.2, 'id': 0},
                'CPU':               {'thresh': 0.8, 'id': 1},
                'Fan':               {'thresh': 0.8, 'id': 2},
                'Hard Disk':         {'thresh': 0.5, 'id': 3},
                'Motherboard':       {'thresh': 0.8, 'id': 4},
                'RAM':               {'thresh': 0.7, 'id': 5},
                'Screw':             {'thresh': 0.2, 'id': 6},
                'SSD':               {'thresh': 0.8, 'id': 7},
                'Battery':           {'thresh': 0.5, 'id': 8},
                'WLAN':              {'thresh': 0.7, 'id': 9},
                'CD-ROM':            {'thresh': 0.5, 'id': 10},
                'Laptop_Back_Cover': {'thresh': 0.92, 'id': 11}
            }

        self.image_topic = image_topic
        self.image = None


        # dictionary to convert class id to class name
        self.cid_to_cname = {vals['id']: cname for cname, vals in self.class_thresh.items()}

        # dictionary to convert class name to class id
        self.cname_to_cid = {cname: vals['id'] for cname, vals in self.class_thresh.items()}

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
    
    def recieve_and_detect(self):
        # # Wait for rgb camera stream to publish a frame.
        # image_msg = rospy.wait_for_message(self.image_topic, Image)
        
        # # Convert msg to numpy image.
        # image_np = numpify(image_msg)

        image_np = img = cv2.imread(
            '/home/abdelrhman/bag_files/laptop_back_cover/exp_1500_no_direct_light_full_hd/imgs/1.jpg')

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
    
    def get_class_detections(self, detections, class_name,
                             format=('x1', 'y1', 'w', 'h'),
                             get_scores=False,
                             min_score=0):
        """
        Extract detections of a specific class from all the detections in the image.
        """
        boxes = []
        scores = []
        for i, cid in enumerate(detections['detection_classes']):
            if cid == self.cname_to_cid[class_name]:
                box = convert_format(detections['detection_boxes'][i], in_format=('y1', 'x1', 'y2', 'x2'), out_format=format)
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
                                    min_hole_dist=3):
        """Waits for an image msg from camera topic, then applies detection model
        to get the detected classes, finally generate a cutting path and publish it.

        param min_screw_score: a score threshold for detected screws confidence.
        """
        
        # Get detected laptop_covers.
        cover_boxes, cover_scores = self.get_class_detections(detections=detections,
                                                            class_name='Laptop_Back_Cover',
                                                            get_scores=True)
        
        # Get laptop_cover with highest confidence.
        best_cover_score = 0
        best_cover_box = None
        for box, score in zip(cover_boxes, cover_scores):
            if score > best_cover_score:
                best_cover_score = score
                best_cover_box = box
        
        # Get screw holes.
        screw_boxes = self.get_class_detections(detections=detections,
                                                class_name='Screw',
                                                min_score=min_screw_score)
        
        # Get Cutting Path.
        cut_path = plan_cover_cutting_path(laptop_coords=best_cover_box,
                                           holes_coords=screw_boxes,
                                           method=0,
                                           interpolate=True, interp_step=4,
                                           tol=tol, min_hole_dist=min_hole_dist)
        
        image_np = np.copy(image)

        # Visualise detected laptop_cover
        for i in range(len(best_cover_box) - 1):
            cv2.rectangle(image_np, tuple(best_cover_box[0:2]),
                        (best_cover_box[0] + best_cover_box[2], best_cover_box[1] + best_cover_box[3]), (0, 0, 255), 2)
        
        # Visualise detected screws.
        for screw_box in screw_boxes:
            for i in range(len(screw_box) - 1):
                cv2.rectangle(image_np, tuple(screw_box[0:2]),
                        (screw_box[0] + screw_box[2], screw_box[1] + screw_box[3]), (0, 0, 255), 2)
                cv2.circle(
                    image_np, (screw_box[0] + screw_box[2] // 2, screw_box[1] + screw_box[3] // 2), 1, (0, 0, 255), 2)
        
        # Visualise the cutting path.
        for i in range(len(cut_path) - 1):
            cv2.line(image_np, tuple(cut_path[i]), tuple(cut_path[i+1]), (0, 0, 255), 2)    

        # Show image and wait for pressed key to continue or exit(if key=='e').
        cv2.imshow("image_window", image_np)
        key = cv2.waitKey(0) & 0xFF
        
        cv2.destroyAllWindows()

        if key == ord('e'):
            return None, None
        
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
        0, '/home/abdelrhman/TensorFlow/workspace/yolov5')

    publish_cut_path = False
    publish_screw_centers = True

    model = Model(model_path='/home/abdelrhman/ewaste_ws/src/perception/models/',
                  image_topic='/camera/color/image_raw',
                  cutting_plan_topic="/cutting_path", model_type='yolo')

    while not rospy.is_shutdown():
        # Recieve an image msg from camera topic, then, return image and detections.
        image, detections = model.recieve_and_detect()

        # Generate the cover cutting path from given detections and image to visulaise on.
        cut_path, screw_holes = model.generate_cover_cutting_path(image, detections,
                                                                  min_screw_score=0,
                                                                  tol=50, min_hole_dist=20) # generated path params

        # screw_centers = adjust_hole_center(image, screw_holes)
        
        # Publish the generated cutting path if not empty.
        if screw_holes is not None and publish_screw_centers:
            screw_centers = [(sh[0] + (sh[2] // 2), sh[1] + (sh[3] // 2)) for sh in screw_holes]
            model.publish_cut_path(screw_centers)

        # Publish the generated cutting path if not empty.
        if cut_path is not None and publish_cut_path:
            model.publish_cut_path(cut_path)
        
        print("Input Any Key to Continue")
        input()
