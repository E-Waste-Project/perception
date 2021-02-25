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

from perception.laptop_perception_helpers import plan_cover_cutting_path
from perception.coco_datasets import convert_format
import cv2


class Model:
    def __init__(self,
                 model_path='/home/zaferpc/abb_ws/src/perception/models/',
                 image_topic='/camera/color/image_raw',
                 cutting_plan_topic='/cutting_path'):
        
        PATH_TO_MODEL_DIR = model_path + 'saved_model'
        PATH_TO_LABELS = model_path + 'label_map.pbtxt'

        print('Loading model... ', end='')
        start_time = time.time()

        # Load the save tensorflow model.
        self.detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR) 
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))

        self.image_topic = image_topic
        self.image = None

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
        # Wait for rgb camera stream to publish a frame.
        image_msg = rospy.wait_for_message(self.image_topic, Image)
        
        # Convert msg to numpy image.
        image_np = numpify(image_msg)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run forward prop of model to get the detections.
        detections = self.detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: threshue[0, :num_detections].numpy()
                      for key, threshue in detections.items()}
        detections['num_detections'] = num_detections

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
        
        return image, detections
    
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
    
    def generate_cover_cutting_path(self, image, detections, min_screw_score=0):
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
                                        method=1, interpolate=True, interp_step=2, tol=100, min_hole_dist=3)
        
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
        
        # Visualise the cutting path.
        for i in range(len(cut_path) - 1):
            cv2.line(image_np, tuple(cut_path[i]), tuple(cut_path[i+1]), (0, 0, 255), 2)    

        # Show image and wait for pressed key to continue or exit(if key=='e').
        cv2.imshow("image_window", image_np)
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('e'):
            return
        
        return cut_path

    def publish_cut_path(self, cut_path):
        # Publish Cutting Path.
        path_msg = Float32MultiArray()
        for x, y in cut_path:
            path_msg.data.append(y)
            path_msg.data.append(x)
        self.path_publisher.publish(path_msg)


if __name__ == "__main__":
    
    rospy.init_node("components_detection")

    model = Model(model_path='/home/zaferpc/abb_ws/src/perception/models/',
                  image_topic='/camera/color/image_raw',
                  cutting_plan_topic="/cutting_path")

    # Recieve an image msg from camera topic, then, return image and detections.
    image, detections = model.recieve_and_detect()

    # Generate the cover cutting path from given detections and visualise on image.
    cut_path = model.generate_cover_cutting_path(image, detections, min_screw_score=0)

    # Publish the generated cutting path if not empty.
    if cut_path is not None:
        model.publish_cut_path(cut_path)
