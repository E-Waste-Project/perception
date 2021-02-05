#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from rospy_numpy import numpify, msgify

import numpy as np
import tensorflow as tf

import time
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings


class Model:
    def __init__(self, model_path='/home/zaferpc/abb_ws/src/Disassembly-Perception/src/perception/models/'):
        
        PATH_TO_MODEL_DIR = model_path + 'saved_model'
        PATH_TO_LABELS = model_path + 'label_map.pbtxt'

        print('Loading model... ', end='')
        start_time = time.time()

        # Load the save tensorflow model.
        self.detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR) 
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))

        # define class thresholds, ids, and names.
        self.class_thresh = {
            'Battery':           {'thresh': 0.5, 'id': 0},
            'Connector':         {'thresh': 0.2, 'id': 1},
            'CPU':               {'thresh': 0.8, 'id': 2},
            'Fan':               {'thresh': 0.8, 'id': 3},
            'Hard Disk':         {'thresh': 0.5, 'id': 4},
            'Motherboard':       {'thresh': 0.8, 'id': 5},
            'RAM':               {'thresh': 0.7, 'id': 6},
            'Screw':             {'thresh': 0.2, 'id': 7},
            'SSD':               {'thresh': 0.8, 'id': 8},
            'WLAN':              {'thresh': 0.7, 'id': 9},
            'CD-ROM':            {'thresh': 0.5, 'id': 10},
            'Laptop_Back_Cover': {'thresh': 0.92, 'id': 11}
        }

        # dictionary to convert class id to class name
        self.cid_to_cname = {vals['id']: cname for cname, vals in self.class_thresh.items()}

        # dictionary to convert class name to class id
        self.cname_to_cid = {cname: vals['id'] for cname, vals in self.class_thresh.items()}

        # dictionary to convert class id to class threshold
        self.cid_to_cthresh = {vals['id']: vals['thresh'] for cname,
                        vals in self.class_thresh.items()}

    def remove_detections(self, detections, indicies_to_remove):
        detections['detection_boxes'] = np.delete(
            detections['detection_boxes'], indicies_to_remove, axis=0)
        detections['detection_classes'] = np.delete(
            detections['detection_classes'], indicies_to_remove)
        detections['detection_scores'] = np.delete(
            detections['detection_scores'], indicies_to_remove)
    
    def recieve_and_detect(self):
        # Wait for rgb camera stream to publish a frame.
        image_msg = rospy.wait_for_message('/camera/color/image_raw', Image)
        
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

        return detections
    
    def get_class_detections(self, detections, class_id, get_scores=False, min_score=0):
        boxes = []
        scores = []
        for i, cid in enumerate(detections['detection_classes']):
            if cid == class_id:
                box = detections['detection_boxes'][i]
                boxes.append(box)
                score = detections['detection_scores'][i]
                if get_scores and (score >= min_score):
                    scores.append(score)
        
        if get_scores:
            return boxes, scores
        else:
            return boxes
    
    def generate_and_publish_cutting_path(self):
        detections = self.recieve_and_detect()
        
        # Get laptop_cover with highest confidence.
        cover_boxes, cover_scores = self.get_class_detections(detections, 
        self.cname_to_cid['Laptop_Back_Cover'], get_scores=True)

        best_cover_score_index = 0
        max_score = 0
        for i, score in enumerate(cover_scores):
            pass
            
        
        
        

if __name__ == "__main__":

    rospy.init_node("components_detection")

    model = Model()
