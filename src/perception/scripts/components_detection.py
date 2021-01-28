#!/usr/bin/env python
import rospy
import os
import time
import pathlib

import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from PIL import Image, ImageFont, ImageDraw 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings


class Model:
    def __init__(self, model_path='/home/zaferpc/abb_ws/src/Disassembly-Perception/src/perception/models/'):
        PATH_TO_MODEL_DIR = model_path + 'saved_model'
        PATH_TO_LABELS = model_path + 'label_map.pbtxt'

        # print('Loading model... ', end='')
        # start_time = time.time()

        self.detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR) 
        
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
        
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print('Done! Took {} seconds'.format(elapsed_time))


def get_img_paths(data_dir="images/cover_on/test/", size=10):
    filenames = [str(i) + '.jpg' for i in range(1, size)]
    image_paths = [data_dir + filename for filename in filenames]
    return image_paths

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


if __name__ == "__main__":

    rospy.init_node("components_detection")

    model = Model()

    i = 0
    min_score_thresh = {1: 0.5,  # Battery
                        2: 0.20,  # Connector
                        3: 0.80,  # CPU
                        4: 0.80, # Fan
                        5: 0.50,  # Hard Disk
                        6: 0.80,  # Motherboard
                        7: 0.70,  # RAM
                        8: 0.20,  # Screw
                        9: 0.80,  # SSD
                        10: 0.70,  # WLAN
                        11: 0.50,  # CD-ROM
                        12: 0.92}  # Laptop_Back_Cover

    IMAGE_PATHS = get_img_paths("/home/zaferpc/data/laptop_components/exp_auto_direct_light/imgs/")
    
    visualize_detection = True

    for image_path in IMAGE_PATHS:
        i += 1
        print('Running inference for {}... '.format(image_path), end='')

        image_np = load_image_into_numpy_array(image_path)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = model.detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        label_id_offset = 0
        image_np_with_detections = image_np.copy()
        
        indicies_to_remove = []
        for j, score in enumerate(detections['detection_scores']):
            min_score = min_score_thresh[detections['detection_classes'][j] + label_id_offset]
            if score < min_score:
                indicies_to_remove.append(j)

        detections['detection_boxes'] = np.delete(detections['detection_boxes'], indicies_to_remove, axis=0)
        detections['detection_classes'] = np.delete(detections['detection_classes'], indicies_to_remove)
        detections['detection_scores'] = np.delete(detections['detection_scores'], indicies_to_remove)

        
        if visualize_detection:
            viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    model.category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=100,
                    min_score_thresh=.20,
                    agnostic_mode=False)

            plt.imsave("/home/zaferpc/{}.jpg".format(i), image_np_with_detections)
        print('Done')