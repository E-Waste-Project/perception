#!/usr/bin/env python
import cv2
from perception.yolo_detector import Yolo
from perception.coco_datasets import convert_format
from perception.laptop_perception_helpers import plan_cover_cutting_path, interpolate_path,\
                                                 plan_port_cutting_path, filter_boxes_from_image,\
                                                     draw_lines, draw_boxes, box_near_by_dist, box_to_center
from perception.msg import PerceptionData
import sys
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from ros_numpy import numpify

from math import sqrt
import numpy as np
import tensorflow as tf
import time
import socket
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

user = socket.gethostname()
ws = 'ewaste_ws' if user == 'abdelrhman' else 'abb_ws'

class Model:
    def __init__(self,
                 model_type='ssd',
                 model_path='/home/' + user + '/' + ws + '/src/perception/models/',
                 image_topic='/camera/color/image_raw',
                 cutting_plan_topic='/cutting_path',
                 perception_data_topic='/perception_data',
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
                'Laptop_Back_Cover': {'thresh': 0.6, 'id': 11},
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
        self.screws_near_cpu = None

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
        self.perception_data_publisher = rospy.Publisher(perception_data_topic,
                                              PerceptionData,
                                              queue_size=1)
        self.state_publisher = rospy.Publisher(
            "/found_components", String, queue_size=1)
        rospy.Subscriber("/capture_state", String, self.capture_callback)
    
    def capture_callback(self, msg):
        # Recieve an image msg from camera topic, then, return image and detections.
        image, detections = self.recieve_and_detect()

        # Generate the cover cutting path, and screw holes from given detections and image to visulaise on.
        cut_path, screw_holes = self.generate_cover_cutting_path(image, detections,
                                                                  min_screw_score=0,
                                                                  tol=20, method=1, min_hole_dist=10, hole_tol=0,  # generated path params
                                                                  return_holes_inside_cut_path=False,
                                                                  filter_screws=False,
                                                                  avoid_screws_near_cpu=False)
        
        # Generate the cover cutting path, and screw holes from given detections and image to visulaisSe on.
        ports_cut_path = self.generate_ports_cutting_path(image, detections, draw=True)

        # Generate the screws cut paths
        screws_cut_path = self.generate_screws_cutting_path(screw_holes, interpolate=False)
        
        # Generate the screws_near_cpu cut paths
        screws_near_cpu_cut_path = self.generate_screws_cutting_path(self.screws_near_cpu, interpolate=False)
        
        # Construct perception_data msg.
        data_msg = PerceptionData()
        
        # Add back cover cut path
        data_msg.back_cover_cut_path = self.construct_float_multi_array(cut_path)
        for i in range(len(ports_cut_path)):
            path_msg = self.construct_float_multi_array(ports_cut_path[i])
            data_msg.ports_cut_path.append(path_msg)
        
        # Add screws    
        screw_boxes = []
        [screw_boxes.extend([(sb[0], sb[1]), (sb[2], sb[3])]) for sb in screw_holes]
        data_msg.screws = self.construct_float_multi_array(screw_boxes)
        
        # Add cpu screws
        cpu_screw_boxes = []
        [cpu_screw_boxes.extend([(sb[0], sb[1]), (sb[2], sb[3])]) for sb in self.screws_near_cpu]
        data_msg.screws_near_cpu = self.construct_float_multi_array(cpu_screw_boxes)
        
        # Add screws cut path
        for i in range(len(screws_cut_path)):
            path_msg = self.construct_float_multi_array(screws_cut_path[i])
            data_msg.screws_cut_path.append(path_msg)
        
        # Add cpu screws cut path
        for i in range(len(screws_near_cpu_cut_path)):
            path_msg = self.construct_float_multi_array(screws_near_cpu_cut_path[i])
            data_msg.screws_near_cpu_cut_path.append(path_msg)
        
        # Add detected CD-ROM.
        data_msg.cd_rom = self.get_detection_as_msg(class_name="CD-ROM", best_only=True)
        
        # Add detected Hard Disk.
        data_msg.hard_disk = self.get_detection_as_msg(class_name="Hard Disk", best_only=True)
        
        # Add detected Fan.
        data_msg.fan = self.get_detection_as_msg(class_name="Fan", best_only=True)
        
        # Add detected CPUs.
        data_msg.cpu = self.get_detection_as_msg(class_name="CPU", best_only=False)
        
        # Add detected motherboard.
        data_msg.motherboard = self.get_detection_as_msg(class_name="Motherboard", best_only=True)

        # Publish Perception Data
        self.perception_data_publisher.publish(data_msg)
        
        components_msg = String()
        components_msg.data = 'components'
        cut_boxes = []

        # Publish the generated flipping plan data if not empty.
        if publish_flipping_plan_data:
            flipping_plan_data = model.get_flipping_plan_data(detections)
            if len(flipping_plan_data) > 0:
                print("Publishing flipping plan data")
                cut_boxes.extend(flipping_plan_data)
        
        # Publish the generated cutting path if not empty.
        if len(cut_path) > 0 and publish_cut_path:
            cut_boxes.extend(cut_path)
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
    
    def get_detection_as_msg(self, class_name, best_only=False, preprocessor=None):
        preprocessor = box_to_center if preprocessor is None else preprocessor
        cpu_boxes = self.get_class_detections(detections=detections, class_name=class_name)
        boxes = boxes[0] if best_only else boxes
        processed_boxes = [preprocessor(box) for box in boxes]
        return self.construct_float_multi_array(processed_boxes)

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
                    self.state_publisher.publish(components_msg)
                    print("Camera Disconnected")
                    connection_msg = rospy.wait_for_message(
                        "/connection_error_handled", String)
                    continue

            # Convert msg to numpy image.
            image_np = numpify(image_msg)

            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        self.image = image_np
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

        detections['detection_boxes'][:, 0] *= image_np.shape[0]
        detections['detection_boxes'][:, 2] *= image_np.shape[0]
        detections['detection_boxes'][:, 1] *= image_np.shape[1]
        detections['detection_boxes'][:, 3] *= image_np.shape[1]

        return image_np, detections

    def get_class_detections(self, detections, class_name, min_score=0, best_only=False, get_scores=False, format=(
            'x1', 'y1', 'w', 'h'), sort=True):
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
        
    def get_flipping_plan_data(self, detections):
        box_cname = 'Laptop_Back_Cover'
        # Get detected laptop_covers.
        best_cover_box = self.get_class_detections(detections=detections,
                                                   class_name=box_cname,
                                                   best_only=True)
        x, y, w, h = best_cover_box[0], best_cover_box[1], best_cover_box[2], best_cover_box[3]
        flip_point = (x + w // 2, y + h)
        arc_center_point = (x + w // 2, y)
        return [flip_point, arc_center_point]
    
    def generate_screws_cutting_path(self, screw_boxes, interpolate=False, npoints=20):
        cut_boxes = []
        for sh in screw_boxes:
            x, y, x2, y2 = sh[0], sh[1], sh[0] + sh[2], sh[1] + sh[3]
            box_path = [(x, y), (x, y2), (x2, y2), (x2, y), (x, y)]
            box_path = interpolate_path(box_path, npoints=npoints) if interpolate else box_path
            cut_boxes.extend(box_path)
        return cut_boxes
    
    
    def generate_ports_cutting_path(self, image, detections, draw=True):
        """Generate a cutting path and publish it.
        """

        image_np = np.copy(image)

        box_cname = 'Motherboard'
        # Get detected laptop_covers.
        cover_boxes, cover_scores = self.get_class_detections(detections=detections,
                                                              class_name=box_cname,
                                                              get_scores=True)
        
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
        for cname in ['Port', 'Connector']:
            port_boxes.extend(self.get_class_detections(detections=detections, class_name=cname))
         
        # Plan the Cutting Path.
        ports_cut_paths = []
        if len(best_box) > 0 and len(port_boxes) > 0:
            ports_cut_paths = plan_port_cutting_path(
                best_box, port_boxes, near_edge_dist=20, grouping_dist=40, cutting_dist=5)
        
        # Visualise the cutting path.
        # Visualise detected screws/ports/connectors.
        if draw:
            draw_boxes(image_np, port_boxes)
            for ports_cut_path in ports_cut_paths:
                draw_lines(image_np, ports_cut_path)

        cv2.imshow("image_window", image_np)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("image_window")

        if key == ord('e'):
            return [], []

        return ports_cut_path
    
    
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
                                    avoid_screws_near_cpu=False,
                                    draw=True):
        """Generate a cutting path and publish it.
        """

        image_np = np.copy(image)

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
            if draw:
                draw_boxes(image_np, [best_cover_box])
        else:
            return [], []
        
        screw_boxes = self.get_class_detections(detections=detections,
                                                class_name='Screw',
                                                min_score=min_screw_score)
        
        if avoid_screws_near_cpu:
            cpu_boxes = self.get_class_detections(detections=detections,
                                                class_name='CPU')

            self.screws_near_cpu = list(filter(lambda box: box_near_by_dist(box, cpu_boxes, min_hole_dist), screw_boxes))
            
            screw_boxes = [box for box in screw_boxes if box not in self.screws_near_cpu]


        screw_boxes = [[sb[0] - hole_tol, sb[1] - hole_tol, sb[2] +
                        2*hole_tol, sb[3] + 2*hole_tol] for sb in screw_boxes]
        
        # Visualise detected screws/ports/connectors.
        if draw:
            draw_boxes(image_np, screw_boxes, draw_center=True)

        if filter_screws:
            screw_boxes = filter_boxes_from_image(screw_boxes, self.image, "Choose Boxes, then press 'c'")
            # Visualise detected screws.
            draw_boxes(image_np, screw_boxes)
        
        # Plan the Cutting Path.
        cut_path = []
        if len(best_cover_box) > 0 and len(screw_boxes) > 0:
            cut_path, holes_inside_cut_path = plan_cover_cutting_path(laptop_coords=best_cover_box,
                                                                        holes_coords=screw_boxes,
                                                                        method=method,
                                                                        interpolate=False, interp_step=4,
                                                                        tol=tol, min_hole_dist=min_hole_dist)

            if return_holes_inside_cut_path and len(best_cover_box) > 0:
                screw_boxes = holes_inside_cut_path
        
        # Visualise the cutting path.
        draw_lines(image_np, cut_path)

        cv2.imshow("image_window", image_np)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("image_window")

        if key == ord('e'):
            return [], []
        
        return cut_path, screw_boxes

    
    def construct_float_multi_array(self, cut_path):
        # Publish Cutting Path.
        path_msg = Float32MultiArray()
        for x, y in cut_path:
            path_msg.data.append(y)
            path_msg.data.append(x)
        return path_msg
    
    
    def publish_path(self, cut_path):
        path_msg = self.construct_float_multi_array(cut_path)
        self.path_publisher.publish(path_msg)


if __name__ == "__main__":

    rospy.init_node("components_detection")

    sys.path.insert(
        0, '/home/' + user + '/TensorFlow/workspace/yolov5')

    # Parameters that can be given from command-line / parameter-server.
    ns = '/components_detection'
    publish_flipping_plan_data = rospy.get_param(ns+'/publish_flipping_plan_data', False)
    publish_cut_path = rospy.get_param(ns+'/publish_cut_path', False)
    publish_screw_centers = rospy.get_param(ns+'/publish_screw_centers', False)
    use_state = rospy.get_param(ns+'/use_state', True)

    model = Model(model_path='/home/' + user + '/' + ws + '/src/perception/models/',
                  image_topic='/camera/color/image_raw',
                  cutting_plan_topic="/cutting_path", model_type='yolo', imgsz=832)

    img_num = 10
    images_path = '/home/' + user + '/TensorFlow/workspace/training_demo/images/test/'
    if use_state:
        rospy.spin()
    else:
        while not rospy.is_shutdown():
            # Recieve an image msg from camera topic, then, return image and detections.
            image, detections = model.recieve_and_detect(
                read_img=False,
                image_path= images_path + str(img_num) + '.jpg')

            # Generate the cover cutting path, and screw holes from given detections and image to visulaisSe on.
            cut_path, screw_holes = model.generate_cover_cutting_path(image, detections,
                                                                    min_screw_score=0,
                                                                    # generated path params
                                                                    tol=20, method=0, min_hole_dist=10, hole_tol=0,
                                                                    return_holes_inside_cut_path=False,
                                                                    filter_screws=False,
                                                                    avoid_screws_near_cpu=True)
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
                cut_boxes = model.generate_screws_cutting_path(screw_holes, interpolate=False)
                model.publish_path(cut_boxes)
            
            # Publish the generated cutting path if not empty.
            if len(cut_path) > 0 and publish_cut_path:
                model.publish_path(cut_path)

            print("Input Any Key to Continue")
            input()
            img_num += 1
            if img_num > 17:
                img_num = 10
    
