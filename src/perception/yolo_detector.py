import time

import torch
import numpy as np
import cv2

from perception.models.experimental import attempt_load
from perception.utils.datasets import letterbox
from perception.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from perception.utils.torch_utils import select_device, time_synchronized


class Yolo:

    def __init__(self, weights, imgsz, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False):
        self.conf_thres, self.iou_thres, self.classes = conf_thres, iou_thres, classes
        self.agnostic_nms = agnostic_nms
        # Initialize
        set_logging()
        self.device = select_device()
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(
            weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(
            imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        # run once
        _ = self.model(
            img.half() if self.half else img) if self.device.type != 'cpu' else None

        print(f'Done. ({time.time() - t0:.3f}s)')

    def detect(self, input_img, augment=False):
        # Preprocess img
        # Padded resize
        img = letterbox(input_img, new_shape=self.imgsz)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # Get names and colors
        names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t2 = time_synchronized()

        detections = {}
        # Process detections
        for det in pred:  # detections per image
            s, im0 = '', input_img
            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                detections['detection_classes'] = []
                detections['detection_boxes'] = []
                detections['detection_scores'] = []
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                      ) / gn).view(-1).tolist()  # normalized xywh
                    detections['detection_boxes'].append(xywh)
                    detections['detection_scores'].append(conf)
                    detections['detection_classes'].append(cls)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            return detections


if __name__ == "__main__":
    model = Yolo(
        '/home/abdelrhman/TensorFlow/workspace/yolov5/runs/train/exp4/weights/last.pt', 832)
    img = cv2.imread(
        '/home/abdelrhman/bag_files/laptop_components/exp_1500_no_direct_light_full_hd/imgs/1.jpg')
    detections = model.detect(img)
    for i, box in enumerate(detections['detection_boxes']):
        if detections['detection_classes'][i] != 6:
            continue
        box = [box[0] * img.shape[1], box[1] *
               img.shape[0], box[2] * img.shape[1], box[3] * img.shape[0]]
        box = [int(b) for b in box]
        cv2.rectangle(img, (box[0]-box[2]//2, box[1]-box[3]//2), (box[0]+box[2]//2, box[1]+box[3]//2), (255, 0, 0), 2)
    cv2.imshow('detections', img)
    cv2.waitKey(0)
