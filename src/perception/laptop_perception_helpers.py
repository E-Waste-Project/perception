import cv2
import numpy as np
from bisect import bisect_right, bisect_left
from copy import deepcopy
from perception.coco_datasets import convert_format
from math import fabs, sqrt, sin, cos, pi
import pyrealsense2 as rs2

from realsense2_camera.msg import Extrinsics
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseArray, Pose
import ros_numpy
import rospy


class Rect:

    color_dict = {"b": (255, 0, 0), "g": (0, 255, 0), "r": (0, 0, 255)}

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.w = width
        self.h = height
        self.x2 = self.x + self.w
        self.y2 = self.y + self.h

    def crop_img(self, input_img):
        return input_img[self.y : self.y2, self.x : self.x2].copy()

    def draw_on(self, input_img, color="g", thickness=1):
        if color not in self.color_dict.keys():
            raise ValueError(
                "Available Colors are 'b', 'g', and 'r' for blue, green, and red respectively"
            )
        cv2.rectangle(
            input_img,
            (self.x, self.y),
            (self.x2, self.y2),
            self.color_dict[color],
            thickness,
        )

    def shift_by(self, x, y):
        self.x += x
        self.x2 += x
        self.y += y
        self.y2 += y

    def enlarge_by(self, val):
        self.x -= val
        self.y -= val
        self.w += 2 * val
        self.h += 2 * val
        self.x2 += val
        self.y2 += val

    def add(self, rect):
        self.shift_by(rect.x, rect.y)
    
    def rect_to_list(self):
        return [self.x , self.y, self.w, self.h]

    def __eq__(self, rect):
        if (
            self.x == rect.x
            and self.y == rect.y
            and self.w == rect.w
            and self.h == rect.h
        ):
            return True
        else:
            return False

    def __ne__(self, rect):
        return False if self == rect else True

    def __str__(self):
        return "({0}, {1}, {2}, {3})".format(self.x, self.y, self.w, self.h)

    def rect_to_path(self):
        return [
            (self.x, self.y),
            (self.x, self.y2),
            (self.x2, self.y2),
            (self.x2, self.y),
            (self.x, self.y),
        ]


def nothing(x):
    pass


def enclosing_rect_area(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return w * h


def circularity(x):
    return (4 * cv2.contourArea(x) * np.pi) / (cv2.arcLength(x, True) ** 2 + 1e-7)


def circularity_using_moments(cnt):
    M = cv2.moments(cnt)
    return (M["m00"] ** 2) / (2 * np.pi * (M["m20"] + M["m02"]) + 1e-7)


def perimeter(x):
    return cv2.arcLength(x, True)


def aspect_ratio(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return abs((float(w) / h))


def euclidean_dist(point1, point2):
    diff_sq = 0
    for p1, p2 in zip(point1, point2):
        diff_sq += (p1 - p2) ** 2
    return sqrt(diff_sq)


def euclidean_dist_array(points, point):
    x_diff = points[:, 0] - point[0]
    y_diff = points[:, 1] - point[1]
    return np.sqrt(x_diff ** 2 + y_diff ** 2)


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled, (cx, cy)


def sample_from_func(start, func, step_sz, n_steps):
    return [func(start + step_sz * i) for i in range(n_steps)]


def get_line_func(p1, p2):
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    return lambda x: p2[1] - slope * (p2[0] - x)


def read_and_resize(directory, img_id, size=(720, 480), compression=".jpg"):
    read_img = cv2.imread(directory + str(img_id) + compression)
    if size is not None:
        read_img = cv2.resize(read_img, size)
    return read_img


def draw_lines(image_np, points_list, color=(0, 0, 255)):
    for i in range(len(points_list) - 1):
        cv2.line(image_np, tuple(points_list[i]), tuple(points_list[i + 1]), color, 2)


def draw_boxes(
    image,
    boxes,
    color=(0, 255, 0),
    thickness=2,
    draw_center=False,
    in_format=("x1", "y1", "w", "h"),
):
    for box in boxes:
        if len(box) < 1:
            return
        conv_box = convert_format(box, in_format=in_format)
        cv2.rectangle(
            image, tuple(conv_box[0:2]), tuple(conv_box[2:4]), color, thickness
        )
        if draw_center:
            cv2.circle(
                image,
                box_to_center(conv_box, in_format=("x1", "y1", "x2", "y2")),
                1,
                color,
                thickness,
            )


def box_to_center(box, in_format=("x1", "y1", "w", "h")):
    converted_box = convert_format(
        deepcopy(box), in_format=in_format, out_format=("x1", "y1", "w", "h")
    )
    center = (
        converted_box[0] + converted_box[2] // 2,
        converted_box[1] + converted_box[3] // 2,
    )
    return center


def point_near_box_by_dist(point, box, dist_as_side_ratio=0.5):
    x, y = point[0], point[1]
    x1, y1, w1, h1 = box[0], box[1], box[2], box[3]
    x2, y2 = x1 + w1, y1 + h1
    xc, yc = x1 + w1 // 2, y1 + h1 // 2
    if (
        fabs(xc - x) <= w1 * dist_as_side_ratio
        and fabs(yc - y) <= h1 * dist_as_side_ratio
    ):
        return True
    else:
        return False


def box_near_by_dist(box1, boxes, dist_as_side_ratio=0.5):
    """Checks if box1 is near any of the boxes by dist,
    if it is near any one of them, it returns True,
    else if it is not near any one of them, it returns False.
    """
    x1, y1, w, h = box1[0], box1[1], box1[2], box1[3]
    x2, y2 = x1 + w, y1 + h
    points = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
    center_point = x1 + w // 2, y1 + h // 2
    def points_near_box(points, box):
        for point in points:
            if point_near_box_by_dist(point, box, dist_as_side_ratio):
                return True
        return False
    for box in boxes:
        bx1, by1, bw, bh = box[0], box[1], box[2], box[3]
        bx2, by2 = bx1 + bw, by1 + bh 
        box_points = [(bx1, by1), (bx1, by2), (bx2, by2), (bx2, by1)]
        
        if points_near_box(points, box) or points_near_box(box_points, box1):
            return True
        
    return False


def filter_boxes_from_image(boxes, image, window_name, create_new_window=True):

    print("=======================================================================")
    print("Choose boxes by clicking on them on the image, Press 'c' when finished")
    print("=======================================================================")

    refPt = []
    img = image.copy()
    if create_new_window:
        cv2.namedWindow(window_name)

    def choose_screw(event, x, y, flags, param):
        # if the left mouse button was clicked, record the point
        if event == cv2.EVENT_LBUTTONUP:
            refPt.append((x, y))
            # draw a rectangle around the region of interest
            cv2.circle(img, refPt[-1], 5, (0, 255, 0), 5)

    cv2.setMouseCallback(window_name, choose_screw)
    key = 0
    # Show image and wait for pressed key to continue or exit(if key=='e').
    while key != ord("c"):
        cv2.imshow(window_name, img)
        key = cv2.waitKey(20) & 0xFF
    cv2.destroyWindow(window_name)
    filtered_boxes = []
    for point in refPt:
        px, py = point
        for box in boxes:
            x, y, w, h = box
            x1, y1 = x + w, y + h
            if x <= px <= x1 and y <= py <= y1:
                filtered_boxes.append(box)
    return filtered_boxes


def preprocess(input_img, **kwargs):
    # Contrast Norm + Gauss Blur + Adaptive Threshold + Dilation + Canny
    median_sz = kwargs.get("median_sz", 12)
    gauss_kernel = kwargs.get("gauss_kernel", 21)
    clahe_kernel = kwargs.get("clahe_kernel", 2)
    morph_kernel = kwargs.get("morph_kernel", 3)
    iterations = kwargs.get("iterations", 3)
    dilate = kwargs.get("dilate", False)
    use_canny = kwargs.get("use_canny", False)
    thresh1 = kwargs.get("thresh1", 42)
    thresh2 = kwargs.get("thresh2", 111)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(clahe_kernel, clahe_kernel))
    output_img = clahe.apply(input_img)

    output_img = cv2.medianBlur(output_img, ksize=median_sz)

    output_img = cv2.GaussianBlur(output_img, (gauss_kernel, gauss_kernel), 0)

    output_img = cv2.adaptiveThreshold(
        output_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    if use_canny:
        output_img = cv2.Canny(output_img, thresh1, thresh2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    if dilate:
        output_img = cv2.dilate(output_img, kernel, iterations=iterations)
    output_img = cv2.morphologyEx(
        output_img, cv2.MORPH_CLOSE, kernel, iterations=iterations
    )

    return output_img


def filter_contours(input_contours, **kwargs):
    sorting_key = kwargs.get("sorting_key", None)
    min_val = kwargs.get("min_val", None)
    max_val = kwargs.get("max_val", None)
    reverse = kwargs.get("reverse", False)

    sorted_contours = sorted(input_contours, key=sorting_key, reverse=reverse)
    if sorting_key is not None:
        contours_feature = list(map(sorting_key, sorted_contours))
    start = 0
    if min_val is not None:
        start = bisect_left(contours_feature, min_val)
    if max_val is not None:
        end = bisect_right(contours_feature, max_val)
        filtered_contours = sorted_contours[start : end + 1]
    else:
        filtered_contours = sorted_contours[start:]
    if not reverse:
        filtered_contours.reverse()
    return filtered_contours


def find_contours(input_img, **kwargs):
    # Takes gray_scale img, returns rect values of detected laptop.
    preprocessed_img = preprocess(input_img, **kwargs)

    all_contours, hierarchy = cv2.findContours(
        preprocessed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    filtered_contours = filter_contours(all_contours, **kwargs)
    return filtered_contours

def rotate_boxes(boxes, center, angle):
    # depth_image = deepcopy(self.color_dist_image)
    # (h, w) = depth_image.shape[:2]
    # (cX, cY) = box_to_center(laptop_px)
    # M = cv2.getRotationMatrix2D((cX, cY), self.laptop_angle, 1.0)
    # inv_M = cv2.getRotationMatrix2D((cX, cY), -self.laptop_angle, 1.0)
    # screw_pixels_arr = np.array(screw_pixels)
    # conv_cover_box = convert_format(best_cover_box)
    # best_cover_box_arr = np.array([[conv_cover_box[0], conv_cover_box[2]],
    #                                [conv_cover_box[1], conv_cover_box[3]]])
    # to_rotate = np.ones((3, screw_pixels_arr.shape[1]+2))
    # to_rotate[:-1, :-2] = screw_pixels_arr
    # to_rotate[:-1, -2:] = best_cover_box_arr
    # print("screw_boxes = ", screw_pixels_arr)
    # rotated_screws_boxes = np.round(np.dot(M, to_rotate)).astype(np.uint16)
    # rotated_cover = rotated_screws_boxes[:, -2:]
    # rotated_screws_boxes = rotated_screws_boxes[:, :-2]
    # print("rotated_screw_boxes = ", rotated_screws_boxes)
    # rotated = cv2.warpAffine(depth_image, M, (w, h))
    # cv2.imshow("Rotated by {} Degrees".format(-self.laptop_angle), rotated)
    
    
    # screw_boxes = rotated_screws_boxes.T.reshape((-1, 4), order="C").tolist()
    # rotated_cover_box = rotated_cover.T.reshape((-1, 4), order="C").tolist()[0]
    # rotated_cover_box = [rotated_cover_box[1], rotated_cover_box[0], rotated_cover_box[3], rotated_cover_box[2]]
    # print("rotated_cover_box = ", rotated_cover_box)
    # conv_rotated_cover_box = convert_format(
    #         rotated_cover_box,
    #         in_format=("x1", "y1", "x2", "y2"),
    #         out_format=("x1", "y1", "w", "h"),
    #     )
    pass

def detect_laptop(input_img, draw_on=None, **kwargs):
    # Takes gray_scale img, returns rect values of detected laptop.

    kwargs = {
        "min_val": 200000,
        "max_val": 500000,
        "gauss_kernel": 1,
        "median_sz": 33,
        "clahe_kernel": 2,
        "morph_kernel": 1,
        "iterations": 2,
        "dilate": False,
        "reverse": False,
        "sorting_key": enclosing_rect_area,
    }

    filtered_contours = find_contours(input_img, **kwargs)

    if len(filtered_contours) > 0:
        cnt = filtered_contours[0]

        # Find rotated rect parameters.
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        clamp_pixels = 0
        xs = list(enumerate([box[0], box[1], box[2], box[3]]))
        xs.sort(key=lambda x: x[1][0])

        box[xs[0][0]][0] += clamp_pixels
        box[xs[1][0]][0] += clamp_pixels
        box[xs[2][0]][0] -= clamp_pixels
        box[xs[3][0]][0] -= clamp_pixels

        left_points = box[0:2]
        (upper_left_corner, lower_left_corner) = (
            (xs[0][1], xs[1][1]) if xs[0][1][1] < xs[1][1][1] else (xs[1][1], xs[0][1])
        )
        (upper_right_corner, lower_right_corner) = (
            (xs[2][1], xs[3][1]) if xs[2][1][1] < xs[3][1][1] else (xs[3][1], xs[2][1])
        )
        box_list = [
            upper_left_corner,
            upper_right_corner,
            lower_right_corner,
            lower_left_corner,
        ]

        # Retrieve the key parameters of the rotated bounding box
        center = (int(rect[0][0]), int(rect[0][1]))
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(round(rect[2]))

        if width < height:
            angle = 90 - angle
            nw = (width // 2) * sin(angle * pi / 180)
            nh = (width // 2) * cos(angle * pi / 180)
            flip_radius = width
        else:
            angle = -angle
            nw = (height // 2) * sin(angle * pi / 180)
            nh = (height // 2) * cos(angle * pi / 180)
            flip_radius = height
        print("detected_flip_radius = ", flip_radius)
        print("box = ", box)
        print("rect = ", rect)
        flip_point = (int(center[0] + nw), int(center[1] + nh))
        upper_point = (int(center[0] + nw), int(center[1] - nh))
        laptop_data = [
            center[0],
            center[1],
            flip_point[0],
            flip_point[1],
            upper_point[0],
            upper_point[1],
            box_list[0][0],  # upper left corner x
            box_list[0][1],  # upper left corner y
            box_list[1][0],  # upper right corner x
            box_list[1][1],  # upper right corner y
            box_list[2][0],  # lower right corner x
            box_list[2][1],  # lower right corner y
            box_list[3][0],  # lower left corner x
            box_list[2][1],  # lower left corner y
            angle
        ]
        if draw_on is not None:
            cv2.drawContours(draw_on, [box], 0, (0, 255, 0), thickness=5)
            cv2.circle(draw_on, (center[0], center[1]), 10, (255, 0, 0), thickness=-1)
            cv2.circle(
                draw_on, (flip_point[0], flip_point[1]), 10, (0, 255, 0), thickness=-1
            )
            cv2.circle(
                draw_on, (upper_point[0], upper_point[1]), 10, (0, 0, 255), thickness=-1
            )

            cv2.circle(
                draw_on, (box_list[0][0], box_list[0][1]), 10, (255, 0, 0), thickness=-1
            )
            cv2.circle(
                draw_on, (box_list[1][0], box_list[1][1]), 10, (0, 255, 0), thickness=-1
            )
            cv2.circle(
                draw_on, (box_list[2][0], box_list[2][1]), 10, (0, 0, 255), thickness=-1
            )
        return laptop_data
    else:
        return None


def detect_picking_point(
    input_img,
    center=None,
    depth_img=None,
    dist_mat=None,
    draw_on=None,
    use_depth=False,
    use_center=False,
    method=0,
    **kwargs
):
    # Takes gray_scale img, returns rect values of detected laptop.

    all_kwargs = {
        "min_val": 0,
        "max_val": 500000,
        "gauss_kernel": 1,
        "median_sz": 23,
        "clahe_kernel": 2,
        "morph_kernel": 1,
        "iterations": 1,
        "dilate": False,
        "reverse": False,
        "sorting_key": enclosing_rect_area,
    }

    for key, val in kwargs.items():
        if key in all_kwargs.keys():
            all_kwargs[key] = val

    filtered_contours = find_contours(input_img, **all_kwargs)

    filled_cnt_img = np.zeros(input_img.shape)
    cnt = filtered_contours[0]
    scaled_cnt, (cx, cy) = scale_contour(cnt, 0.8)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(draw_on, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.drawContours(draw_on, [cnt], 0, (255, 0, 0), 2)
    cv2.drawContours(filled_cnt_img, [scaled_cnt], 0, (255, 255, 255), thickness=-1)
    cv2.circle(draw_on, (x + w // 2, y + h // 2), 5, color=(0, 255, 0), thickness=2)
    mother_pixels = np.where(filled_cnt_img > 0)
    mother_pixels = list(mother_pixels)
    points_indices = np.row_stack(mother_pixels).T
    # point = np.array([input_img.shape[0] // 2, input_img.shape[1] // 2])
    if use_center:
        point = np.array([center[0], center[1]])
    else:
        point = np.array([cy, cx])
    points_dist = euclidean_dist_array(points_indices, point)
    if use_depth:
        print(mother_pixels)
        print("y_before", mother_pixels[0].shape)
        points_depth = depth_img[mother_pixels]
        print("depth_before", points_depth.shape)
        print("dist_before", points_depth.shape)
        non_zero_depth_indices = np.where(points_depth > 0)
        points_depth = points_depth[non_zero_depth_indices]
        points_dist = points_dist[non_zero_depth_indices]
        mother_pixels[0] = mother_pixels[0][non_zero_depth_indices]
        mother_pixels[1] = mother_pixels[1][non_zero_depth_indices]
        print("y_after", mother_pixels[0].shape)
        print("depth_after", points_depth.shape)
        print("dist_after", points_dist.shape)
        if method == 1:
            first_n_percent = int(0.5 * points_depth.shape[0])
            indices = np.argsort(np.abs(points_depth))[: max(first_n_percent, 1)]
            points_dist = points_dist[indices]
            points_depth = points_depth[indices]
            mother_pixels[0] = mother_pixels[0][indices]
            mother_pixels[1] = mother_pixels[1][indices]
            indices = np.argsort(points_dist)
            points_dist = points_dist[indices]
            points_depth = points_depth[indices]
            picking_point_idx = indices[0]
        elif method == 0:
            points_depth -= np.mean(points_depth)
            points_depth /= np.std(points_depth)
            points_dist -= np.mean(points_dist)
            points_dist /= np.std(points_dist)
            indices = np.argsort(points_dist - 0.3 * points_depth)
            indices = np.argsort(np.abs(points_dist))
            picking_point_idx = indices[len(indices) // 2]

        mother_pixels[0] = mother_pixels[0][indices]
        mother_pixels[1] = mother_pixels[1][indices]

        print(indices)
        xyz = dist_mat[:, mother_pixels[0], mother_pixels[1]].reshape(-1, 3)
        print("xyz shape = ", xyz.shape)
        print("num of indices = ", len(indices))
        for i in range(xyz.shape[0]):
            x, y, z = xyz[i, 0], xyz[i, 1], xyz[i, 2]
            x_cond = np.isclose(x - 0.036, xyz[:, 0], rtol=0, atol=5e-3)
            y_cond = np.isclose(y, xyz[:, 1], rtol=0, atol=5e-3)
            z_cond = np.isclose(z, xyz[:, 2], rtol=0, atol=2e-3)
            indices = np.where(np.logical_and(x_cond, np.logical_and(y_cond, z_cond)))
            if indices[0].shape[0] >= 1:
                print(indices)
                idx = indices[0][-1]
                print("picking_point = ", xyz[i, :])
                print("other_finger_picking_points = ", xyz[indices[0], :])
                picking_point_idx = i
                other_finger_picking_point_idx = idx
                other_finger_picking_point = (
                    mother_pixels[1][other_finger_picking_point_idx],
                    mother_pixels[0][other_finger_picking_point_idx],
                )
                if draw_on is not None:
                    cv2.circle(
                        draw_on,
                        other_finger_picking_point,
                        10,
                        color=(255, 0, 0),
                        thickness=-1,
                    )
                break
            else:
                print("No similar point for picking_point = ", xyz[i, :])
    else:
        picking_point_idx = np.argmin(points_dist)
    print(picking_point_idx)
    picking_point = (
        mother_pixels[1][picking_point_idx],
        mother_pixels[0][picking_point_idx],
    )
    if draw_on is not None:
        cv2.circle(draw_on, picking_point, 10, color=(0, 0, 255), thickness=-1)
        cv2.circle(draw_on, (point[1], point[0]), 10, color=(0, 255, 0), thickness=2)
    return picking_point


def detect_holes(input_img, draw_on=None):
    # Takes gray_scale img, returns rect values of detected laptop.

    min_len = 21
    max_len = 63
    min_circ = 0.4

    k = 2 * 16 - 1
    c = 2

    preprocessed_img = preprocess(input_img, gauss_kernel=k, clahe_kernel=c)

    all_contours, hierarchy = cv2.findContours(
        preprocessed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    filtered_contours = filter_contours(
        all_contours,
        sorting_key=perimeter,
        min_val=min_len,
        max_val=max_len,
        reverse=False,
    )
    filtered_contours = filter_contours(
        filtered_contours, sorting_key=circularity, min_val=min_circ, reverse=False
    )

    if len(filtered_contours) > 0:
        rect_params_list = []
        for contour in filtered_contours:
            rect_params_list.append(Rect(*cv2.boundingRect(contour)))
        if draw_on is not None:
            for rect_params in rect_params_list:
                rect_params.draw_on(draw_on)
        return rect_params_list
    else:
        return None


def rectangular_path_method(holes_coords, left, right, upper, lower, min_hole_dist):

    # Remove holes that are outside the cutting path rectangular area.
    holes_coords = list(
        filter(
            lambda coord: (
                ((left <= coord.x <= right) or (left <= coord.x2 <= right))
                and ((upper <= coord.y <= lower) or (upper <= coord.y2 <= lower))
            ),
            holes_coords,
        )
    )

    holes_intersecting_left_edge = list(
        filter(lambda coord: (coord.x <= left <= coord.x2), holes_coords)
    )
    # Sort the holes according to the coordinates of their right edge.
    x_sorted_holes_coords = sorted(
        holes_intersecting_left_edge, key=lambda coord: coord.x2
    )

    # Adjust the cutting rectangle left edge position,
    # to be at least far from nearest hole right edge by min_hole_dist.
    while len(holes_intersecting_left_edge) > 0:
        x2 = x_sorted_holes_coords[-1].x2
        left = x2 + min_hole_dist
        holes_intersecting_left_edge = list(
            filter(lambda coord: (coord.x <= left <= coord.x2), holes_coords)
        )
        # Sort the holes according to the coordinates of their right edge.
        x_sorted_holes_coords = sorted(
            holes_intersecting_left_edge, key=lambda coord: coord.x2
        )

    holes_intersecting_right_edge = list(
        filter(lambda coord: (coord.x <= right <= coord.x2), holes_coords)
    )
    # Sort the holes according to the coordinates of their left edge.
    x_sorted_holes_coords = sorted(
        holes_intersecting_right_edge, key=lambda coord: coord.x
    )

    # Adjust the cutting rectangle right edge position,
    # to be at least far from nearest hole left edge by min_hole_dist.
    while len(holes_intersecting_right_edge) > 0:
        x = x_sorted_holes_coords[0].x
        right = x - min_hole_dist
        holes_intersecting_right_edge = list(
            filter(lambda coord: (coord.x <= right <= coord.x2), holes_coords)
        )
        # Sort the holes according to the coordinates of their left edge.
        x_sorted_holes_coords = sorted(
            holes_intersecting_right_edge, key=lambda coord: coord.x
        )

    holes_intersecting_upper_edge = list(
        filter(lambda coord: (coord.y <= upper <= coord.y2), holes_coords)
    )
    # Sort the holes according to the coordinates of their lower edge.
    y_sorted_holes_coords = sorted(
        holes_intersecting_upper_edge, key=lambda coord: coord.y2
    )

    # Adjust the cutting rectangle upper edge position,
    # to be at least far from nearest hole lower edge by min_hole_dist.
    while len(holes_intersecting_upper_edge) > 0:
        y2 = y_sorted_holes_coords[-1].y2
        upper = y2 + min_hole_dist
        holes_intersecting_upper_edge = list(
            filter(lambda coord: (coord.y <= upper <= coord.y2), holes_coords)
        )
        # Sort the holes according to the coordinates of their lower edge.
        y_sorted_holes_coords = sorted(
            holes_intersecting_upper_edge, key=lambda coord: coord.y2
        )

    holes_intersecting_lower_edge = list(
        filter(lambda coord: (coord.y <= lower <= coord.y2), holes_coords)
    )
    # Sort the holes according to the coordinates of their upper edge.
    y_sorted_holes_coords = sorted(
        holes_intersecting_lower_edge, key=lambda coord: coord.y
    )

    # Adjust the cutting rectangle lower edge position,
    # to be at least far from nearest hole upper edge by min_hole_dist.
    while len(holes_intersecting_lower_edge) > 0:
        y = y_sorted_holes_coords[0].y
        lower = y - min_hole_dist
        holes_intersecting_lower_edge = list(
            filter(lambda coord: (coord.y <= lower <= coord.y2), holes_coords)
        )
        # Sort the holes according to the coordinates of their upper edge.
        y_sorted_holes_coords = sorted(
            holes_intersecting_lower_edge, key=lambda coord: coord.y
        )

    cut_rect = Rect(left, upper, right - left, lower - upper)

    holes_inside_cut_path = list(
        filter(
            lambda coord: (
                ((left <= coord.x <= right) or (left <= coord.x2 <= right))
                and ((upper <= coord.y <= lower) or (upper <= coord.y2 <= lower))
            ),
            holes_coords,
        )
    )

    return cut_rect, holes_inside_cut_path


def custom_path_method(
    holes_coords, left, right, upper, lower, min_hole_dist, edges_to_include=None
):
    """Produce a cutting path that preserves overall edge location but when a hole is near an edge,
    It avoids the hole by moving around it then returning to the edge location, and continue
    moving along it.

    param min_hole_dist: should be > hole diameter
    """
    # Initialize cutting path.
    cut_path = [
        [left, upper],
        [left, lower],
        [right, lower],
        [right, upper],
        [left, upper],
    ]

    edges = {"left": left, "lower": lower, "right": right, "upper": upper}

    # Assuming we start from upper left corner and start going down then right.
    # Then we reverse(i.e. move up then left)
    reverse = dict(zip(edges.keys(), (False, False, True, True)))
    horizontal = dict(zip(edges.keys(), (False, True, False, True)))

    # 'dir' here is used as a multiplier for when we reverse motion,
    # i.e. after moving down->right we move up->left
    dir = 1
    # These indicies indicate that for left for example,
    # when we move around the hole, we move in x direction (i.e. to the right),
    # that's why x changes from 0 to 1 while y doesn't change, then move in y_direction,
    # thus y changes from 0 to 1 while x remains constant and so on.
    x_indicies = (0, 1, 1, 0)
    y_indicies = (0, 0, 1, 1)
    prev_edges_points_num = 1

    def vertical_condition(coord):
        return (upper <= coord.y <= lower) or (upper <= coord.y2 <= lower)

    def horizontal_condition(coord):
        return (left <= coord.x <= right) or (left <= coord.x2 <= right)

    handeled_u_left_corner = False
    handeled_l_left_corner = False
    handeled_l_right_corner = False
    handeled_u_right_corner = False

    for edge_num, edge in enumerate(edges.keys()):
        if edges_to_include is not None:
            if edge not in edges_to_include:
                continue
        # Take only holes that are inside the edge length.
        condition = horizontal_condition if horizontal[edge] else vertical_condition
        filtered_holes_coords = list(filter(condition, holes_coords))

        # Find holes that has its edge near edge of contour by at most min_hole_dist.
        if horizontal[edge]:
            holes_near_edge = list(
                filter(
                    lambda hole: (
                        hole.y - min_hole_dist <= edges[edge] <= hole.y2 + min_hole_dist
                    ),
                    filtered_holes_coords,
                )
            )
        else:
            holes_near_edge = list(
                filter(
                    lambda hole: (
                        hole.x - min_hole_dist <= edges[edge] <= hole.x2 + min_hole_dist
                    ),
                    filtered_holes_coords,
                )
            )

        # Sort holes according to edge so that path moves down->right->up->left.
        # for left->lower->right->upper.
        def sorting_coord_fn(hole):
            return hole.x if horizontal[edge] else hole.y

        holes_near_edge.sort(key=sorting_coord_fn, reverse=reverse[edge])

        # Adjust cutting path to avoid these holes by adding path points around these holes.
        for hole in holes_near_edge:
            x, y, x2, y2 = hole.x, hole.y, hole.x2, hole.y2
            if reverse[edge]:
                x, x2, y, y2 = x2, x, y2, y
                dir = -1

            if horizontal[edge]:
                points_x = [x - dir * min_hole_dist, x2 + dir * min_hole_dist]
                points_y = [edges[edge], y - dir * min_hole_dist]
            else:
                points_x = [edges[edge], x2 + dir * min_hole_dist]
                points_y = [y - dir * min_hole_dist, y2 + dir * min_hole_dist]

            idx_0_flag = False
            idx_2_flag = False
            for idx, (x_idx, y_idx) in enumerate(zip(x_indicies, y_indicies)):
                p_idx = edge_num + prev_edges_points_num
                new_point_x, new_point_y = points_x[x_idx], points_y[y_idx]

                # Upper Left Corner Hole
                if (new_point_y < upper + 2 * min_hole_dist) and edge == "left":
                    if idx == 0:
                        continue
                    new_point_y = upper
                    cut_path[0] = (new_point_x, new_point_y)
                    cut_path[-1] = (new_point_x, new_point_y)
                    handeled_u_left_corner = True
                elif (
                    (new_point_x < left + 2 * min_hole_dist)
                    and edge == "upper"
                    and handeled_u_left_corner
                ):
                    break

                # Lower Left Corner Hole
                elif (new_point_y > lower - 2 * min_hole_dist) and edge == "left":
                    if idx == 3:
                        continue
                    new_point_y = lower
                    handeled_l_left_corner = True
                    cut_path.remove(cut_path[p_idx])
                    prev_edges_points_num -= 1
                elif (
                    (new_point_x < left + 2 * min_hole_dist)
                    and edge == "lower"
                    and handeled_l_left_corner
                ):
                    break

                # Lower Right Corner Hole
                elif (new_point_x > right - 2 * min_hole_dist) and edge == "lower":
                    if idx == 3:
                        continue
                    new_point_x = right
                    handeled_l_right_corner = True
                    cut_path.remove(cut_path[p_idx])
                    prev_edges_points_num -= 1
                elif (
                    (new_point_y >= lower - 2 * min_hole_dist)
                    and edge == "right"
                    and handeled_l_right_corner
                ):
                    break

                # Upper Right Corner Hole
                elif (new_point_y <= (upper + 2 * min_hole_dist)) and edge == "right":
                    if idx == 3:
                        continue
                    new_point_y = upper
                    handeled_u_right_corner = True
                    cut_path.remove(cut_path[p_idx])
                    prev_edges_points_num -= 1
                elif (
                    (new_point_x >= right - 2 * min_hole_dist)
                    and edge == "upper"
                    and handeled_u_right_corner
                ):
                    break

                # Remove edges that are near each other to save time and make it cleaner.
                # Also treat holes that are near each other as one big hole.
                elif (
                    (
                        (new_point_x - cut_path[p_idx - 1][0] < min_hole_dist)
                        and edge == "lower"
                    )
                    or (
                        (new_point_x - cut_path[p_idx - 1][0] > -min_hole_dist)
                        and edge == "upper"
                    )
                    or (
                        (new_point_y - cut_path[p_idx - 1][1] < min_hole_dist)
                        and edge == "left"
                    )
                    or (
                        (new_point_y - cut_path[p_idx - 1][1] > -min_hole_dist)
                        and edge == "right"
                    )
                    or (new_point_y <= upper and edge == "right")
                    or (new_point_y >= lower and edge == "left")
                    or (
                        ((new_point_x <= left) or (new_point_x >= right))
                        and edge == "upper"
                    )
                ):
                    if idx == 0:
                        idx_0_flag = True
                        cut_path.remove(cut_path[p_idx - 1])
                        prev_edges_points_num -= 1
                        continue
                    if idx == 1 and idx_0_flag:
                        if idx == 1:
                            idx_0_flag = False
                        if horizontal[edge]:
                            if (
                                cut_path[p_idx - 1][1] <= new_point_y
                                and edge == "lower"
                                or cut_path[p_idx - 1][1] >= new_point_y
                                and edge == "upper"
                            ):
                                new_point_x = cut_path[p_idx - 1][0]
                            else:
                                cut_path[p_idx - 1][0] = new_point_x
                        else:
                            if (
                                cut_path[p_idx - 1][0] >= new_point_x and edge == "left"
                            ) or (
                                cut_path[p_idx - 1][0] <= new_point_x
                                and edge == "right"
                            ):
                                new_point_y = cut_path[p_idx - 1][1]
                            else:
                                cut_path[p_idx - 1][1] = new_point_y
                    # Handle holes at end of an edge but not a corner hole.
                    if idx == 2:
                        new_point_x = cut_path[p_idx][0]
                        cut_path.remove(cut_path[p_idx])
                        prev_edges_points_num -= 1
                        idx_2_flag = True
                    if idx == 3 and idx_2_flag:
                        idx_2_flag = False
                        continue

                cut_path.insert(p_idx, [new_point_x, new_point_y])
                prev_edges_points_num += 1
        x_indicies, y_indicies = y_indicies, x_indicies

    return cut_path


def interpolate(p1, p2, step):
    if p2 == p1:
        return [p1]
    sign = int(abs(p2 - p1) / (p2 - p1))
    # print("step = ", step, "sign  = ", sign)
    return list(range(p1, p2, max(step, 1) * sign))


def interpolate_path(path, npoints=20):
    new_path = []
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        if abs(y2 - y1) > 0:
            step = round(abs(y2 - y1) / npoints)
            new_ys = interpolate(y1, y2, step)
            new_xs = [x1] * len(new_ys)
        else:
            step = round(abs(x2 - x1) / npoints)
            new_xs = interpolate(x1, x2, step)
            new_ys = [y1] * len(new_xs)
        new_path.extend([(x, y) for x, y in zip(new_xs, new_ys)])
        new_path.append((x2, y2))
    return new_path


def plan_cover_cutting_path(
    input_img=None,
    tol=30,
    min_hole_dist=5,
    draw_on=None,
    laptop_coords=None,
    holes_coords=None,
    method=0,
    interpolate=True,
    npoints=20,
    edges_to_include=None,
):
    """Takes gray_scale img containing a laptop Or takes laptop & holes coordinates,
     returns cutting path points as list of tuples.

    param tol: Defines The Coordinates of The initial Bounding Box To Be Cut
     by taking a tolerance from laptop bounding box.

    param min_hole_dist: A minimum distance criteria between a hole and any edge
     in the cutting rectangle.

    param laptop_coords: Detected laptop box values,
     should be a tuble of 4 values x, y, width and height.

    param holes_coords: Detected holes box values,
     should be a list of tubles of 4 values x, y, width and height.

    """

    if input_img is not None:
        # Copy original image into 'gray'
        gray = input_img.copy()

        # Detect Laptop Bounding Box Coordinates
        laptop_coords = detect_laptop(gray)

        # Crop The Image to The Laptop Bounding Box
        cropped_gray = laptop_coords.crop_img(gray)

        # Detect The holes on The Cropped Image
        holes_coords = detect_holes(cropped_gray)
    elif laptop_coords is not None:
        laptop_coords = Rect(*laptop_coords)
        if holes_coords is not None:
            holes_coords = [Rect(*hole) for hole in deepcopy(holes_coords)]
    else:
        raise ValueError("Either provide an input image or the laptop coordinates")

    # Assign The Coordinates of The initial Bounding Box To Be Cut by taking a tolerance from laptop bounding box.
    cut_rect = Rect(
        laptop_coords.x + tol,
        laptop_coords.y + tol,
        laptop_coords.w - 2 * tol,
        laptop_coords.h - 2 * tol,
    )

    # Assign the cutting rectangle initial edge coordinates
    left = cut_rect.x
    right = left + cut_rect.w
    upper = cut_rect.y
    lower = upper + cut_rect.h

    if draw_on is not None:
        # Draw the laptop bounding box in blue
        laptop_coords.draw_on(draw_on, "b")
        # Draw the initial cutting path in green.
        # cut_rect.draw_on(draw_on)

    if holes_coords is not None:

        # Draw the hole(s) bounding boxes, and make their coordinates absolute
        for hole in holes_coords:
            if input_img is not None:
                hole.add(laptop_coords)
            if draw_on is not None:
                hole.draw_on(draw_on)

        holes_inside_cut_path = []
        if method == 0:
            cut_rect, holes_inside_cut_path_as_rects = rectangular_path_method(
                holes_coords, left, right, upper, lower, min_hole_dist
            )
            for hole_rect in holes_inside_cut_path_as_rects:
                holes_inside_cut_path.append(hole_rect.rect_to_list())
            cut_path = cut_rect.rect_to_path()
        elif method == 1:
            cut_path = custom_path_method(
                holes_coords,
                left,
                right,
                upper,
                lower,
                min_hole_dist,
                edges_to_include=edges_to_include,
            )
            for hole in holes_coords:
                if box_near_by_dist(laptop_coords.rect_to_list(), [hole.rect_to_list()]):
                    holes_inside_cut_path.append(hole.rect_to_list())
            
        elif method is None:
            cut_path = cut_rect.rect_to_path()
        else:
            raise ValueError("Wrong method number")

        if interpolate:
            cut_path = interpolate_path(cut_path, npoints=npoints)

        if draw_on is not None:
            # Draw the final cutting path in red.
            # cut_rect.draw_on(draw_on, 'r')
            laptop_coords.draw_on(draw_on, 'r')
            pass
        
        # cropped_img = cut_rect.crop_img(original_img)
        # cropped_gray = cut_rect.crop_img(gray)
        # detect_holes(cropped_gray, draw_on=cropped_img)

    return cut_path, holes_inside_cut_path


def group_boxes(boxes, grouping_dist, condition):
    box_groups = []
    for bnum, box in enumerate(boxes):
        if bnum == 0:
            box_groups.append([box])
        if bnum == len(boxes) - 1:
            break
        nbox = boxes[bnum + 1]
        cx1, cy1, cx2, cy2 = box.x, box.y, box.x2, box.y2
        nx1, ny1, nx2, ny2 = nbox.x, nbox.y, nbox.x2, nbox.y2
        vars_dict = {
            "cx1": cx1,
            "cy1": cy1,
            "cx2": cx2,
            "cy2": cy2,
            "nx1": nx1,
            "ny1": ny1,
            "nx2": nx2,
            "ny2": ny2,
        }
        if (vars_dict[condition[0]] - vars_dict[condition[1]]) < grouping_dist:
            box_groups[-1].append(nbox)
        else:
            box_groups.append([nbox])
    return box_groups


def plan_port_cutting_path(
    motherboard_coords, ports_coords, near_edge_dist, grouping_dist, cutting_dist
):
    left, upper, w, h = (
        motherboard_coords[0],
        motherboard_coords[1],
        motherboard_coords[2],
        motherboard_coords[3],
    )
    right, lower = left + w, upper + h
    if ports_coords is not None:
        ports_coords = [Rect(*port) for port in deepcopy(ports_coords)]
    ned = near_edge_dist

    # Get only ports that are in or near the motherboard area.
    ports_coords = list(
        filter(
            lambda coord: (
                (left - ned <= coord.x <= right + ned)
                or (left - ned <= coord.x2 <= right + ned)
            )
            and (
                (upper - ned <= coord.y <= lower + ned)
                or (upper - ned <= coord.y2 <= lower + ned)
            ),
            ports_coords,
        )
    )

    # Get ports that are near each edge.
    ports_near_left_edge = list(
        filter(lambda coord: (coord.x - ned <= left <= coord.x2 + ned), ports_coords)
    )
    ports_near_lower_edge = list(
        filter(lambda coord: (coord.y - ned <= lower <= coord.y2 + ned), ports_coords)
    )
    ports_near_right_edge = list(
        filter(lambda coord: (coord.x - ned <= right <= coord.x2 + ned), ports_coords)
    )
    ports_near_upper_edge = list(
        filter(lambda coord: (coord.y - ned <= upper <= coord.y2 + ned), ports_coords)
    )

    # Sort ports to be in the direction of motion of each edge.
    ports_near_left_edge = sorted(
        ports_near_left_edge, key=lambda coord: coord.y, reverse=False
    )
    ports_near_lower_edge = sorted(
        ports_near_lower_edge, key=lambda coord: coord.x, reverse=False
    )
    ports_near_right_edge = sorted(
        ports_near_right_edge, key=lambda coord: coord.y2, reverse=True
    )
    ports_near_upper_edge = sorted(
        ports_near_upper_edge, key=lambda coord: coord.x2, reverse=True
    )

    # Group ports that are near each other together to be cut all at once.
    port_groups_near_left_edge = group_boxes(
        ports_near_left_edge, grouping_dist, ["ny1", "cy2"]
    )
    port_groups_near_lower_edge = group_boxes(
        ports_near_lower_edge, grouping_dist, ["nx1", "cx2"]
    )
    port_groups_near_right_edge = group_boxes(
        ports_near_right_edge, grouping_dist, ["cy1", "ny2"]
    )
    port_groups_near_upper_edge = group_boxes(
        ports_near_upper_edge, grouping_dist, ["cx1", "nx2"]
    )

    # Remove dublicated ports that are handeled on two edges, so that they are handeled only once.
    # Also keep it in the group with the bigger size to save time in cutting operation.
    port_groups_near_vertical_edges = []
    port_groups_near_vertical_edges.extend(port_groups_near_right_edge)
    port_groups_near_vertical_edges.extend(port_groups_near_left_edge)
    vertical_ports_to_remove = []
    ports_to_remove_near_lower_edge = []
    ports_to_remove_near_upper_edge = []
    for vg_idx, vertical_port_group in enumerate(port_groups_near_vertical_edges):
        for v_idx, vertical_port in enumerate(vertical_port_group):
            for pg_idx, port_group in enumerate(port_groups_near_lower_edge):
                for p_idx, port in enumerate(port_group):
                    if port == vertical_port:
                        if len(port_group) >= len(vertical_port_group):
                            vertical_ports_to_remove.append((vg_idx, vertical_port))
                        else:
                            ports_to_remove_near_lower_edge.append((pg_idx, port))
            for pg_idx, port_group in enumerate(port_groups_near_upper_edge):
                for p_idx, port in enumerate(port_group):
                    if port == vertical_port:
                        if len(port_group) >= len(vertical_port_group):
                            vertical_ports_to_remove.append((vg_idx, vertical_port))
                        else:
                            ports_to_remove_near_upper_edge.append((pg_idx, port))
    [
        port_groups_near_lower_edge[g_idx].remove(p_idx)
        for g_idx, p_idx in ports_to_remove_near_lower_edge
    ]
    [
        port_groups_near_upper_edge[g_idx].remove(p_idx)
        for g_idx, p_idx in ports_to_remove_near_upper_edge
    ]
    [
        port_groups_near_vertical_edges[g_idx].remove(p_idx)
        for g_idx, p_idx in vertical_ports_to_remove
    ]

    # Construct ports cutting path for each group of ports for each edge
    cut_paths = []
    for group in port_groups_near_left_edge:
        if len(group) < 1:
            continue
        x = min(group, key=lambda port: port.x).x
        y = group[0].y - cutting_dist
        x2 = max(group, key=lambda port: port.x2).x2 + cutting_dist
        y2 = group[-1].y2 + cutting_dist
        cut_paths.append([(x, y), (x2, y), (x2, y2), (x, y2)])

    for group in port_groups_near_lower_edge:
        if len(group) < 1:
            continue
        x = group[0].x - cutting_dist
        y = min(group, key=lambda port: port.y).y - cutting_dist
        x2 = group[-1].x2 + cutting_dist
        y2 = max(group, key=lambda port: port.y2).y2
        cut_paths.append([(x, y2), (x, y), (x2, y), (x2, y2)])

    for group in port_groups_near_right_edge:
        if len(group) < 1:
            continue
        x = min(group, key=lambda port: port.x).x - cutting_dist
        y = group[-1].y - cutting_dist
        x2 = max(group, key=lambda port: port.x2).x2
        y2 = group[0].y2 + cutting_dist
        cut_paths.append([(x2, y2), (x, y2), (x, y), (x2, y)])

    for group in port_groups_near_upper_edge:
        if len(group) < 1:
            continue
        x = group[-1].x - cutting_dist
        y = min(group, key=lambda port: port.y).y
        x2 = group[0].x2 + cutting_dist
        y2 = max(group, key=lambda port: port.y2).y2 + cutting_dist
        cut_paths.append([(x2, y), (x2, y2), (x, y2), (x, y)])

    return cut_paths


def constrain_environment(dist_mat, x_lim, y_lim, z_lim):
    # convert depth to a uint8 image with pixel values (0 -> 255)
    dist_image = deepcopy(dist_mat[2]) * 255 / np.max(dist_mat[2])
    dist_image = dist_image.astype(np.uint8)
    _, dist_image = cv2.threshold(dist_image, 0, 255, cv2.THRESH_BINARY)

    x, y, z = dist_mat[0], dist_mat[1], dist_mat[2]

    x_thresh = cv2.inRange(x, *x_lim)
    _, x_thresh = cv2.threshold(x_thresh, 0, 1, cv2.THRESH_BINARY)

    y_thresh = cv2.inRange(y, *y_lim)
    _, y_thresh = cv2.threshold(y_thresh, 0, 1, cv2.THRESH_BINARY)

    z_thresh = cv2.inRange(z, *z_lim)
    _, z_thresh = cv2.threshold(z_thresh, 0, 1, cv2.THRESH_BINARY)

    # color_img *= (x_thresh * y_thresh * z_thresh).reshape((x.shape[0], x.shape[1], 1))
    dist_image *= x_thresh * y_thresh * z_thresh
    return dist_image


def find_nearest_point_with_non_zero_depth(
    dist_mat, point, min_depth=0.26, max_depth=0.4
):
    center = deepcopy(point)
    print("prev_center = ", center)
    print("dist_mat_shape = ", dist_mat.shape)
    point = dist_mat[:, center[1], center[0]]
    print("prev_3d_pos = ", point)
    if min_depth < point[2] < max_depth:
        return center
    non_zero_indices = np.where(
        min_depth
        <= np.logical_and(
            dist_mat[2, :, :] <= max_depth, dist_mat[2, :, :] >= min_depth
        )
    )
    print("min_z = ", np.min(dist_mat[2, non_zero_indices[0], non_zero_indices[1]]))
    print(
        "depth_values_in_range = ",
        dist_mat[2, non_zero_indices[0], non_zero_indices[1]],
    )
    non_zero_arr = np.row_stack(non_zero_indices).T
    # print("non_zero_arr = ", non_zero_arr)
    points_dist = euclidean_dist_array(non_zero_arr, (center[1], center[0]))
    # print("points_dist = ", points_dist)
    center_idx = np.argsort(points_dist)
    center = [non_zero_indices[1][center_idx[0]], non_zero_indices[0][center_idx[0]]]
    print("new_center = ", center)
    print("new_3d_pos = ", dist_mat[:, center[1], center[0]])
    return center

def find_nearest_box_points_with_non_zero_depth(boxes, dist_mat, in_format=('x1', 'y1', 'x2', 'y2')):
    new_boxes = deepcopy(boxes)
    new_boxes = [convert_format(b, in_format=in_format) for b in new_boxes]
    for b in new_boxes:
        for i in range(2):
            new_point = find_nearest_point_with_non_zero_depth(
                dist_mat, b[i * 2 : i * 2 + 2]
            )
            new_boxes[-1][i * 2], new_boxes[-1][i * 2 + 1] = new_point[0], new_point[1]
    return new_boxes

def detect_laptop_pose(
    dist_mat, x_min=-2, x_max=2, y_min=-2, y_max=2, z_min=0, z_max=4, draw=False
):
    dist_image = constrain_environment(
        deepcopy(dist_mat),
        x_lim=(x_min, x_max),
        y_lim=(y_min, y_max),
        z_lim=(z_min, z_max),
    )

    color_img = cv2.cvtColor(dist_image, cv2.COLOR_GRAY2BGR)
    draw_on = color_img if draw else None
    laptop_data_px = detect_laptop(dist_image, draw_on=draw_on)
    center = laptop_data_px[0:2]
    center = find_nearest_point_with_non_zero_depth(dist_mat, center)
    laptop_data_px[0], laptop_data_px[1] = center[0], center[1]
    if draw:
        cv2.circle(draw_on, tuple(center), 5, (255, 255, 0), thickness=-1)
    return laptop_data_px, color_img, dist_image


def filter_xyz_list(xyz_list):
    xyz_arr = np.array(xyz_list).reshape((-1, 3))
    print("xyz_arr_shape_before_filter = ", xyz_arr.shape)
    print("xyz_arr = ", xyz_arr)
    median_z = np.median(xyz_arr[:, 2])
    indices_to_remove = []
    for i in range(xyz_arr.shape[0]):
        if (abs(xyz_arr[i, 2] - median_z) >= 0.05) or (xyz_arr[i, 2] <= 0.26):
            indices_to_remove.append(i)
    xyz_arr = np.delete(xyz_arr, indices_to_remove, axis=0)
    print("xyz_arr_shape_after_filter", xyz_arr.shape)
    xyz_list = xyz_arr.reshape((-1,)).tolist()
    return xyz_list


def xyz_list_to_pose_array(xyz_list):
    pose_array = PoseArray()
    for i in range(0, len(xyz_list), 3):  # x, y, z for each point
        pose = Pose()
        pose.position.x = xyz_list[i]
        pose.position.y = xyz_list[i + 1]
        pose.position.z = xyz_list[i + 2]
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 1
        pose_array.poses.append(pose)
    return pose_array


class RealsenseHelpers:
    def __init__(self, raw_depth=True):
        self.color_intrin_topic = "/camera/color/camera_info"
        self.depth_intrin_topic = "/camera/depth/camera_info"
        self.depth_to_color_extrin_topic = "/camera/extrinsics/depth_to_color"
        self.raw_depth_topic = "/camera/depth/image_rect_raw"
        self.aligned_depth_topic = "/camera/aligned_depth_to_color/image_raw"
        if raw_depth:
            self.depth_topic = self.raw_depth_topic
            self.intrinsics_topic = self.depth_intrin_topic
        else:
            self.depth_topic = self.aligned_depth_topic
            self.intrinsics_topic = self.color_intrin_topic
        self.extrinsics_topic = "/camera/extrinsics/depth_to_color"

    def boxes_px_to_xyz(
        self,
        boxes,
        dist_mat,
        preprocessor=box_to_center,
        filter_data=False,
        return_as="1d",
    ):
        boxes_xyz_list = []
        for box in boxes:
            processed_box = preprocessor(box)
            xyz_list = self.px_to_xyz(
                px_data=processed_box, dist_mat=dist_mat, filter_data=filter_data
            )
            if return_as == "1d":
                boxes_xyz_list.extend(xyz_list)
            elif return_as == "2d":
                boxes_xyz_list.append(xyz_list)
        return boxes_xyz_list

    def px_to_xyz(
        self,
        px_data,
        depth_img=None,
        intrinsics=None,
        dist_mat=None,
        px_data_format="list",
        return_as="1d",
        reversed_pixels=False,
        filter_data=False,
        transform_mat=None
    ):

        if len(px_data) < 1:
            return []

        if dist_mat is None:
            if depth_img is None:
                depth_img_msg = rospy.wait_for_message(self.aligned_depth_topic, Image)
                depth_img = ros_numpy.numpify(depth_img_msg)
            if intrinsics is None:
                intrinsics = rospy.wait_for_message(self.color_intrin_topic, CameraInfo)
            dist_mat = self.calculate_dist_3D(depth_img, intrinsics)

        px_data_arr = np.array(px_data).T.reshape((2, -1), order="F")
        (x_idx, y_idx) = (0, 1) if not reversed_pixels else (1, 0)
        xyz_arr = dist_mat[:, px_data_arr[y_idx], px_data_arr[x_idx]]
        if transform_mat is not None:
            xyz_arr = self.apply_cam_to_cam_transform(xyz_arr, transform_mat=transform_mat)
        if return_as == "1d":
            # 'F' means to read/write data from rows then when finished go to next column,
            # Check this https://docs.oracle.com/cd/E19957-01/805-4940/z400091044d0/index.html
            xyz_arr = np.reshape(xyz_arr, (-1,), order="F")
        data_xyz = xyz_arr.tolist()
        return data_xyz

    def xyz_to_pose_array(self, xyz_list):
        return xyz_list_to_pose_array(xyz_list)

    def get_depth_to_color_extrinsics(self):
        extrinsics = rospy.wait_for_message(
            self.depth_to_color_extrin_topic, Extrinsics
        )
        return extrinsics

    def get_depth_to_color_transform(self):
        extrinsics = rospy.wait_for_message(
            self.depth_to_color_extrin_topic, Extrinsics
        )
        return self.get_transform_from_extrinsics(extrinsics)

    def get_transform_from_extrinsics(self, extrinsics):
        r = np.array(extrinsics.rotation)
        t = np.array(extrinsics.translation)
        transform_mat = np.zeros((3, 4))
        transform_mat[:, -1] = t
        transform_mat[:, :-1] = r.reshape((3, 3), order="F")
        return transform_mat

    def get_inverse_transform(self, transform_mat_3x4):
        transform_mat = np.zeros((4, 4))
        transform_mat[:-1, :] = transform_mat_3x4
        transform_mat[-1, -1] = 1
        inverse_transform_mat = np.linalg.inv(transform_mat)
        return inverse_transform_mat[:3, :]

    def get_color_to_depth_extrinsics(
        self, depth_to_color_extrinsics=None, depth_to_color_transform=None
    ):
        if depth_to_color_transform is None:
            if depth_to_color_extrinsics is None:
                depth_to_color_extrinsics = self.get_depth_to_color_extrinsics()
            depth_to_color_transform = self.get_transform_from_extrinsics(
                depth_to_color_extrinsics
            )
        inverse_transform_mat = self.get_inverse_transform(depth_to_color_transform)
        # print("Original_transform_mat = ", transform_mat)
        # print("Inverse_transform_mat = ", inverse_transform_mat)
        color_to_depth_extrinsics = Extrinsics()
        color_to_depth_extrinsics.rotation = (
            inverse_transform_mat[:3, :3].reshape((9,), order="F").tolist()
        )
        color_to_depth_extrinsics.translation = (
            inverse_transform_mat[:3, -1].reshape((3,)).tolist()
        )
        return color_to_depth_extrinsics

    def get_color_to_depth_transform(
        self,
        color_to_depth_extrinsics=None,
        depth_to_color_extrinsics=None,
        depth_to_color_transform=None,
    ):
        if color_to_depth_extrinsics is None:
            if depth_to_color_transform is None:
                if depth_to_color_extrinsics is None:
                    depth_to_color_extrinsics = self.get_depth_to_color_extrinsics()
                depth_to_color_transform = self.get_transform_from_extrinsics(
                    depth_to_color_extrinsics
                )
            color_to_depth_transform = self.get_inverse_transform(
                depth_to_color_transform
            )
        else:
            color_to_depth_transform = self.get_transform_from_extrinsics(
                color_to_depth_extrinsics
            )
        return color_to_depth_transform

    def apply_cam_to_cam_transform(
        self, dist_mat, transform_mat=None, extrinsics=None, invert_transform=False
    ):
        if transform_mat is None:
            transform_mat = self.get_transform_from_extrinsics(extrinsics)
        if invert_transform:
            transform_mat = self.get_inverse_transform(transform_mat)
        modified_dist_mat = np.ones((4, np.prod(dist_mat.shape[1:], axis=0)))
        modified_dist_mat[:-1, :] = dist_mat.reshape((3, -1))
        return np.dot(transform_mat, modified_dist_mat).reshape(dist_mat.shape)

    def get_intrinsics(self, intrinsics_topic):
        return rospy.wait_for_message(intrinsics_topic, CameraInfo)

    def get_intrinsics_as_dict_from_intrinsics_camera_info(self, intrinsics):
        intr = {
            "fx": 0,
            "fy": 0,
            "px": 0,
            "py": 0,
            "w": 0,
            "h": 0,
            "distortion_model": "",
        }
        intr["fx"] = intrinsics.K[0]
        intr["fy"] = intrinsics.K[4]
        intr["px"] = intrinsics.K[2]
        intr["py"] = intrinsics.K[5]
        intr["w"] = intrinsics.width
        intr["h"] = intrinsics.height
        intr["distortion_model"] = intrinsics.distortion_model
        return intr

    def calculate_dist_3D(self, depth_img, intrinsics, index_mat=None):
        intr = self.get_intrinsics_as_dict_from_intrinsics_camera_info(intrinsics)
        depth_img = depth_img * 0.001
        if index_mat is None:
            index_mat = np.indices(depth_img.shape[:2])
            # print(index_mat.shape)
        dist_mat = np.zeros((3, *depth_img.shape))
        dist_mat[0] = (index_mat[1] - intr["px"]) * depth_img / intr["fx"]
        dist_mat[1] = (index_mat[0] - intr["py"]) * depth_img / intr["fy"]
        dist_mat[2] = depth_img
        return dist_mat

    def calculate_pixels_from_points(self, dist_mat, intrinsics):
        intr = self.get_intrinsics_as_dict_from_intrinsics_camera_info(intrinsics)
        pixel_mat = np.zeros((2, dist_mat.shape[1]))
        pixel_mat[0] = ((dist_mat[0] / dist_mat[2]) * intr["fx"]) + intr["px"]
        pixel_mat[1] = ((dist_mat[1] / dist_mat[2]) * intr["fy"]) + intr["py"]
        wrong_indices = np.argwhere(pixel_mat[0] < 0)
        pixel_mat[0, wrong_indices] = 0
        wrong_indices = np.argwhere(pixel_mat[1] < 0)
        pixel_mat[1, wrong_indices] = 0
        print("px", intr["px"])
        print("py", intr["py"])
        print("fx", intr["fx"])
        print("fy", intr["fy"])
        return pixel_mat.astype(np.uint16)

    def get_depth_img_from_cam(self, depth_topic="/camera/depth/image_rect_raw"):
        # Recieve depth image and intrinsics
        depth_img_msg = rospy.wait_for_message(depth_topic, Image)
        depth_img = ros_numpy.numpify(depth_img_msg)
        return depth_img

    def get_dist_mat_from_cam(
        self,
        depth_topic="/camera/depth/image_rect_raw",
        intrinsics_topic="/camera/depth/camera_info",
        extrinsics_topic="/camera/extrinsics/depth_to_color",
        transform_to_color=False,
    ):
        # Recieve depth image and intrinsics
        depth_img = self.get_depth_img_from_cam(depth_topic)
        intrinsics = self.get_intrinsics(intrinsics_topic)
        # Calculate the 3d data of each pixel in the depth image and pu it in dist_mat.
        dist_mat = self.calculate_dist_3D(depth_img, intrinsics)
        if transform_to_color:
            # Get the extrinsics data and transform data in dist_mat from depth camera frame to color frame.
            extrinsics = rospy.wait_for_message(extrinsics_topic, Extrinsics)
            dist_mat = self.apply_cam_to_cam_transform(dist_mat, extrinsics=extrinsics)
        return dist_mat

    def adjust_pixels_to_boundary(self, pixels, size):
        px = deepcopy(pixels)
        for i in [0, 1]:
            px[i] = np.where(px[i] < size[i], px[i], size[i] - 1)
            px[i] = np.where(px[i] >= 0, px[i], 0)
        return px

    def cam_pixels_to_other_cam_pixels_unaligned(
        self, from_px, dist_mat, to_cam_intrin, cam_to_cam_extrin, filter_xyz=True
    ):

        # Give me a 2d/1d list of pixels
        points = self.px_to_xyz(from_px, dist_mat=dist_mat, return_as="2d")
        if filter_xyz:
            points = filter_xyz_list(np.array(points).T)
        points = np.array(points).reshape((3, -1), order="F")  # (3, npoints)
        wrong_points = np.where(np.logical_or(points[2] <= 0.26, points[2] >= 0.4))
        # print("wrong_points = ", wrong_points)
        if cam_to_cam_extrin is not None:
            points = self.apply_cam_to_cam_transform(
                points, extrinsics=cam_to_cam_extrin
            )
        if filter_xyz:
            points = filter_xyz_list(points.T.tolist())
        points = np.array(points).reshape((3, -1), order="F")  # (3, npoints)
        wrong_points = np.where(np.logical_or(points[2] <= 0.26, points[2] >= 0.4))
        # print("points_y = ", points[1])
        # print("points_z = ", points[2])
        # print("wrong_points = ", wrong_points)
        pixels = self.calculate_pixels_from_points(points, to_cam_intrin)
        # print("pixels_before_round = ", pixels)
        pixels = np.round(pixels).astype(np.uint16)
        # print("pixels_before_adjustment = ", pixels)
        pixels = self.adjust_pixels_to_boundary(
            pixels, (dist_mat.shape[2], dist_mat.shape[1])
        )

        return pixels

    def color_pixels_to_depth_pixels_and_back(
        self,
        color_px,
        dist_mat_aligned,
        dist_mat,
        depth_intrin,
        color_intrin,
        color_to_depth_extrin,
        depth_to_color_extrin,
    ):

        # Give me a 2d/1d list of pixels
        depth_pixels = self.cam_pixels_to_other_cam_pixels_unaligned(
            color_px,
            dist_mat_aligned,
            depth_intrin,
            color_to_depth_extrin,
            depth_range=False,
        )
        depth_pixels = depth_pixels.T
        color_pixels = self.cam_pixels_to_other_cam_pixels_unaligned(
            depth_pixels, dist_mat, color_intrin, depth_to_color_extrin
        )
        return color_pixels.reshape((2, -1))


def adjust_hole_center(image, hole_boxes):
    output = image.copy()
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # k = 2 * 16 - 1
    # c = 2
    # gray = preprocess(gray, gauss_kernel=k, clahe_kernel=c)
    new_hole_centers = []
    for i, hole in enumerate(hole_boxes):
        rect_hole = Rect(*hole)
        # rect_hole.enlarge_by(5)
        # gray_crop = rect_hole.crop_img(gray)
        # # show the output image
        # cv2.imshow("gray_crop", gray_crop)
        # cv2.waitKey(0)
        gray_crop = output
        # detect circles in the image
        circles = cv2.HoughCircles(gray_crop, cv2.HOUGH_GRADIENT, 1.01, 1, 200, 100)
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                x += rect_hole.x
                y += rect_hole.y
                prev_x = rect_hole.x + (rect_hole.w // 2)
                prev_y = rect_hole.y + (rect_hole.h // 2)
                cv2.circle(output, (prev_x, prev_y), 1, (0, 0, 255), 1)
                cv2.circle(output, (x, y), 1, (0, 255, 0), 1)
                # cv2.rectangle(output, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)
                new_hole_centers.append([x, y])
            # show the output image
            cv2.imshow("output", output)
            cv2.waitKey(0)
    return new_hole_centers


# receives RGB image and a list of [x,y,w,h] lists
def correct_circles(img, circles_list):
    cv2.namedWindow("cropped_image")
    cv2.namedWindow("output")
    param1 = 40
    param2 = 38
    kernel = 0
    k = 3
    min_len = 0
    max_len = 31
    min_circ = 0
    sz = 3
    dist = 3

    cv2.createTrackbar("param1", "cropped_image", param1, 500, lambda x: None)
    cv2.createTrackbar("param2", "cropped_image", param2, 500, lambda x: None)
    cv2.createTrackbar("kernel", "cropped_image", kernel, 255, lambda x: None)
    cv2.createTrackbar("k", "cropped_image", k, 20, lambda x: None)
    cv2.createTrackbar("min_len", "cropped_image", min_len, 255, lambda x: None)
    cv2.createTrackbar("max_len", "cropped_image", max_len, 255, lambda x: None)
    cv2.createTrackbar("min_circ", "cropped_image", min_circ, 255, lambda x: None)
    cv2.createTrackbar("sz", "cropped_image", sz, 50, lambda x: None)
    cv2.createTrackbar("dist", "cropped_image", dist, 50, lambda x: None)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    new_circles_list = []
    circle_new = None
    key = 0
    use_canny = True
    for circle_old in circles_list:
        #############################
        # Pre-Processing            #
        #############################
        while key != ord("e"):
            cropped_img = deepcopy(
                gray[
                    circle_old[1] : circle_old[1] + circle_old[3] + 1,
                    circle_old[0] : circle_old[0] + circle_old[2] + 1,
                ]
            )
            cropped_color_img = deepcopy(
                img[
                    circle_old[1] : circle_old[1] + circle_old[3] + 1,
                    circle_old[0] : circle_old[0] + circle_old[2] + 1,
                ]
            )
            param1 = max(cv2.getTrackbarPos("param1", "cropped_image"), 1)
            param2 = max(cv2.getTrackbarPos("param2", "cropped_image"), 1)
            kernel_sz = max(cv2.getTrackbarPos("kernel", "cropped_image"), 1)
            k = max(2 * cv2.getTrackbarPos("k", "cropped_image") - 1, 1)
            min_len = cv2.getTrackbarPos("min_len", "cropped_image")
            max_len = cv2.getTrackbarPos("max_len", "cropped_image")
            min_circ = cv2.getTrackbarPos("min_circ", "cropped_image") / 255.0
            sz = cv2.getTrackbarPos("sz", "cropped_image")
            dist = max(cv2.getTrackbarPos("dist", "cropped_image"), 1)
            output = cv2.GaussianBlur(cropped_img, (k, k), 0)

            # ret, output = cv2.threshold(output, 80, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            output = cv2.adaptiveThreshold(
                output,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )

            if use_canny:
                output = cv2.Canny(output, param1, param2)

            # kernel = np.ones((2, 2), np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_sz, kernel_sz))
            # Use erosion and dilation combination to eliminate false positives.
            # output = cv2.erode(output, kernel, iterations=1)
            output = cv2.dilate(output, kernel, iterations=1)
            # output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel, iterations=1)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

            # ========================================================= #
            cv2.imshow("output", output)
            key = cv2.waitKey(10) & 0xFF
            #############################
            # Find and Filter contours  #
            #############################

            contours, hierarchy = cv2.findContours(
                output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )

            # #
            # # contours = sorted(contours, key=cv2.contourArea, reverse=False)[min_area:top_area]
            contours = sorted(contours, key=circularity)
            contours_circ = list(map(circularity, contours))
            start = bisect_left(contours_circ, min_circ)
            contours = contours[start:]
            contours.reverse()

            contours = sorted(contours, key=cv2.contourArea)
            contours_per = list(map(cv2.contourArea, contours))
            start = bisect_left(contours_per, min_len)
            end = bisect_right(contours_per, max_len)
            contours = contours[start : end + 1]
            contours = sorted(
                contours, key=lambda c: (perimeter(c) / (max_len + 1)) * circularity(c)
            )
            contours.reverse()

            old_center = [circle_old[2] // 2, circle_old[3] // 2]
            # contours = sorted(contours, key=aspect_ratio)[:top_len]
            sz = min(len(contours), sz)
            if len(contours) > 0:
                contours_poly = [None] * sz
                boundRect = [None] * sz
                centers = [None] * sz
                radius = [None] * sz
                for i in range(sz):
                    contours_poly[i] = cv2.approxPolyDP(contours[i], 3, True)
                    boundRect[i] = cv2.boundingRect(contours_poly[i])
                    centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
                filtered_centers = list(
                    filter(
                        lambda c: euclidean_dist(c[1], old_center) <= dist,
                        enumerate(centers),
                    )
                )
                best_center = old_center
                if len(filtered_centers) > 0:
                    best_center = filtered_centers[0][1]
                    i = filtered_centers[0][0]
                color = (0, 0, 255)
                cv2.circle(
                    cropped_color_img,
                    (int(best_center[0]), int(best_center[1])),
                    1,
                    color,
                    1,
                )
                cv2.circle(
                    cropped_color_img,
                    (int(old_center[0]), int(old_center[1])),
                    1,
                    (255, 0, 0),
                    1,
                )
                cv2.drawContours(cropped_color_img, contours[0], -1, (0, 255, 0), 1)
            cv2.imshow("cropped_image", cropped_color_img)
            key = cv2.waitKey(10) & 0xFF
            # center_difference = [i[0] - old_center[0],
            #                     i[1] - old_center[1]]

            # new_circles_list.append([center_difference[0] + circle_old[0],     #New X
            #                         center_difference[1] + circle_old[1],     #New Y
            #                         circle_old[2],                        #Same W
            #                         circle_old[3]])
        # param1 = 50
        # param2 = 20
        # dp = 40

        # cv2.createTrackbar('param1', 'cropped_image',
        #                    param1, 500, lambda x: None)
        # cv2.createTrackbar('param2', 'cropped_image',
        #                param2, 500, lambda x: None)
        # cv2.createTrackbar('thresh', 'cropped_image',
        # 0, 255, lambda x: None)
        # cv2.createTrackbar('dp', 'cropped_image',
        #                    dp, 500, lambda x: None)
        # cv2.createTrackbar('minR', 'cropped_image',
        #                    0, 255, lambda x: None)
        # cv2.createTrackbar('maxR', 'cropped_image',
        #                    0, 255, lambda x: None)
        # cv2.createTrackbar('minDist', 'cropped_image',
        #                    0, 255, lambda x: None)
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # new_circles_list = []
        # circle_new = None
        # key = 0
        # for circle_old in circles_list:
        #     while True and key != ord('e'):
        #         cropped_img = deepcopy(gray[circle_old[1] : circle_old[1]+circle_old[3]+1 ,
        #                         circle_old[0] : circle_old[0]+circle_old[2]+1])
        #         cropped_color_img = deepcopy(img[circle_old[1]: circle_old[1]+circle_old[3]+1,
        #                                     circle_old[0]: circle_old[0]+circle_old[2]+1])
        #         param1 = max(cv2.getTrackbarPos('param1', 'cropped_image'), 1)
        #         param2 = max(cv2.getTrackbarPos('param2', 'cropped_image'),1)
        #         thresh = cv2.getTrackbarPos('thresh', 'cropped_image')
        #         dp = cv2.getTrackbarPos('dp', "cropped_image") / 255.0 + 1
        #         minR = cv2.getTrackbarPos('minR', 'cropped_image')
        #         maxR = cv2.getTrackbarPos('maxR', 'cropped_image')
        #         minDist = max(cv2.getTrackbarPos('minDist', 'cropped_image'), 1)
        #         k = thresh*2 + 1
        #         print(param1, param2, thresh, dp, minR, maxR, minDist, k)
        #         cropped_img = cv2.GaussianBlur(cropped_img, (k,k), 0)
        #         # cv2.imshow("cropped_image", cropped_img)laptop_coords
        #         # cv2.waitKey(0)
        #         # apply Otsu's automatic thresholding
        #         # (T, cropped_img) = cv2.threshold(cropped_img, thresh, 255, cv2.THRESH_BINARY)

        #         cv2.imshow("cropped_image", cropped_img)
        #         key = cv2.waitKey(10) & 0xFF
        #         circle_new = cv2.HoughCircles(
        #             cropped_img, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minR, maxRadius=maxR)
        #         if circle_new is not None:
        #             circle_new = np.uint16(np.around(circle_new))
        #             for i in circle_new[0, :]:

        #                 old_center = [circle_old[2]//2 , circle_old[3]//2]
        #                 center_difference = [i[0] - old_center[0],
        #                                     i[1] - old_center[1]]

        #                 new_circles_list.append([center_difference[0] + circle_old[0],     #New X
        #                                         center_difference[1] + circle_old[1],     #New Y
        #                                         circle_old[2],                        #Same W
        #                                         circle_old[3]])
        #                                     # draw the outer circle
        #                 cv2.circle(cropped_color_img,(i[0],i[1]),i[2],(0,255,0),1)
        #                 # draw the center of the circle
        #                 cv2.circle(cropped_color_img,(i[0],i[1]),1,(0,0,255),1)
        #                 cv2.circle(cropped_color_img,
        #                            (old_center[0], old_center[1]), 1, (255, 0, 0), 1)
        #                 cv2.imshow("cropped_image", cropped_color_img)
        #                 key = cv2.waitKey(10) & 0xFF

        #             print("Circle Found")
        #         else:
        #             new_circles_list.append(circle_old)
        #             print("No Circle")

        return new_circles_list


if __name__ == "__main__":

    data_dir = "/home/ubuntu/data/laptop_base/"
    dset_sz = 34
    image_id = 1
    img = read_and_resize(data_dir, image_id)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("image_window")
    # cv2.namedWindow("output_window")

    while True:
        laptop_coords = detect_laptop(gray, draw_on=img)

        cv2.imshow("image_window", img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("e"):
            break
        elif key == ord("c"):
            image_id += 1
            if image_id < dset_sz:
                img = read_and_resize(data_dir, image_id)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                break

    # close all open windows
    cv2.destroyAllWindows()
