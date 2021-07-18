import cv2
import numpy as np
from bisect import bisect_right, bisect_left
from copy import deepcopy
from perception.coco_datasets import convert_format
from math import fabs, sqrt, sin, cos, pi
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


def read_and_resize(directory, img_id, size=(720, 480), compression=".jpg"):
    read_img = cv2.imread(directory + str(img_id) + compression)
    if size is not None:
        read_img = cv2.resize(read_img, size)
    return read_img


def draw_lines(image_np, points_list):
    for i in range(len(points_list) - 1):
        cv2.line(
            image_np, tuple(points_list[i]), tuple(points_list[i + 1]), (0, 0, 255), 2
        )


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2, draw_center=False):
    for box in boxes:
        cv2.rectangle(
            image, tuple(box[0:2]), (box[0] + box[2], box[1] + box[3]), color, thickness
        )
        if draw_center:
            cv2.circle(
                image, (box[0] + box[2] // 2, box[1] + box[3] // 2), 1, color, thickness
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
    near = False
    for box in boxes:
        near = point_near_box_by_dist(center_point, box, dist_as_side_ratio)
        if near:
            break
    return near


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


def detect_laptop(input_img, draw_on=None, **kwargs):
    # Takes gray_scale img, returns rect values of detected laptop.

    kwargs = {
        "min_val": 200000,
        "max_val": 500000,
        "gauss_kernel": 1,
        "median_sz": 23,
        "clahe_kernel": 2,
        "morph_kernel": 59,
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

        # Retrieve the key parameters of the rotated bounding box
        center = (int(rect[0][0]), int(rect[0][1]))
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(rect[2])

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

        flip_point = (int(center[0] + nw), int(center[1] + nh))
        upper_point = (int(center[0] + nw), int(center[1] - nh))
        laptop_data = [
            center[0],
            center[1],
            flip_point[0],
            flip_point[1],
            upper_point[0],
            upper_point[1],
        ]
        if draw_on is not None:
            cv2.drawContours(draw_on, [box], 0, (0, 255, 0), thickness=5)
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
            x_cond = np.isclose(x - 0.036, xyz[:, 0], rtol=5e-3, atol=5e-3)
            y_cond = np.isclose(y, xyz[:, 1], rtol=5e-3, atol=5e-3)
            z_cond = np.isclose(z, xyz[:, 2], rtol=2e-3, atol=2e-3)
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
                    cv2.circle(draw_on, other_finger_picking_point, 10, color=(255, 0, 0), thickness=-1)
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
    print("step = ", step, "sign  = ", sign)
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
    interp_step=2,
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
                holes_inside_cut_path.append(
                    [hole_rect.x, hole_rect.y, hole_rect.w, hole_rect.h]
                )
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
        elif method is None:
            cut_path = cut_rect.rect_to_path()
        else:
            raise ValueError("Wrong method number")

        if interpolate:
            cut_path = interpolate_path(cut_path, step=interp_step)

        if draw_on is not None:
            # Draw the final cutting path in red.
            # cut_rect.draw_on(draw_on, 'r')
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
    return laptop_data_px, color_img


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
        if raw_depth:
            self.depth_topic = "/camera/depth/image_rect_raw"
            self.intrinsics_topic = "/camera/depth/camera_info"
        else:
            self.depth_topic = "/camera/aligned_depth_to_color/image_raw"
            self.intrinsics_topic = "/camera/color/camera_info"
        self.extrinsics_topic = "/camera/extrinsics/depth_to_color"

    def px_to_xyz(
        self,
        px_data,
        depth_img=None,
        intrinsics=None,
        dist_mat=None,
        px_data_format="list",
    ):
        if dist_mat is None:
            dist_mat = self.calculate_dist_3D(depth_img, intrinsics)
        data_xyz = []
        i = 0
        while True:
            if px_data_format == "list_of_tuples":
                x_px, y_px = px_data[i]
                i += 1
            else:
                x_px = px_data[i]
                y_px = px_data[i + 1]
                i += 2

            xyz = dist_mat[:, y_px, x_px].tolist()
            data_xyz.extend(xyz)
            if i >= (len(px_data)):
                break
        return data_xyz

    def transform_depth_to_color_frame(self, dist_mat, extrinsics):
        r = np.array(extrinsics.rotation)
        t = np.array(extrinsics.translation)
        transform_mat = np.zeros((3, 4))
        transform_mat[:, -1] = t
        transform_mat[:, :-1] = r.reshape((3, 3))
        modified_dist_mat = np.ones((4, dist_mat.shape[1] * dist_mat.shape[2]))
        modified_dist_mat[:-1, :] = dist_mat.reshape((3, -1))
        return np.dot(transform_mat, modified_dist_mat).reshape(dist_mat.shape)

    def get_intrinsics_as_dict_from_intrinsics_camera_info(self, intrinsics):
        intr = {"fx": 0, "fy": 0, "px": 0, "py": 0, "w": 0, "h": 0}
        intr["fx"] = intrinsics.K[0]
        intr["fy"] = intrinsics.K[4]
        intr["px"] = intrinsics.K[2]
        intr["py"] = intrinsics.K[5]
        intr["w"] = intrinsics.width
        intr["h"] = intrinsics.height
        return intr

    def calculate_dist_3D(self, depth_img, intrinsics):
        intr = self.get_intrinsics_as_dict_from_intrinsics_camera_info(intrinsics)
        depth_img = depth_img * 0.001
        index_mat = np.indices(depth_img.shape)
        print(index_mat.shape)
        dist_mat = np.zeros((3, intr["h"], intr["w"]))
        dist_mat[0] = (index_mat[1] - intr["px"]) * depth_img / intr["fx"]
        dist_mat[1] = (index_mat[0] - intr["py"]) * depth_img / intr["fy"]
        dist_mat[2] = depth_img
        return dist_mat

    def get_dist_mat_from_cam(
        self,
        depth_topic="/camera/depth/image_rect_raw",
        intrinsics_topic="/camera/depth/camera_info",
        extrinsics_topic="/camera/extrinsics/depth_to_color",
        transform_to_color=False,
    ):
        # Recieve depth image and intrinsics
        depth_img_msg = rospy.wait_for_message(depth_topic, Image)
        depth_img = ros_numpy.numpify(depth_img_msg)
        intrinsics = rospy.wait_for_message(intrinsics_topic, CameraInfo)
        # Calculate the 3d data of each pixel in the depth image and pu it in dist_mat.
        dist_mat = self.calculate_dist_3D(depth_img, intrinsics)
        if transform_to_color:
            # Get the extrinsics data and transform data in dist_mat from depth camera frame to color frame.
            extrinsics = rospy.wait_for_message(extrinsics_topic, Extrinsics)
            dist_mat = self.transform_depth_to_color_frame(dist_mat, extrinsics)
        return dist_mat


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
        #         # cv2.imshow("cropped_image", cropped_img)
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
