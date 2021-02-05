import cv2
import numpy as np
from bisect import bisect_right, bisect_left
from copy import deepcopy


class Rect:

    color_dict = {'b':(255, 0, 0), 'g':(0, 255, 0), 'r':(0, 0, 255)}

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.w = width
        self.h = height
        self.x2 = self.x + self.w
        self.y2 = self.y + self.h
    
    def crop_img(self, input_img):
        return input_img[self.y: self.y2, self.x: self.x2].copy()
    
    def draw_on(self, input_img, color='g', thickness=2):
        if color not in self.color_dict.keys():
            raise ValueError("Available Colors are 'b', 'g', and 'r' for blue, green, and red respectively")
        cv2.rectangle(input_img, (self.x, self.y), (self.x2, self.y2), self.color_dict[color], thickness)
    
    def shift_by(self, x, y):
        self.x += x
        self.x2 += x
        self.y += y
        self.y2 += y
    
    def add(self, rect):
        self.shift_by(rect.x, rect.y)

    def rect_to_path(self):
        return [(self.x, self.y), (self.x, self.y2), (self.x2, self.y), (self.x2, self.y2), (self.x, self.y)]


def nothing(x):
    pass


def enclosing_rect_area(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return w*h


def circularity(x):
    return (4 * cv2.contourArea(x) * np.pi) / (cv2.arcLength(x, True) ** 2 + 1e-7)


def circularity_using_moments(cnt):
    M = cv2.moments(cnt)
    return (M['m00'] ** 2) / (2 * np.pi * (M['m20'] + M['m02']) + 1e-7)


def perimeter(x):
    return cv2.arcLength(x, True)


def aspect_ratio(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return abs((float(w) / h) - 1)


def read_and_resize(directory, img_id, size=(720, 480), compression='.jpg'):
    read_img = cv2.imread(directory + str(img_id) + compression)
    resized_img = cv2.resize(read_img, size)
    return resized_img


def preprocess(input_img, gauss_kernel=21, clahe_kernel=2,
               morph_kernel=3, iterations=3, dilate=False, use_canny=False, thresh1=42, thresh2=111):
    # Contrast Norm + Gauss Blur + Adaptive Threshold + Dilation + Canny
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(clahe_kernel, clahe_kernel))
    output_img = clahe.apply(input_img)

    output_img = cv2.GaussianBlur(output_img, (gauss_kernel, gauss_kernel), 0)

    output_img = cv2.adaptiveThreshold(output_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    if use_canny:
        output_img = cv2.Canny(output_img, thresh1, thresh2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    if dilate:
        output_img = cv2.dilate(output_img, kernel, iterations=iterations)
    output_img = cv2.morphologyEx(output_img, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return output_img


def filter_contours(input_contours, sorting_key, min_val=None, max_val=None, reverse=False):
    sorted_contours = sorted(input_contours, key=sorting_key, reverse=reverse)
    contours_feature = list(map(sorting_key, sorted_contours))
    start = 0
    if min_val is not None:
        start = bisect_left(contours_feature, min_val)
    if max_val is not None:
        end = bisect_right(contours_feature, max_val)
        filtered_contours = sorted_contours[start:end+1]
    else:
        filtered_contours = sorted_contours[start:]
    return filtered_contours


def detect_laptop(input_img, draw_on=None):
    # Takes gray_scale img, returns rect values of detected laptop.

    min_len = 138000
    max_len = 200000
    k = 15
    c = 2

    preprocessed_img = preprocess(input_img, gauss_kernel=k, clahe_kernel=c, morph_kernel=5, iterations=1, dilate=True)

    all_contours, hierarchy = cv2.findContours(preprocessed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    filtered_contours = filter_contours(all_contours, sorting_key=enclosing_rect_area,
                                        min_val=min_len, max_val=max_len, reverse=False)
    if len(filtered_contours) > 0:
        rect_params = Rect(*cv2.boundingRect(filtered_contours[0]))
        if draw_on is not None:
            rect_params.draw_on(draw_on)
        return rect_params
    else:
        return None


def detect_holes(input_img, draw_on=None):
    # Takes gray_scale img, returns rect values of detected laptop.

    min_len = 21
    max_len = 63
    min_circ = 0.68

    k = 2 * 16 - 1
    c = 2

    preprocessed_img = preprocess(input_img, gauss_kernel=k, clahe_kernel=c)

    all_contours, hierarchy = cv2.findContours(preprocessed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    filtered_contours = filter_contours(all_contours, sorting_key=perimeter,
                                        min_val=min_len, max_val=max_len, reverse=False)
    filtered_contours = filter_contours(filtered_contours, sorting_key=circularity,
                                        min_val=min_circ, reverse=False)

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


def rectangular_path_method(holes_coords, left_edge, right_edge, upper_edge, lower_edge, min_hole_dist):

    # Sort the holes according to the coordinates of their right edge.
    x_sorted_holes_coords = sorted(holes_coords, key=lambda coord: coord.x2)

    # Remove holes that are outside the cutting path rectangular area.
    x_sorted_holes_coords = list(filter(lambda coord: upper_edge <= coord.y <= lower_edge,
                                        x_sorted_holes_coords))

    # Get right edge coordinates of holes.
    sorted_x_values = list(map(lambda coord: coord.x2, x_sorted_holes_coords))

    # Adjust the cutting rectangle left edge position,
    # to be at least far from nearest hole right edge by min_hole_dist.
    while len(sorted_x_values) > 0:
        left_edge_nearest_hole_idx = min(bisect_left(
            sorted_x_values, left_edge), len(sorted_x_values)-1)
        hole = x_sorted_holes_coords[left_edge_nearest_hole_idx]
        x = hole.x
        y = hole.y
        x2 = hole.x2
        y2 = hole.y2
        if abs(x - left_edge) < min_hole_dist and upper_edge <= y <= lower_edge:
            left_edge = x2 + min_hole_dist
        else:
            break

    # Get left edge coordinates of holes.
    sorted_x_values = list(
        map(lambda coord: coord.x, x_sorted_holes_coords))

    # Adjust the cutting rectangle right edge position,
    # to be at least far from nearest hole left edge by min_hole_dist.
    while len(sorted_x_values) > 0:
        right_edge_nearest_hole_idx = max(
            0, bisect_right(sorted_x_values, right_edge) - 1)
        x = x_sorted_holes_coords[right_edge_nearest_hole_idx].x
        y = x_sorted_holes_coords[right_edge_nearest_hole_idx].y
        if abs(x - right_edge) < min_hole_dist and upper_edge <= y <= lower_edge:
            right_edge = x - min_hole_dist
        else:
            break

    # Sort the holes according to the coordinates of their lower edge.
    y_sorted_holes_coords = sorted(
        holes_coords, key=lambda coord: coord.y2)

    # Remove holes that are outside the cutting path rectangular area.
    y_sorted_holes_coords = list(filter(lambda coord: left_edge <= coord.x <= right_edge,
                                        y_sorted_holes_coords))
    # Get lower edge coordinates of holes.
    sorted_y_values = list(
        map(lambda coord: coord.y2, y_sorted_holes_coords))

    # Adjust the cutting rectangle upper edge position,
    # to be at least far from nearest hole lower edge by min_hole_dist.
    while len(sorted_y_values) > 0:
        upper_edge_nearest_hole_idx = min(bisect_left(
            sorted_y_values, upper_edge), len(sorted_y_values)-1)
        x = y_sorted_holes_coords[upper_edge_nearest_hole_idx].x
        y = y_sorted_holes_coords[upper_edge_nearest_hole_idx].y
        y2 = y_sorted_holes_coords[upper_edge_nearest_hole_idx].y2
        if abs(y - upper_edge) < min_hole_dist and left_edge <= x <= right_edge:
            upper_edge = y2 + min_hole_dist
        else:
            break

    # Get upper edge coordinates of holes.
    sorted_y_values = list(
        map(lambda coord: coord.y, y_sorted_holes_coords))

    # Adjust the cutting rectangle lower edge position,
    # to be at least far from nearest hole upper edge by min_hole_dist.
    while len(sorted_y_values) > 0:
        lower_edge_nearest_hole_idx = bisect_right(
            sorted_y_values, lower_edge) - 1
        x = y_sorted_holes_coords[lower_edge_nearest_hole_idx].x
        y = y_sorted_holes_coords[lower_edge_nearest_hole_idx].y
        if abs(y - lower_edge) < min_hole_dist and left_edge <= x <= right_edge:
            lower_edge = y - min_hole_dist
        else:
            break

    cut_rect = Rect(left_edge, upper_edge, right_edge - \
                    left_edge, lower_edge - upper_edge)
    
    return cut_rect

# IMPROVEMENT:: Make it able to work properly when hole is at one of the contour corners,
# Or make sure that the given contour doesn't have a hole in one of it's corners.
def custom_path_method(holes_coords, left_edge, right_edge, upper_edge, lower_edge, min_hole_dist):
    """Produce a cutting path that preserves overall edge location but when a hole is near an edge,
    It avoids the gole by moving around it then returning to the edge location, and continue
    moving along it.
        
    param min_hole_dist: should be > hole diameter
    """
    # Initialize cutting path.
    cut_path = [(left_edge, upper_edge), (left_edge, lower_edge), 
    (right_edge, lower_edge), (right_edge, upper_edge), (left_edge, upper_edge)]
    
    # Assuming we start from upper left corner and start going down then right.
    # Then we reverse(i.e. move up then left)
    reverse = (False, False, True, True)
    horizontal = (False, True, False, True)

    # This is the hole edge that we compare with each contour edge.
    # Thus if it is the contour left edge we compare with hole right edge and so on.
    # i.e. we compare with hole edge that is further from the contour edge inside contour area.
    ref_coord_fns = (lambda h: h.x2, lambda h: h.y, lambda h: h.x, lambda h: h.y2)
    edges = (left_edge, lower_edge, right_edge, upper_edge)

    # 'dir' here is used as a multiplier for when we reverse motion,
    # i.e. after moving down->right we move up->left
    dir = 1
    # These indicies indicate that for left_edge for example,
    # when we move around the hole, we move in x direction (i.e. to the right),
    # that's why x changes from 0 to 1 while y doesn't change, then move in y_direction,
    # thus y changes from 0 to 1 while x remains constant and so on. 
    x_indicies = (0, 1, 1, 0)
    y_indicies = (0, 0, 1, 1)
    prev_edges_points_num = 1
    # Remove holes that are outside the cutting path rectangular area.

    def vertical_condition(coord): return (upper_edge <= coord.y <= lower_edge)
    def horizontal_condition(coord): return (left_edge <= coord.x <= right_edge)

    for edge in range(len(edges)):
        condition = horizontal_condition if horizontal[edge] else vertical_condition
        filtered_holes_coords = list(filter(condition, holes_coords))

        # Find holes that has its edge near edge of contour by at most min_hole_dist.
        holes_near_edge = list(filter(lambda hole: abs(
            ref_coord_fns[edge](hole) - edges[edge]) < min_hole_dist, filtered_holes_coords))

        # Sort holes according to edge so that path moves down->right->up->left.
        # for left_edge->lower_edge->right_edge->upper_edge.
        def sorting_coord_fn(hole): return hole.x if horizontal[edge] else hole.y
        holes_near_edge.sort(key=sorting_coord_fn, reverse=reverse[edge])

        # Adjust cutting path to avoid these holes by adding path points around these holes.
        for hole in holes_near_edge:
            x, y, x2, y2 = hole.x, hole.y, hole.x2, hole.y2
            if reverse[edge]:
                x, x2, y, y2 = x2, x, y2, y
                dir = -1

            if horizontal[edge]:
                points_x = (x - dir * min_hole_dist, x2 + dir * min_hole_dist)
                points_y = (edges[edge], y - dir * min_hole_dist)
            else:
                points_x = (edges[edge], x2 + dir * min_hole_dist)
                points_y = (y - dir * min_hole_dist, y2 + dir * min_hole_dist)

            for x_idx, y_idx in zip(x_indicies, y_indicies):
                cut_path.insert(edge + prev_edges_points_num,(points_x[x_idx], points_y[y_idx]))
                prev_edges_points_num += 1
        x_indicies, y_indicies = y_indicies, x_indicies
    
    return cut_path


def plan_cover_cutting_path(input_img=None, tol=30, min_hole_dist=10, draw_on=None,
 laptop_coords=None, holes_coords=None, method=0):
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
        raise ValueError(
            "Either provide an input image or the laptop coordinates")

    
    # Assign The Coordinates of The initial Bounding Box To Be Cut by taking a tolerance from laptop bounding box.
    cut_rect = Rect(laptop_coords.x + tol, laptop_coords.y + tol, laptop_coords.w - 2*tol, laptop_coords.h - 2*tol)    

    # Assign the cutting rectangle initial edge coordinates
    left_edge = cut_rect.x
    right_edge = left_edge + cut_rect.w
    upper_edge = cut_rect.y
    lower_edge = upper_edge + cut_rect.h

    if draw_on is not None:
        # Draw the laptop bounding box in blue
        laptop_coords.draw_on(draw_on, 'b')
        # Draw the initial cutting path in green.
        # cut_rect.draw_on(draw_on)

    if holes_coords is not None:

        # Draw the hole(s) bounding boxes, and make their coordinates absolute
        for hole in holes_coords:
            hole.add(laptop_coords)
            if draw_on is not None:
                hole.draw_on(draw_on)

        if method == 0:
            cut_rect = rectangular_path_method(holes_coords, left_edge, right_edge, upper_edge, lower_edge, min_hole_dist)
            cut_path = cut_rect.rect_to_path()
        elif method == 1:
            cut_path = custom_path_method(holes_coords, left_edge, right_edge, upper_edge, lower_edge, min_hole_dist)
        else:
            raise ValueError("Wrong method number")

        if draw_on is not None:
            # Draw the final cutting path in red.
            # cut_rect.draw_on(draw_on, 'r')
            pass

        # cropped_img = cut_rect.crop_img(original_img)
        # cropped_gray = cut_rect.crop_img(gray)
        # detect_holes(cropped_gray, draw_on=cropped_img)
        
    return cut_path


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

        if key == ord('e'):
            break
        elif key == ord('c'):
            image_id += 1
            if image_id < dset_sz:
                img = read_and_resize(data_dir, image_id)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                break

    # close all open windows
    cv2.destroyAllWindows()
