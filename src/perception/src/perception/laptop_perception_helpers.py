import cv2
import numpy as np
from bisect import bisect_right, bisect_left

from numpy.lib.function_base import select


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
        cv2.rectangle(input_img, (self.x, self.y), (self.x2, self.y2), self.color_dict[color], thickness)
    
    def shift_by(self, x, y):
        self.x += x
        self.x2 += x
        self.y += y
        self.y2 += y
    
    def add(self, rect):
        self.shift_by(rect.x, rect.y)


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


def read_and_resize(directory, img_id, size=(720, 480)):
    read_img = cv2.imread(directory + str(img_id) + '.jpg')
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
    

def plan_cover_cutting_path(input_img, tol=30, min_hole_dist=10, draw_on=None):
    """Takes gray_scale img containing a laptop, returns rect values of cutting path.

    param tol: Defines The Coordinates of The initial Bounding Box To Be Cut
     by taking a tolerance from laptop bounding box.

    param min_hole_dist: A minimum distance criteria between a hole and any edge
     in the cutting rectangle.

    """

    # Copy original image into 'gray'
    gray = input_img.copy()

    # Detect Laptop Bounding Box Coordinates
    laptop_coords = detect_laptop(gray)

    # Crop The Image to The Laptop Bounding Box
    cropped_gray = laptop_coords.crop_img(gray)                                                                                                        

    # Detect The holes on The Cropped Image
    holes_coords = detect_holes(cropped_gray)

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
        cut_rect.draw_on(draw_on)

    if holes_coords is not None:

        # Draw the hole(s) bounding boxes, and make their coordinates absolute
        for hole in holes_coords:
            hole.add(laptop_coords)
            if draw_on is not None:
                hole.draw_on(draw_on)

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
            left_edge_nearest_hole_idx = min(bisect_left(sorted_x_values, left_edge), len(sorted_x_values)-1)
            x = x_sorted_holes_coords[left_edge_nearest_hole_idx].x
            y = x_sorted_holes_coords[left_edge_nearest_hole_idx].y
            x2 = x_sorted_holes_coords[left_edge_nearest_hole_idx].x2
            if abs(x - left_edge) < min_hole_dist and upper_edge <= y <= lower_edge:
                left_edge = x2 + min_hole_dist
            else:
                break

        # Get left edge coordinates of holes.
        sorted_x_values = list(map(lambda coord: coord.x, x_sorted_holes_coords))

        # Adjust the cutting rectangle right edge position,
        # to be at least far from nearest hole left edge by min_hole_dist.
        while len(sorted_x_values) > 0:
            right_edge_nearest_hole_idx = max(0, bisect_right(sorted_x_values, right_edge) - 1)
            x = x_sorted_holes_coords[right_edge_nearest_hole_idx].x
            y = x_sorted_holes_coords[right_edge_nearest_hole_idx].y
            if abs(x - right_edge) < min_hole_dist and upper_edge <= y <= lower_edge:
                right_edge = x - min_hole_dist
            else:
                break

        # Sort the holes according to the coordinates of their lower edge.
        y_sorted_holes_coords = sorted(holes_coords, key=lambda coord: coord.y2)

        # Remove holes that are outside the cutting path rectangular area.
        y_sorted_holes_coords = list(filter(lambda coord: left_edge <= coord.x <= right_edge,
                                            y_sorted_holes_coords))
        # Get lower edge coordinates of holes.
        sorted_y_values = list(map(lambda coord: coord.y2, y_sorted_holes_coords))

        # Adjust the cutting rectangle upper edge position,
        # to be at least far from nearest hole lower edge by min_hole_dist.
        while len(sorted_y_values) > 0:
            upper_edge_nearest_hole_idx = min(bisect_left(sorted_y_values, upper_edge), len(sorted_y_values)-1)
            x = y_sorted_holes_coords[upper_edge_nearest_hole_idx].x
            y = y_sorted_holes_coords[upper_edge_nearest_hole_idx].y
            y2 = y_sorted_holes_coords[upper_edge_nearest_hole_idx].y2
            if abs(y - upper_edge) < min_hole_dist and left_edge <= x <= right_edge:
                upper_edge = y2 + min_hole_dist
            else:
                break

        # Get upper edge coordinates of holes.
        sorted_y_values = list(map(lambda coord: coord.y, y_sorted_holes_coords))

        # Adjust the cutting rectangle lower edge position,
        # to be at least far from nearest hole upper edge by min_hole_dist.
        while len(sorted_y_values) > 0:
            lower_edge_nearest_hole_idx = bisect_right(sorted_y_values, lower_edge) - 1
            x = y_sorted_holes_coords[lower_edge_nearest_hole_idx].x
            y = y_sorted_holes_coords[lower_edge_nearest_hole_idx].y
            if abs(y - lower_edge) < min_hole_dist and left_edge <= x <= right_edge:
                lower_edge = y - min_hole_dist
            else:
                break
        
        cut_rect = Rect(left_edge, upper_edge, right_edge - left_edge, lower_edge - upper_edge)

        if draw_on is not None:
            # Draw the final cutting path in red.
            cut_rect.draw_on(draw_on, 'r')

        # cropped_img = cut_rect.crop_img(original_img)
        # cropped_gray = cut_rect.crop_img(gray)
        # detect_holes(cropped_gray, draw_on=cropped_img)
        
    return cut_rect


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
