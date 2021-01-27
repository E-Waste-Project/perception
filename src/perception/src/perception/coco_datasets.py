# Source: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#create-custom-coco-dataset
import base64
import IPython
import json
import numpy as np
import os
import random
import requests
from io import BytesIO
from math import trunc
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
import collections

# Load the dataset json


class CocoDataset():
    def __init__(self, annotation_path, image_dir):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.colors = ['blue', 'purple', 'red', 'green', 'orange', 'salmon', 'pink', 'gold',
                       'orchid', 'slateblue', 'limegreen', 'seagreen', 'darkgreen', 'olive',
                       'teal', 'aquamarine', 'steelblue', 'powderblue', 'dodgerblue', 'navy',
                       'magenta', 'sienna', 'maroon']

        json_file = open(self.annotation_path)
        self.coco = json.load(json_file)
        json_file.close()

        #self.process_info()
        #self.process_licenses()
        self.process_categories()
        self.process_images()
        self.process_segmentations()

    def display_info(self):
        print('Dataset Info:')
        print('=============')
        for key, item in self.info.items():
            print('  {}: {}'.format(key, item))

        requirements = [['description', str],
                        ['url', str],
                        ['version', str],
                        ['year', int],
                        ['contributor', str],
                        ['date_created', str]]
        for req, req_type in requirements:
            if req not in self.info:
                print('ERROR: {} is missing'.format(req))
            elif type(self.info[req]) != req_type:
                print('ERROR: {} should be type {}'.format(req, str(req_type)))
        print('')

    def display_licenses(self):
        print('Licenses:')
        print('=========')

        requirements = [['id', int],
                        ['url', str],
                        ['name', str]]
        for license in self.licenses:
            for key, item in license.items():
                print('  {}: {}'.format(key, item))
            for req, req_type in requirements:
                if req not in license:
                    print('ERROR: {} is missing'.format(req))
                elif type(license[req]) != req_type:
                    print('ERROR: {} should be type {}'.format(
                        req, str(req_type)))
            print('')
        print('')

    def display_categories(self):
        print('Categories:')
        print('=========')
        for sc_key, sc_val in self.super_categories.items():
            print('  super_category: {}'.format(sc_key))
            for cat_id in sc_val:
                print('    id {}: {}'.format(
                    cat_id, self.categories[cat_id]['name']))
            print('')

    def display_image(self, image_id, show_polys=True, show_bbox=True, show_labels=True, show_crowds=True, use_url=False):
        print('Image:')
        print('======')
        if image_id == 'random':
            image_id = random.choice(list(self.images.keys()))

        # Print the image info
        image = self.images[image_id]
        for key, val in image.items():
            print('  {}: {}'.format(key, val))

        # Open the image
        if use_url:
            image_path = image['coco_url']
            response = requests.get(image_path)
            image = PILImage.open(BytesIO(response.content))

        else:
            image_path = os.path.join(self.image_dir, image['file_name'])
            image = PILImage.open(image_path)

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = "data:image/png;base64, " + \
            base64.b64encode(buffered.getvalue()).decode()

        # Calculate the size and adjusted display size
        max_width = 900
        image_width, image_height = image.size
        adjusted_width = min(image_width, max_width)
        adjusted_ratio = adjusted_width / image_width
        adjusted_height = adjusted_ratio * image_height

        # Create list of polygons to be drawn
        polygons = {}
        bbox_polygons = {}
        rle_regions = {}
        poly_colors = {}
        labels = {}
        print('  segmentations ({}):'.format(
            len(self.segmentations[image_id])))
        for i, segm in enumerate(self.segmentations[image_id]):
            polygons_list = []
            if segm['iscrowd'] != 0:
                # Gotta decode the RLE
                px = 0
                x, y = 0, 0
                rle_list = []
                for j, counts in enumerate(segm['segmentation']['counts']):
                    if j % 2 == 0:
                        # Empty pixels
                        px += counts
                    else:
                        # Need to draw on these pixels, since we are drawing in vector form,
                        # we need to draw horizontal lines on the image
                        x_start = trunc(
                            trunc(px / image_height) * adjusted_ratio)
                        y_start = trunc(px % image_height * adjusted_ratio)
                        px += counts
                        x_end = trunc(trunc(px / image_height)
                                      * adjusted_ratio)
                        y_end = trunc(px % image_height * adjusted_ratio)
                        if x_end == x_start:
                            # This is only on one line
                            rle_list.append(
                                {'x': x_start, 'y': y_start, 'width': 1, 'height': (y_end - y_start)})
                        if x_end > x_start:
                            # This spans more than one line
                            # Insert top line first
                            rle_list.append(
                                {'x': x_start, 'y': y_start, 'width': 1, 'height': (image_height - y_start)})

                            # Insert middle lines if needed
                            lines_spanned = x_end - x_start + 1  # total number of lines spanned
                            full_lines_to_insert = lines_spanned - 2
                            if full_lines_to_insert > 0:
                                full_lines_to_insert = trunc(
                                    full_lines_to_insert * adjusted_ratio)
                                rle_list.append(
                                    {'x': (x_start + 1), 'y': 0, 'width': full_lines_to_insert, 'height': image_height})

                            # Insert bottom line
                            rle_list.append(
                                {'x': x_end, 'y': 0, 'width': 1, 'height': y_end})
                if len(rle_list) > 0:
                    rle_regions[segm['id']] = rle_list
            else:
                # Add the polygon segmentation
                for segmentation_points in segm['segmentation']:
                    segmentation_points = np.multiply(
                        segmentation_points, adjusted_ratio).astype(int)
                    polygons_list.append(
                        str(segmentation_points).lstrip('[').rstrip(']'))

            polygons[segm['id']] = polygons_list

            if i < len(self.colors):
                poly_colors[segm['id']] = self.colors[i]
            else:
                poly_colors[segm['id']] = 'white'

            bbox = segm['bbox']
            print(bbox)
            bbox_points = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1],
                           bbox[0] + bbox[2], bbox[1] +
                           bbox[3], bbox[0], bbox[1] + bbox[3],
                           bbox[0], bbox[1]]
            bbox_points = np.multiply(bbox_points, adjusted_ratio).astype(int)
            bbox_polygons[segm['id']] = str(
                bbox_points).lstrip('[').rstrip(']')

            labels[segm['id']] = (
                self.categories[segm['category_id']]['name'], (bbox_points[0], bbox_points[1] - 4))

            # Print details
            print('    {}:{}:{}'.format(
                segm['id'], poly_colors[segm['id']], self.categories[segm['category_id']]))

        # Draw segmentation polygons on image
        html = '<div class="container" style="position:relative;">'
        html += '<img src="{}" style="position:relative;top:0px;left:0px;width:{}px;">'.format(
            img_str, adjusted_width)
        html += '<div class="svgclass"><svg width="{}" height="{}">'.format(
            adjusted_width, adjusted_height)

        if show_polys:
            for seg_id, points_list in polygons.items():
                fill_color = poly_colors[seg_id]
                stroke_color = poly_colors[seg_id]
                for points in points_list:
                    html += '<polygon points="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0.5" />'.format(
                        points, fill_color, stroke_color)

        if show_crowds:
            for seg_id, rect_list in rle_regions.items():
                fill_color = poly_colors[seg_id]
                stroke_color = poly_colors[seg_id]
                for rect_def in rect_list:
                    x, y = rect_def['x'], rect_def['y']
                    w, h = rect_def['width'], rect_def['height']
                    html += '<rect x="{}" y="{}" width="{}" height="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0.5; stroke-opacity:0.5" />'.format(
                        x, y, w, h, fill_color, stroke_color)

        if show_bbox:
            for seg_id, points in bbox_polygons.items():
                fill_color = poly_colors[seg_id]
                stroke_color = poly_colors[seg_id]
                html += '<polygon points="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0" />'.format(
                    points, fill_color, stroke_color)

        if show_labels:
            for seg_id, label in labels.items():
                color = poly_colors[seg_id]
                html += '<text x="{}" y="{}" style="fill:{}; font-size: 12pt;">{}</text>'.format(
                    label[1][0], label[1][1], color, label[0])

        html += '</svg></div>'
        html += '</div>'
        html += '<style>'
        html += '.svgclass { position:absolute; top:0px; left:0px;}'
        html += '</style>'
        return html

    def process_info(self):
        self.info = self.coco['info']

    def process_licenses(self):
        self.licenses = self.coco['licenses']

    def process_categories(self):
        self.categories = {}
        self.super_categories = {}
        for category in self.coco['categories']:
            cat_id = category['id']
            super_category = category['supercategory']

            # Add category to the categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
            else:
                print("ERROR: Skipping duplicate category id: {}".format(category))

            # Add category to super_categories dict
            if super_category not in self.super_categories:
                # Create a new set with the category id
                self.super_categories[super_category] = {cat_id}
            else:
                self.super_categories[super_category] |= {
                    cat_id}  # Add category id to the set

    def process_images(self):
        self.images = {}
        for image in self.coco['images']:
            image_id = image['id']
            if image_id in self.images:
                print("ERROR: Skipping duplicate image id: {}".format(image))
            else:
                self.images[image_id] = image

    def process_segmentations(self):
        self.segmentations = {}
        for segmentation in self.coco['annotations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)

    def get_labels(self, width=None, height=None, format=('x1', 'y1', 'w', 'h'), relative=False):
        format_dict = {'x1': lambda x: x[0], 'x2': lambda x: x[0] + x[2], 'y1': lambda x: x[1],
                       'y2': lambda x: x[1] + x[3], 'w': lambda x: x[2], 'h': lambda x: x[3],
                       'xc': lambda x: x[0] + x[2] / 2.0, 'yc': lambda x: x[1] + x[3] / 2.0}
        all_imgs_labels = {}
        if ((width is not None) or (height is not None)) and relative:
            raise ValueError(
                "Can't be relative and have certain width or height")

        for img_id in self.segmentations.keys():
            img_name = self.images[img_id]['file_name']
            all_imgs_labels[img_name] = {}
            img_labels = all_imgs_labels[img_name]
            width_ratio, height_ratio = 1.0, 1.0
            if width is not None:
                width_ratio = width / self.images[img_id]['width']
            if height is not None:
                height_ratio = height / self.images[img_id]['height']
            if relative:
                width_ratio /= self.images[img_id]['width']
                height_ratio /= self.images[img_id]['height']
            for segm in self.segmentations[img_id]:
                cat_name = self.categories[segm['category_id']]['name']
                bbox = segm['bbox'].copy()
                if cat_name not in img_labels:
                    img_labels[cat_name] = []
                bbox[0] *= width_ratio
                bbox[2] *= width_ratio
                bbox[1] *= height_ratio
                bbox[3] *= height_ratio
                new_bbox = [format_dict[element]
                            (bbox) for element in format]
                img_labels[cat_name].append(new_bbox)
        self.all_imgs_labels = dict(collections.OrderedDict(sorted(all_imgs_labels.items(), key=lambda x: int(x[0][:-4]))))
        return self.all_imgs_labels
        
    def write_labels_to_file(self,
                             directory,
                             separator=' ',
                             add_class_name=True,
                             add_class_id=False,
                             add_conf=False):
        
        classes_ids = { val['name']: val['id'] - 1 for val in  self.categories.values()}
        # Convert to string and adjust box format.
        for img_name, img in self.all_imgs_labels.items():
            formated_string_labels = []
            for c in img:
                for box in img[c]:
                    string = c + separator if add_class_name else ""
                    string += str(classes_ids[c]) + separator if add_class_id else ""
                    string += str(box[-1]) + separator if add_conf else ""
                    string += str(box[0])
                    for x in box[1:4]:
                        string += separator + str(x)
                    string += "\n"
                    formated_string_labels.append(string)
            # Writing data to a file.
            with open(directory + img_name[:-3] + "txt", "w") as file: 
                file.writelines(formated_string_labels)
                
    def read_labels_from_files(self, directory, has_conf=True, add_conf=False, box_formatter=lambda x: x):
        classes_names = {val['id'] - 1: val['name'] for val in self.categories.values()}
        all_imgs_labels = {}
        for filename in os.listdir(directory):
            if not filename.endswith(".txt"):
                continue
            imgname = filename.replace('.txt', '.jpg')
            # first get all lines from file
            with open(directory + filename, 'r') as f:
                lines = f.readlines()

            # remove spaces
            img_labels = {}
            for line in lines:
                label = line.replace('\n', '').split(sep=' ')
                cname = classes_names[int(label[0])]
                if cname not in img_labels:
                    img_labels[cname] = []
                if has_conf:
                    img_labels[cname].append(box_formatter(label[1:-1]))
                    if add_conf:
                        img_labels[cname][-1].append(float(label[-1]))
                else:
                    img_labels[cname].append(box_formatter(label[1:]))
            all_imgs_labels[imgname] = img_labels
        self.all_imgs_labels = dict(collections.OrderedDict(
            sorted(all_imgs_labels.items(), key=lambda x: int(x[0][:-4]))))
        return self.all_imgs_labels

if __name__ == "__main__":

    import collections

    annotation_path = '/home/abdelrhman/data/laptop_components/train2/laptop_components.json'
    image_dir = '/home/abdelrhman/data/laptop_components/train1'

    coco_dataset = CocoDataset(annotation_path, image_dir)
    all_imgs_labels = coco_dataset.get_labels(
        format=('y1', 'x1', 'y2', 'x2'), relative=True)
    all_imgs_labels = collections.OrderedDict(
        sorted(all_imgs_labels.items(), key=lambda x: int(x[0][:-4])))
    print(all_imgs_labels.keys())
