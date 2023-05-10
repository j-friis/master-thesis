import pandas as pd
import numpy as np
import glob
import laspy
import cv2

import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import rasterio
from rasterio.features import shapes

import shapely
from shapely.geometry import Polygon, mapping
from PIL import Image
from matplotlib.path import Path as plt_path

class PolygonCNN(object):
    def __init__(self, path_to_data, path_to_model, network_size, image_size):
        self.path_to_data = path_to_data
        self.model = torch.load(path_to_model)
        self.network_size = network_size
        self.image_size = image_size
        self.transform_img_gray = transforms.Compose([transforms.Resize((image_size,image_size)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def __call__(self, filename):
        lines_image = self.ImageProcessing(filename)
        reg_polygons, multi_polygons, bbox_reg_polygon, bbox_multi_polygons = self.Polygonize(lines_image)
        point_cloud = laspy.read(self.path_to_data+'/LazFilesWithHeightParam/'+filename+'_hag_nn.laz', laz_backend=laspy.compression.LazBackend.LazrsParallel)
        indexes_needed = self.FilterPolygons(reg_polygons, multi_polygons, bbox_reg_polygon, bbox_multi_polygons, point_cloud, lines_image)
        new_las = self.Predictions(indexes_needed, point_cloud)
        return new_las

    def ImageProcessing(self, filename):
        # Load Image
        image = cv2.imread(self.path_to_data+'/ImagesGroundRemoved/'+filename+'_max.tif', cv2.IMREAD_UNCHANGED)
        image = np.where(image >= 0, image, 0)
        image = image/np.max(image)
        image = (image*255).astype(np.uint8)

        # Create pil image
        image = Image.fromarray(image)

        # Resize Image, Transform to tensor, and Normalize
        image = self.transform_img_gray(image)
        x_pixels, y_pixels = image[0].shape
        
        # Crop the image into equal parts
        network_size = self.network_size
        image_size = self.image_size
        amount_of_crops = image_size // network_size
        
        cropped_image_list = []
        for i in range(amount_of_crops):
            for j in range(amount_of_crops):
                
                # Generate slice indices
                x_start_index = network_size*i
                x_end_index = network_size*(i+1)
                y_start_index = network_size*j
                y_end_index = network_size*(j+1)
                
                # Slice image of size [1, image_size, image_size] and obtain the cropped image
                cropped_image = image[0][x_start_index:x_end_index,y_start_index:y_end_index]
                cropped_image_list.append(cropped_image)

        # Apply model
        output_list = []
        self.model.eval()
        with torch.no_grad():
            for small_img in cropped_image_list:
                outputs = torch.squeeze(self.model(small_img.cuda())).detach().cpu()
                outputs = torch.round(torch.sigmoid(outputs)) # Predict either 0 or 1.
                output_list.append(outputs)
            
        row_images = []
        for i in range(amount_of_crops):
            # to obtain each row in an image
            row_to_concat = output_list[(i)*amount_of_crops:(i+1)*amount_of_crops]
            stacked_array = np.concatenate([arr for arr in row_to_concat], axis=1)
            row_images.append(stacked_array)
            
        lines_image = np.concatenate([arr for arr in row_images], axis=0)

        # Get pixels per meter to create a cirkular kernel size of size "meters_around_line"
        
        x_pixels, y_pixels = x_pixels/1000, y_pixels/1000
        meters_around_line = self.meters_around_line
        kernel_size = int(meters_around_line*np.ceil(x_pixels))
        circular_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Create a cirkular kernel using (image, center_coordinates, radius, color, thickness)
        cv2.circle(circular_kernel, (int(kernel_size/2), int(kernel_size/2)), int(kernel_size/2), 255, -1)
        
        # Apply dilation using the cirkular kernel
        lines_image = cv2.dilate(lines_image, circular_kernel, iterations=1)
        return lines_image
    

    def Polygonize(self, lines_image):

        # Create Polygons and Multi Polygons
        mask = (lines_image == 255)
        output = rasterio.features.shapes(lines_image, mask=mask, connectivity=4)
        output_list = list(output)

        # Seperate the Multipolygons and Polygons
        all_polygons = []
        all_multi_polygons =[]
        #ipdb.set_trace()

        for multi_polygon in output_list:
            found_polygon = multi_polygon[0]['coordinates']
            # Then its just a Polygon
            if len(found_polygon) == 1:
                all_polygons.append(Polygon(found_polygon[0]))
            # Else its a multipolygon
            else:
                tmpMulti = []
                for p in found_polygon:
                    tmpMulti.append(Polygon(p))
                all_multi_polygons.append(tmpMulti)


        # Remove all low area multipolygons
        for i, multi_pol in enumerate(all_multi_polygons):
            new_list = [multi_pol[0]]
            # No matter what, dont remove the first one
            for pol in multi_pol[1:]:
                if pol.area > 1000:
                    new_list.append(pol)
            all_multi_polygons[i] = new_list


        simplified_all_polygons = []
        simplified_all_multi_polygons =[]
        # Simplify all standard polygons
        for p in all_polygons:
            simplified_all_polygons.append(shapely.simplify(p, tolerance=self.simplify_tolerance, preserve_topology=True))
        simplified_all_polygons  = [p for p in simplified_all_polygons if not p.is_empty]

        # Simplify all multi polygons
        for multi_pol in all_multi_polygons:
            tmp = []
            for p in multi_pol:
                tmp.append(shapely.simplify(p, tolerance=self.simplify_tolerance, preserve_topology=True))
            tmp  = [p for p in tmp if not p.is_empty]
            simplified_all_multi_polygons.append(tmp)

        # Create bounding box polygons
        bbox_all_polygon_path = []
        tmp = [p.bounds for p in simplified_all_polygons]
        for values in tmp:
            #values = (minx, miny, maxx, maxy)
            x_min = values[0]
            x_max = values[2]
            y_min = values[1]
            y_max = values[3]
            bb = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
            bbox_all_polygon_path.append(plt_path(bb))

        # Create bounding box for multi polygons
        bbox_all_multi_polygons_path = []
        for multi_pol in simplified_all_multi_polygons:
            tmp = [p.bounds for p in multi_pol]
            tmp_multi_pol_boxes = []

            for values in tmp:
                #values = (minx, miny, maxx, maxy)
                x_min = values[0]
                x_max = values[2]
                y_min = values[1]
                y_max = values[3]
                bb = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
                tmp_multi_pol_boxes.append(plt_path(bb))
            bbox_all_multi_polygons_path.append(tmp_multi_pol_boxes)


        # Create plt_path polygons from the simplified shapely polygons
        simplified_all_polygons_path = [plt_path(mapping(p)['coordinates'][0]) for p in simplified_all_polygons]
        simplified_all_multi_polygons_path = []
        for multi_pol in simplified_all_multi_polygons:
            tmp = [plt_path(mapping(p)['coordinates'][0]) for p in multi_pol]
            simplified_all_multi_polygons_path.append(tmp)

        return simplified_all_polygons_path, simplified_all_multi_polygons_path, bbox_all_polygon_path, bbox_all_multi_polygons_path
    
    def MaxMinNormalize(self, arr):
        return (arr - np.min(arr))/(np.max(arr)-np.min(arr))

    def CastAllXValuesToImage(self, arr, x_pixels):
        return self.MaxMinNormalize(arr)*x_pixels

    def CastAllYValuesToImage(self, arr, y_pixels):
        return (1-self.MaxMinNormalize(arr))*y_pixels
    
    def FilterPolygons(self, reg_polygons, multi_polygons, bbox_reg_polygon, bbox_multi_polygons, point_cloud, image):
        # Pixels per kilometer
        x_pixels, y_pixels = image.shape
        x_values = self.CastAllXValuesToImage(point_cloud.X, x_pixels)
        y_values = self.CastAllYValuesToImage(point_cloud.Y, y_pixels)

        # Format: [(1,1), (3,5), (1,5), ...] with 30 mio samples
        list_zipped = np.array(list(zip(x_values, y_values)))

        # Generate a bool list to obtain the final indexes from the dataset
        indexes_needed = np.zeros(len(x_values), dtype=bool)

        # Run through all polygons and check which points are inside the polygon
        for i in range(len(reg_polygons)):
            # Check if point is inside the bounding box
            indexes_inside_box = bbox_reg_polygon[i].contains_points(list_zipped)
            indexes_inside_box = np.array([index for index, x in enumerate(indexes_inside_box) if x])

            # Generate small dataset
            tmp = list_zipped[indexes_inside_box]

            # Check if any of these points are in the polygon
            indexes_inside_polygon = reg_polygons[i].contains_points(tmp)

            # Find the indexes from the box that is also inside the polygon
            final_indexes = indexes_inside_box[indexes_inside_polygon]

            # Update the indexes
            indexes_needed[final_indexes] = 1

        for i in range(len(multi_polygons)):
            tmp_indexes_needed = np.zeros(len(x_values), dtype=bool)
            tmp_indexes_not_needed = np.zeros(len(x_values), dtype=bool)

            # Get the current bb multipolygon and the current simplified multipolygon
            bb_multi_pol = bbox_multi_polygons[i]
            simpli_multi_pol = multi_polygons[i]

            # Find the indexes that are inside the bounding box of the first element
            indexes_inside_box = bb_multi_pol[0].contains_points(list_zipped)
            indexes_inside_box = np.array([index for index, x in enumerate(indexes_inside_box) if x])

            # Generate smaller dataset
            tmp = list_zipped[indexes_inside_box]

            # Check if any of these points are in the polygon
            indexes_inside_polygon = simpli_multi_pol[0].contains_points(tmp)

            # Find the indexes from the box that is also inside the polygon
            final_indexes = indexes_inside_box[indexes_inside_polygon]
            tmp_indexes_needed[final_indexes] = 1

            for j in range(1, len(bb_multi_pol)):

                # Get the bounding box of the temp multi polygon

                indexes_inside_box = bb_multi_pol[j].contains_points(list_zipped)
                indexes_inside_box = np.array([index for index, x in enumerate(indexes_inside_box) if x])

                # Generate small dataset
                tmp = list_zipped[indexes_inside_box]

                # Check if any of these points are in the polygon
                indexes_inside_polygon = simpli_multi_pol[j].contains_points(tmp)
                final_indexes = indexes_inside_box[indexes_inside_polygon]

                # Update the indexes
                tmp_indexes_not_needed[final_indexes] = 1

            indexes_needed = indexes_needed | (tmp_indexes_needed & np.invert(tmp_indexes_not_needed))
        return indexes_needed
    
    def Predictions(self, indexes_needed, point_cloud):
        new_point_cloud = point_cloud[indexes_needed]
        return new_point_cloud