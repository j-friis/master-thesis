import os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

import numpy as np
import cv2
import glob
import laspy

import rasterio
from rasterio.features import shapes

from shapely.geometry import Polygon, mapping
import shapely
from matplotlib.path import Path

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from tqdm import tqdm

class TemplateClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, path="~/data/", canny_lower=4, canny_upper=160,hough_lines_treshold=10,
                  min_line_length=5,max_line_gap=10, closing_kernel_size=5,
                    opening_kernel_size=3, meters_around_line=1, simplify_tolerance=1):
        self.path = path
        self.canny_lower = canny_lower
        self.canny_upper = canny_upper
        self.hough_lines_treshold = hough_lines_treshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.closing_kernel_size = closing_kernel_size
        self.opening_kernel_size = opening_kernel_size
        self.meters_around_line = meters_around_line
        self.simplify_tolerance = simplify_tolerance
        self.indexes_needed_list = []


    def GenPath(self, path):
        if path[-1] == '/':
            return path
        else:
            return path+'/'

    def GetPathRelations(self):
        full_path_to_data = self.GenPath(self.path)
        
        ground_removed_image_paths = []
        laz_point_cloud_paths = []
        
        # Find full path to all images
        for path in glob.glob(full_path_to_data+'ImagesGroundRemoved/*'):
            ground_removed_image_paths.append(path)
    
        # Find full path to all laz files
        for path in glob.glob(full_path_to_data+'LazFilesWithHeightParam/*'):
            laz_point_cloud_paths.append(path)
            
        ground_removed_image_paths.sort()
        laz_point_cloud_paths.sort()
        assert(len(ground_removed_image_paths)==len(laz_point_cloud_paths))
        return ground_removed_image_paths, laz_point_cloud_paths

    def hough_lines(self, file: str):

        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        image = np.where(image >= 0, image, 0)
        image = image/np.max(image)

        image = (image*255).astype(np.uint8)

        #kernel = np.ones((70,70),np.uint8)
        closing_kernel = np.ones((self.closing_kernel_size,self.closing_kernel_size),np.uint8)
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, closing_kernel)

        # Apply edge detection method on the image
        edges = cv2.Canny(closing, self.canny_lower, self.canny_upper, None, 3)

        linesP = cv2.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=self.hough_lines_treshold, # Min number of votes for valid line
            minLineLength=self.min_line_length, # Min allowed length of line
            maxLineGap=self.max_line_gap # Max allowed gap between line for joining them
            )

        lines_image = np.zeros_like(edges)

        # Draw the lines
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(lines_image, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3)

        #kernel = np.ones((5,5),np.uint8)
        opening_kernel = np.ones((self.opening_kernel_size,self.opening_kernel_size),np.uint8)
        opening = cv2.morphologyEx(lines_image, cv2.MORPH_OPEN, opening_kernel)
        #dilation = cv2.dilate(lines_image, self.opening_kernel, iterations = 7)


        # Pixels per kilometer
        x_pixels, y_pixels = image.shape

        # Pixels per meter
        x_per_km_pixels, y_per_km_pixels = x_pixels/1000, y_pixels/1000

        # Set kernel size to 1 meter around the each line
        kernel_size = int(self.meters_around_line*np.ceil(x_per_km_pixels))
        # Create kernel
        circular_kernel = np.zeros((kernel_size, kernel_size), np.uint8)

        # Create a cirkular kernel using (image, center_coordinates, radius, color, thickness)
        cv2.circle(circular_kernel, (int(kernel_size/2), int(kernel_size/2)), int(kernel_size/2), 255, -1)

        # Perform dilation with the cirkular kernel
        dilation_cirkular_kernel = cv2.dilate(opening, circular_kernel, iterations=3)

        return dilation_cirkular_kernel, image


    def make_polygons(self, dilation_cirkular_kernel):
        # Create Polygons and Multi Polygons

        mask = (dilation_cirkular_kernel == 255)
        output = rasterio.features.shapes(dilation_cirkular_kernel, mask=mask, connectivity=4)
        output_list = list(output)

        # Seperate the Multipolygons and Polygons
        all_polygons = []
        all_multi_polygons =[]



        for multi_polygon in output_list:
            found_polygon = multi_polygon[0]['coordinates']
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
            bbox_all_polygon_path.append(Path(bb))
            
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
                tmp_multi_pol_boxes.append(Path(bb))
            bbox_all_multi_polygons_path.append(tmp_multi_pol_boxes)


        # Create Path polygons from the simplified shapely polygons
        simplified_all_polygons_path = [Path(mapping(p)['coordinates'][0]) for p in simplified_all_polygons]
        simplified_all_multi_polygons_path = []
        for multi_pol in simplified_all_multi_polygons:
            tmp = [Path(mapping(p)['coordinates'][0]) for p in multi_pol]
            simplified_all_multi_polygons_path.append(tmp)


        print(f"Amount of polygons: {len(simplified_all_polygons_path)}")
        print(f"Amount of multi polygons: {len(simplified_all_multi_polygons_path)}")

        return simplified_all_polygons_path, simplified_all_multi_polygons_path, bbox_all_polygon_path, bbox_all_multi_polygons_path
        

    def filter_polygons(self, reg_polygons, multi_polygons, bbox_reg_polygon, bbox_multi_polygons, point_cloud, image):
        
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
            print("Progress in Polygons: ", i/len(reg_polygons))

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
            print("Progress in MultiPolygons: ", i/len(multi_polygons))


        return indexes_needed

    def MaxMinNormalize(self, arr):
        return (arr - np.min(arr))/(np.max(arr)-np.min(arr))

    def CastAllXValuesToImage(self, arr, x_pixels):
        return self.MaxMinNormalize(arr)*x_pixels

    def CastAllYValuesToImage(self, arr, y_pixels):
        return (1-self.MaxMinNormalize(arr))*y_pixels
    

    def fit(self, X, y):

        ground_removed_image_paths, laz_point_cloud_paths = self.GetPathRelations()

        for tif, laz_file in zip(ground_removed_image_paths, laz_point_cloud_paths):


            dilation_cirkular_kernel, image = self.hough_lines(tif)
            print("hough_lines")
            reg_polygons, multi_polygons, bbox_reg_polygon, bbox_multi_polygons, = self.make_polygons(dilation_cirkular_kernel)
            #viz_polygon(dilation_cirkular_kernel, reg_polygons, multi_polygons)
            print("make_polygons")
            
            point_cloud = laspy.read(laz_file, laz_backend=laspy.compression.LazBackend.LazrsParallel)
            indexes_needed = self.filter_polygons(reg_polygons, multi_polygons, bbox_reg_polygon, bbox_multi_polygons, point_cloud, image)
            print("filter_polygons")
            self.indexes_needed_list.append(indexes_needed)

        return self

    def predict(self, X: str):

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]
    
    def score(self, _, __):
        _, laz_point_cloud_paths = self.GetPathRelations()


        pct_lost_datapoints_list = []
        pct_lost_powerline_list = []
        for i, laz_file in enumerate(laz_point_cloud_paths):
            point_cloud = laspy.read(laz_file, laz_backend=laspy.compression.LazBackend.LazrsParallel)
            indexes_needed = self.indexes_needed_list[i]
            new_point_data = point_cloud[indexes_needed]

            amount_wire = np.sum(point_cloud.classification == 14)
            new_amount_wire = np.sum(new_point_data.classification == 14)
            pct_lost_powerline = 1-(amount_wire/new_amount_wire)
            pct_lost_datapoints_list.append(pct_lost_powerline)

            amount_points = len(point_cloud)
            new_amount_points = len(new_point_data)

            pct_lost_datapoints = 1-(new_amount_points/amount_points)
            pct_lost_powerline_list.append(pct_lost_datapoints)
            print("Done")
        if np.mean(pct_lost_powerline_list) > 0.001:
            return 0
        else:
            return np.mean(pct_lost_datapoints_list)
        

if __name__ == "__main__":

    file = "/home/jf/Downloads/WingDownload/PUNKTSKY_00005_1km_6161_465"
    #clf = TemplateClassifier("/home/jf/data/", 70)
    #gg = clf.GetPathRelations()
    #print(gg)
    #print(file)
    # clf = TemplateClassifier(path="/home/jf/data/", canny_lower=1, canny_upper=1,hough_lines_treshold=1,
    #         min_line_length=1,max_line_gap=1, closing_kernel_size=1,
    #         opening_kernel_size=1, meters_around_line=1, simplify_tolerance=1)

    param = {
            "path": Categorical(["/home/jf/data/"]),
            "canny_lower": [1],
            "canny_upper": [1],
            "hough_lines_treshold": [1],
            "min_line_length": [1],
            "max_line_gap": [1],
            "closing_kernel_size": [1],
            "opening_kernel_size": [1],
            "meters_around_line": [1],
            "simplify_tolerance": [1],
    }

    opt = BayesSearchCV(
        TemplateClassifier(),search_spaces=param,
        cv=2,
        n_iter=1,
        n_jobs=1,
        random_state=0
    )
    # executes bayesian optimization
    _ = opt.fit([[1,2],[1,2],[1,2],[1,2]], [1,2,1,2])

    # model can be saved, used for predictions or scoring
    print(opt.best_score_)
    print(opt.best_params_)
