import matplotlib.pyplot as plt
import numpy as np
import cv2


import rasterio
from rasterio.features import shapes

import matplotlib.patches as mpatches
from shapely.geometry import Point, Polygon, shape, mapping
import shapely
import geopandas as gpd

from matplotlib.path import Path
import laspy
import open3d as o3d

def hough_lines(file: str):
    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    image = np.where(image >= 0, image, 0)
    image = image/np.max(image)

    image = (image*255).astype(np.uint8)
    #plt.title("Original Image")
    #plt.imshow(image, cmap='gray')
    #plt.show()

    kernel = np.ones((70,70),np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    #plt.title("Closing")
    #plt.imshow(closing, cmap='gray')
    #plt.show()


    #max_pooling = skimage.measure.block_reduce(closing, (10,10), np.max)

    #plt.title("Max Pooling")
    #plt.imshow(max_pooling, cmap='gray')
    #plt.show()


    # Apply edge detection method on the image
    edges = cv2.Canny(closing, 4, 160, None, 3)

    #plt.title("Edges")
    #plt.imshow(edges, cmap='gray')
    #plt.show()

    # Probabilistic Line Transform
    # min_line_length, max_line_gap

    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 100)

    #lines_image = np.zeros_like(image)
    lines_image = np.zeros_like(edges)

    # Draw the lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(lines_image, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3)

    #plt.title("Hough Lines")
    #plt.imshow(lines_image, cmap='gray')
    #plt.show()

    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(lines_image, kernel, iterations = 7)

    #plt.imshow(dilation, cmap='gray')
    #plt.title("Dialation")
    #plt.show()

    # Pixels per kilometer
    x_pixels, y_pixels = image.shape

    # Pixels per meter
    x_pixels, y_pixels = x_pixels/1000, y_pixels/1000

    # Set kernel size to 1 meter around the each line
    kernel_size = int(2*np.ceil(x_pixels))

    # Create kernel
    circular_kernel = np.zeros((kernel_size, kernel_size), np.uint8)

    # Create a cirkular kernel using (image, center_coordinates, radius, color, thickness)
    cv2.circle(circular_kernel, (int(kernel_size/2), int(kernel_size/2)), int(kernel_size/2), 255, -1)

    # Perform dilation with the cirkular kernel
    dilation_cirkular_kernel = cv2.dilate(dilation, circular_kernel, iterations=3)

    # plt.title("Dilation Cirkular Kernel")
    # plt.imshow(dilation_cirkular_kernel, cmap="gray")
    # plt.show()

    return dilation_cirkular_kernel, image

def make_polygons(dilation_cirkular_kernel):
    # Create Polygons and Multi Polygons

    mask = (dilation_cirkular_kernel == 255)
    output = rasterio.features.shapes(dilation_cirkular_kernel, mask=mask, connectivity=4)
    output_list = list(output)

    # Seperate the Multipolygons and Polygons
    all_polygons = []
    all_multi_polygons =[]



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
        simplified_all_polygons.append(shapely.simplify(p, tolerance=32, preserve_topology=True))
    simplified_all_polygons  = [p for p in simplified_all_polygons if not p.is_empty]

    # Simplify all multi polygons
    for multi_pol in all_multi_polygons:
        tmp = []
        for p in multi_pol:
            tmp.append(shapely.simplify(p, tolerance=32, preserve_topology=True))
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

def MaxMinNormalize(arr):
    return (arr - np.min(arr))/(np.max(arr)-np.min(arr))

def CastAllXValuesToImage(arr, x_pixels):
    return MaxMinNormalize(arr)*x_pixels

def CastAllYValuesToImage(arr, y_pixels):
    return (1-MaxMinNormalize(arr))*y_pixels

def filter_polygons(reg_polygons, multi_polygons, bbox_reg_polygon, bbox_multi_polygons, point_cloud, image):
    
    # Pixels per kilometer
    x_pixels, y_pixels = image.shape
    x_values = CastAllXValuesToImage(point_cloud.X, x_pixels)
    y_values = CastAllYValuesToImage(point_cloud.Y, y_pixels)

    # Format: [(1,1), (3,5), (1,5), ...] with 30 mio samples
    list_zipped = np.array(list(zip(x_values, y_values)))

    # Generate a bool list to obtain the final indexes from the dataset
    indexes_needed = np.zeros(len(x_values), dtype=bool)

    print("Start")
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

    new_point_cloud = point_cloud[indexes_needed]
    return new_point_cloud

def viz_polygon(dilation_cirkular_kernel, reg_polygons, multi_polygons):
    plt.figure()
    plt.imshow(dilation_cirkular_kernel, cmap='gray')

    for p in reg_polygons:
        x = p.vertices[:,0]
        y = p.vertices[:,1]
        plt.scatter(x,y, s=0.01, color='red')

    for multi_p in multi_polygons:
        for p in multi_p:
            x = p.vertices[:,0]
            y = p.vertices[:,1]
            plt.scatter(x,y, s=0.01, color='blue') 
    plt.title("Plot of polygons")
    plt.show()

def viz_cloud(point_cloud, new_point_cloud):
    point_data = np.stack([point_cloud.X, point_cloud.Y, point_cloud.Z], axis=0).transpose((1, 0))
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(point_data)
    o3d.visualization.draw_geometries([geom])

    point_data = np.stack([new_point_cloud.X, new_point_cloud.Y, new_point_cloud.Z], axis=0).transpose((1, 0))
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(point_data)
    o3d.visualization.draw_geometries([geom])
    

if __name__ == "__main__":

    file = "/home/jf/Downloads/WingDownload/PUNKTSKY_00005_1km_6161_465"

    dilation_cirkular_kernel, image = hough_lines(file+"_max.tif")

    reg_polygons, multi_polygons, bbox_reg_polygon, bbox_multi_polygons, = make_polygons(dilation_cirkular_kernel)
    #viz_polygon(dilation_cirkular_kernel, reg_polygons, multi_polygons)
    
    point_cloud = laspy.read(file+".laz", laz_backend=laspy.compression.LazBackend.LazrsParallel)
    new_point_cloud = filter_polygons(reg_polygons, multi_polygons, bbox_reg_polygon, bbox_multi_polygons, point_cloud, image)
    print("Amount of points in Point Cloud: ", len(point_cloud))
    print("Amount of points in new Point Cloud: ", len(new_point_cloud))
    print("Amount of points removed: ", len(point_cloud)-len(new_point_cloud))
    print(f"There are {np.sum(point_cloud.classification == 14)} wire conductor points in the non transformed data")
    print(f"There are {np.sum(new_point_cloud.classification == 14)} wire conductor points in the transformed data")
    viz_cloud(point_cloud, new_point_cloud)