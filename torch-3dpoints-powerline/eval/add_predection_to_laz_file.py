## import the tools
from pathlib import Path
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
import pandas as pd
import laspy
import open3d as o3d
from tqdm.notebook import tqdm
from nearst_neighbors import get_nearest_neighbors
## import the model tools

def add_predection_to_laz_files(filename, data_root_path, vis_folder, processed_folder_name):
    ## read path
    pred_path = os.path.join(vis_folder, filename+"pre.txt")
    pred_data = pd.read_csv(pred_path, sep=" ", header=None).values

    normal_laz_file = os.path.join(data_root_path, "raw", "test",filename+".laz")
    processed_laz_file = os.path.join(data_root_path, "raw", "test", "NewLaz", filename+".laz")
    print(normal_laz_file)
    print(processed_laz_file)
    non_processed_laz = laspy.read(normal_laz_file, laz_backend=laspy.compression.LazBackend.LazrsParallel)
    processed_laz = laspy.read(processed_laz_file, laz_backend=laspy.compression.LazBackend.LazrsParallel)
    non_processed_point_data = np.stack([non_processed_laz.X, non_processed_laz.Y, non_processed_laz.Z], axis=0).transpose((1, 0))
    processed_point_data = np.stack([processed_laz.X, processed_laz.Y, processed_laz.Z], axis=0).transpose((1, 0))
    print(len(non_processed_point_data))
    print(len(processed_point_data))


    powerline_pts = pred_data[np.where(pred_data[:,3] == 1)].copy()
    powerline_pts_coord = powerline_pts[:,:-1].astype(np.int32) 

    print(f"There are {pred_data.shape} points in the las file")
    print(f"There are {pred_data.shape} points from the pt files")

    print(powerline_pts_coord[:10])
    print(processed_point_data[:10])

    print(f"laz data x max {np.max(processed_point_data[:,0])}")
    print(f"pt x max {np.max(powerline_pts_coord[:,0])}")
    print(f"laz data y max {np.max(processed_point_data[:,1])}")
    print(f"pt y max {np.max(powerline_pts_coord[:,1])}")
    print(f"laz data z max {np.max(processed_point_data[:,2])}")
    print(f"pt z max {np.max(powerline_pts_coord[:,2])}")

    processed_laz.add_extra_dim(laspy.ExtraBytesParams(
        name="prediction",
        type=np.uint8,
        description="The prediction of the model"
        ))

    idx = get_nearest_neighbors(powerline_pts_coord,processed_point_data)
    pred = np.zeros(len(processed_point_data))
    pred[idx] = 1
    processed_laz.prediction = pred

    processed_data_root_path = os.path.join(data_root_path, processed_folder_name)
    eval_folder = os.path.join(processed_data_root_path, "eval")
    Path(eval_folder).mkdir(exist_ok=True, parents=True)
    eval_file_name = os.path.join(eval_folder, filename+".laz")
    processed_laz.write(str(eval_file_name), do_compress =True, laz_backend=laspy.compression.LazBackend.LazrsParallel)