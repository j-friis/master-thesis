import argparse
from pathlib import Path
import glob
import os

## import the tools
from pathlib import Path
import os
import numpy as np
import pandas as pd
import laspy

import torch
## to find the neighbor points prediction
from sklearn.neighbors import KDTree
import numpy as np

## import the model tools
from torch_geometric.transforms import Compose
from torch_points3d.core.data_transform import MinPoints,XYZFeature, AddFeatsByKeys, GridSampling3D
from torch_points3d.applications.pretrained_api import PretainedRegistry
from torch_geometric.data import Batch

def get_nearest_neighbors(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""
    tree = KDTree(candidates, leaf_size=20, metric='euclidean')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    closest = np.squeeze(indices)
    return closest


def load_model(model_path: str, data_path: str):
    model = torch.load(model_path)
    model['run_config']['data']['dataroot'] = data_path
    torch.save(model, model_path)
    print(model['run_config']["data"]["train_transform"])
    ## transformer
    pos_z = [ "pos_z" ]
    list_add_to_x = [ True ]
    delete_feats = [ True ]
    lparams = ['512']

    first_subsampling = model['run_config']["data"]["first_subsampling"]
    transform_test = Compose([MinPoints(512),
                        XYZFeature(add_x=False, add_y=False, add_z= True),
                        AddFeatsByKeys(list_add_to_x=list_add_to_x, feat_names= pos_z,delete_feats=delete_feats),
                        GridSampling3D(mode='last', size=first_subsampling, quantize_coords=True)
                        ])

    model_pl = PretainedRegistry.from_file(model_path).cuda()
    return model_pl, transform_test, model['run_config']['data']


def predict(room_info, model, transform_test, test_folder, vis_folder):
    ## loop for every files
    room_names = room_info['room_names']
    room_coord_mins = room_info['room_coord_min']
    room_coord_scales = room_info['room_coord_scale']
    files = list(glob.glob(test_folder + "/*cloud*pt"))

    for file in files:
        sample = os.path.join(test_folder, file)
        pt_data = torch.load(sample)
        room_index = pt_data['room_idx']
        room_name = room_names[room_index]
        vis_out = os.path.join(vis_folder, room_name)
        Path(vis_folder).mkdir(exist_ok=True, parents=True)

        room_coord_scale = room_coord_scales[room_index]
        pos_ = pt_data['points']
        pt_ori = pos_ * room_coord_scale + room_coord_mins[room_index]
        data_s = transform_test(Batch(pos=torch.from_numpy(pos_).float(), batch=torch.zeros(pos_.shape[0]).long()))
        data_s.y = torch.zeros(data_s.batch.shape).long()
        f = get_nearest_neighbors(pos_, data_s.pos)


        with torch.no_grad():
            model.eval()
            model.set_input(data_s, "cuda")
            model.forward(data_s)
        
        pre = model.output.cpu().numpy()
        m = torch.nn.functional.softmax(torch.tensor(pre), dim=1)
        cla_pre = np.argmax(m, axis=1)
        pre_ori = np.arange(len(pos_))
        if len(pos_) == 1:
            pre_ori[0] = cla_pre[0]
        else:
            for i in pre_ori:
                pre_ori[i] = cla_pre[f[i]]
        combine_pre = np.column_stack((pt_ori, pre_ori.T))

        vis_out_file = vis_out +'pre.txt'
        if os.path.exists(vis_out_file):
            file_save = open(vis_out_file, 'a')
        else:
            file_save = open(vis_out_file, 'w')

        file_save = open(vis_out_file, 'a')
        np.savetxt(file_save, combine_pre, fmt = '%1.5f')
    print("save finished")


def add_predection_to_laz_files(filename, data_root_path, vis_folder, processed_folder_name):
    ## read path
    pred_path = os.path.join(vis_folder, filename+"pre.txt")
    pred_data = pd.read_csv(pred_path, sep=" ", header=None).values

    normal_laz_file = os.path.join(data_root_path, "raw", "test",filename+".laz")
    processed_laz_file = os.path.join(data_root_path, "raw", "test", "NewLaz", filename+".laz")

    non_processed_laz = laspy.read(normal_laz_file, laz_backend=laspy.compression.LazBackend.LazrsParallel)
    processed_laz = laspy.read(processed_laz_file, laz_backend=laspy.compression.LazBackend.LazrsParallel)
    non_processed_point_data = np.stack([non_processed_laz.X, non_processed_laz.Y, non_processed_laz.Z], axis=0).transpose((1, 0))
    processed_point_data = np.stack([processed_laz.X, processed_laz.Y, processed_laz.Z], axis=0).transpose((1, 0))


    powerline_pts = pred_data[np.where(pred_data[:,3] == 1)].copy()
    powerline_pts_coord = powerline_pts[:,:-1].astype(np.int32) 

    # print(f"There are {pred_data.shape} points in the las file")
    # print(f"There are {pred_data.shape} points from the pt files")

    # print(powerline_pts_coord[:10])
    # print(processed_point_data[:10])

    # print(f"laz data x max {np.max(processed_point_data[:,0])}")
    # print(f"pt x max {np.max(powerline_pts_coord[:,0])}")
    # print(f"laz data y max {np.max(processed_point_data[:,1])}")
    # print(f"pt y max {np.max(powerline_pts_coord[:,1])}")
    # print(f"laz data z max {np.max(processed_point_data[:,2])}")
    # print(f"pt z max {np.max(powerline_pts_coord[:,2])}")

    non_processed_laz.add_extra_dim(laspy.ExtraBytesParams(
        name="prediction",
        type=np.uint8,
        description="The prediction of the model"
        ))

    idx = get_nearest_neighbors(powerline_pts_coord, non_processed_point_data)
    pred = np.zeros(len(non_processed_point_data))
    pred[idx] = 1
    non_processed_laz.prediction = pred

    processed_data_root_path = os.path.join(data_root_path, processed_folder_name)
    eval_folder = os.path.join(processed_data_root_path, "eval")
    Path(eval_folder).mkdir(exist_ok=True, parents=True)
    eval_file_name = os.path.join(eval_folder, filename+".laz")
    non_processed_laz.write(str(eval_file_name), do_compress =True, laz_backend=laspy.compression.LazBackend.LazrsParallel)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eval entire folder.')
    parser.add_argument('folder', type=str, help='Folder with laz files to predict on')
    parser.add_argument('model', type=str, help='Path to the model')

    args = parser.parse_args()
    data_path = args.folder
    model_path = args.model

    # data_path = "/home/jf/msc_data"
    # model_path = "/home/jf/Documents/msc/torch-3dpoints-powerline/outputs/2023-04-09/13-31-16/SEUnet1850Metersblock50cmvoxel.pt"

    model, transform_test, config = load_model(model_path, data_path)

    ## load transform pt pre
    processed_folder_name = config["processed_folder"] 
    data_root_path = os.path.join(config['dataroot'] , "denmark")
    processed_data_root_path = os.path.join(data_root_path, processed_folder_name)
    test_folder_name = f"test_0_({config['block_size_x']}, {config['block_size_y']})"
    test_folder = os.path.join(processed_data_root_path, test_folder_name)
    pre_trans_path = os.path.join(test_folder, "stats.pt")
    vis_folder =  os.path.join(processed_data_root_path, 'vis')
    #Path(vis_folder).mkdir(exist_ok=True, parents=True)

    room_info = torch.load(pre_trans_path)
    predict(room_info, model, transform_test, test_folder, vis_folder)
    for filename in room_info['room_names']:
        add_predection_to_laz_files(filename,data_root_path, vis_folder,processed_folder_name)
