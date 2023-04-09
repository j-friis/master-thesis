## import the tools
import torch
import glob
from pathlib import Path
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
import glob
import pandas as pd
import ipdb
import laspy
import open3d as o3d
from tqdm.notebook import tqdm

## import the model tools
from torch_geometric.transforms import Compose
from torch_points3d.core.data_transform import MinPoints,XYZFeature, AddFeatsByKeys, GridSampling3D
from torch_points3d.applications.pretrained_api import PretainedRegistry
from torch_geometric.data import Batch, Dataset, Data ,DataLoader
from nearst_neighbors import get_nearest_neighbors
def predict(room_info, model, transform_test, test_folder, vis_folder):
    ## loop for every files
    room_names = room_info['room_names']
    room_coord_mins = room_info['room_coord_min']
    room_coord_scales = room_info['room_coord_scale']
    files = list(glob.glob(test_folder + "*cloud*pt"))

    for file in files:
        #print(file)
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
            print(file)
            print(data_s)
            pre_ori[0] = cla_pre[0]
        else:
            for i in pre_ori:
                pre_ori[i] = cla_pre[f[i]]
        combine_pre = np.column_stack((pt_ori, pre_ori.T))

        vis_out = vis_folder + room_name +'pre.txt'
        if os.path.exists(vis_out):
            file_save = open(vis_out, 'a')
        else:
            file_save = open(vis_out, 'w')

        file_save = open(vis_folder + room_name +'pre.txt', 'a')
        np.savetxt(file_save, combine_pre, fmt = '%1.5f')
    #     file_save.write("\n")    
    #     break
    print("save finished")
