## import the tools
import torch
from sklearn.neighbors import NearestNeighbors as NN
from tqdm.notebook import tqdm

## import the model tools
from torch_geometric.transforms import Compose
from torch_points3d.core.data_transform import MinPoints,XYZFeature, AddFeatsByKeys, GridSampling3D
from torch_points3d.applications.pretrained_api import PretainedRegistry

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