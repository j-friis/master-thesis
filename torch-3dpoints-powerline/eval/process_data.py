## import the tools
import torch
from pathlib import Path
import os
import numpy as np
import glob
import os

def data_processing(config):
    ## load test pt with normalized

    ## load transform pt pre
    processed_folder_name = config['run_config']['data']["processed_folder"] # "processed_hough"
    data_root_path = config['run_config']['data']['dataroot'] + "/denmark"
    processed_data_root_path = os.path.join(data_root_path, processed_folder_name)
    test_fold = os.path.join(processed_data_root_path, "test_0_(0.1, 0.1)")
    pre_trans_path = os.path.join(test_fold, "stats.pt")
    vis_out_folder =  os.path.join(processed_data_root_path, 'vis')

    pre_transform = torch.load(pre_trans_path)
