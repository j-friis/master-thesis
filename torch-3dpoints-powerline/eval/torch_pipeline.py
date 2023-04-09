import argparse
from pathlib import Path
import os

import torch
from load_model import load_model
from predict import predict
from add_predection_to_laz_file import add_predection_to_laz_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eval entire folder.')
    parser.add_argument('folder', type=str, help='Folder with laz files to predict on')
    parser.add_argument('model', type=str, help='Path to the model')

    args = parser.parse_args()
    data_path = args.folder
    model_path = args.model

    model, transform_test, config = load_model(model_path, data_path)

    ## load transform pt pre
    processed_folder_name = config['run_config']['data']["processed_folder"] 
    data_root_path = config['run_config']['data']['dataroot'] + "/denmark"
    processed_data_root_path = os.path.join(data_root_path, processed_folder_name)
    test_folder = os.path.join(processed_data_root_path, "test_0_(0.1, 0.1)")
    pre_trans_path = os.path.join(test_folder, "stats.pt")
    vis_folder =  os.path.join(processed_data_root_path, 'vis')

    room_info = torch.load(pre_trans_path)
    predict(room_info, model, transform_test, test_folder, vis_folder)
    for filename in room_info['room_names']:
        add_predection_to_laz_files(filename,data_root_path, vis_folder,processed_folder_name)
