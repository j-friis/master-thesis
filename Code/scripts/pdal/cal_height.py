import argparse
import pdal
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

# out_file = file_name.split(".")[0]

# json = """
# [
#     "%s",
#     {
#         "type":"filters.hag_delaunay"
#     },
#     {
#         "type":"writers.las",
#         "filename":"%s_hag_delaunay.laz",
#         "extra_dims":"HeightAboveGround=float32",
#         "compression":"laszip"
#     }
# ]
# """ % (file_name, out_file)

def cal_height(dir: str):
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f)) 
                    and "_hag_delaunay" not in f and ".tif" not in f and "height" not in f]
    print(onlyfiles)

    for file in tqdm(onlyfiles):
        file_name = file
        file_name = join(dir, file_name)
        out_file = file_name.split(".")[0]
        out_file = join(dir, out_file)
        json = """
        [
            "%s",
            {
                "type":"filters.hag_nn"
            },
            {
                "type":"writers.las",
                "filename":"%s_hag_nn.laz",
                "extra_dims":"HeightAboveGround=float32",
                "compression":"laszip"
            }
        ]
        """ % (file_name, out_file)
        pipeline = pdal.Pipeline(json)
        count = pipeline.execute()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate the height from the ground.')
    parser.add_argument('folder', type=str, help='Folder with files to calculation')

    args = parser.parse_args()
    dir = args.folder
    cal_height(dir)