import argparse
import pdal
import os
from os import listdir
from os.path import isfile, join
parser = argparse.ArgumentParser(description='Convert Laz to tif.')
parser.add_argument('folder', type=str, help='folder to convert files')

args = parser.parse_args()
dir = args.folder
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

if __name__ == "__main__":

    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f)) and "_hag_delaunay" not in f]
    print(onlyfiles)

    for file in onlyfiles:
        file_name = file
        file_name = join(dir, file_name)
        out_file = file_name.split(".")[0]
        out_file = join(dir, out_file)
        json = """
        [
            "%s",
            {
                "type":"filters.hag_delaunay"
            },
            {
                "type":"writers.las",
                "filename":"%s_hag_delaunay.laz",
                "extra_dims":"HeightAboveGround=float32",
                "compression":"laszip"
            }
        ]
        """ % (file_name, out_file)
        pipeline = pdal.Pipeline(json)
        count = pipeline.execute()
