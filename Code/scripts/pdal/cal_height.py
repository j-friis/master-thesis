import argparse
import pdal
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from multiprocessing import Pool


MAX_WORKERS = 3

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

def pipeline(in_file: str):
    print(in_file)
    file_name = in_file
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
    return f"Done with {file_name}"


def cal_height(dir: str):
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f)) 
                    and "_hag" not in f and ".tif" not in f and "height" not in f]
    print(onlyfiles)

    with Pool(MAX_WORKERS) as p:
        results = tqdm(
            p.imap_unordered(pipeline, onlyfiles),
            total=len(onlyfiles),
        )  # 'total' is redundant here but can be useful
        # when the size of the iterable is unobvious
        #p.map(pipeline, onlyfiles)
        for result in results:
            print(result)


    # for file in tqdm(onlyfiles):
    #     file_name = file
    #     file_name = join(dir, file_name)
    #     out_file = file_name.split(".")[0]
    #     out_file = join(dir, out_file)
    #     json = """
    #     [
    #         "%s",
    #         {
    #             "type":"filters.hag_nn"
    #         },
    #         {
    #             "type":"writers.las",
    #             "filename":"%s_hag_nn.laz",
    #             "extra_dims":"HeightAboveGround=float32",
    #             "compression":"laszip"
    #         }
    #     ]
    #     """ % (file_name, out_file)
    #     pipeline = pdal.Pipeline(json)
    #     count = pipeline.execute()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate the height from the ground.')
    parser.add_argument('folder', type=str, help='Folder with files to calculation')

    args = parser.parse_args()
    dir = args.folder
    cal_height(dir)