import argparse
import pdal
import os
import laspy
from os import listdir
from os.path import isfile, join
#from tqdm import tqdm
from multiprocessing import Pool

from PDAL_CONSTANTS import MAX_WORKERS

def worker(in_file: str):

    out_filename = in_file.split(".")[0]

    out_filename = out_filename.replace("_hag_delaunay",'')
    out_filename = out_filename.replace("_hag_nn",'')
    out_filename = f"{out_filename}_height_filtered.laz"
    
    pdal_data = laspy.read(in_file, laz_backend=laspy.compression.LazBackend.LazrsParallel)
    pdal_data = pdal_data[pdal_data.HeightAboveGround>3 ]
    pdal_data.write(out_filename, do_compress =True, laz_backend=laspy.compression.LazBackend.LazrsParallel)

    file_name = in_file.split('/')[-1]
    file_name = file_name.split("_height_filtered",1)[0]

    return f"Done with {file_name}"



def filter_height(dir: str):

    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f)) and "_hag" in f and ".tif" not in f]
    print(onlyfiles)
    onlyfiles = [join(dir, f) for f in onlyfiles]


    with Pool(MAX_WORKERS) as p:
        # results = tqdm(
        #     p.imap_unordered(worker, onlyfiles),
        #     total=len(onlyfiles),
        # )  # 'total' is redundant here but can be useful
        # when the size of the iterable is unobvious
        p.map(worker, onlyfiles)
        # for result in results:
        #     print(result)

    # for file in tqdm(onlyfiles):
    #     file_name = file
    #     file_name = join(dir, file_name)

    #     out_filename = file_name.split(".")[0]
    #     out_filename = out_filename.replace("_hag_delaunay",'')
    #     out_filename = f"{out_filename}_height_filtered.laz"
    #     out_file = join(dir, out_filename)
        
    #     pdal_data = laspy.read(file_name, laz_backend=laspy.compression.LazBackend.LazrsParallel)
    #     pdal_data = pdal_data[pdal_data.HeightAboveGround>3 ]
    #     pdal_data.write(out_file, do_compress =True, laz_backend=laspy.compression.LazBackend.LazrsParallel)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter the height in laz files.')
    parser.add_argument('folder', type=str, help='Folder with files to filter')

    args = parser.parse_args()
    dir = args.folder
    filter_height(dir)