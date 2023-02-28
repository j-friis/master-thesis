import argparse
import pdal
import os
import laspy
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

def filter_height(dir: str):

    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f)) and "_hag_delaunay" in f and "max" not in f]
    print(onlyfiles)

    for file in tqdm(onlyfiles):
        file_name = file
        file_name = join(dir, file_name)

        out_filename = file_name.split(".")[0]
        out_filename = out_filename.replace("_hag_delaunay",'')
        out_filename = f"{out_filename}_height_filtered.laz"
        out_file = join(dir, out_filename)
        
        pdal_data = laspy.read(file_name, laz_backend=laspy.compression.LazBackend.LazrsParallel)
        pdal_data = pdal_data[pdal_data.HeightAboveGround>3 ]
        pdal_data.write(out_file, do_compress =True, laz_backend=laspy.compression.LazBackend.LazrsParallel)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter the height in laz files.')
    parser.add_argument('folder', type=str, help='Folder with files to filter')

    args = parser.parse_args()
    dir = args.folder
    filter_height(dir)