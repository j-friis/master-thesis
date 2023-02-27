import argparse
import os 
from os import listdir
from os.path import isfile, join
parser = argparse.ArgumentParser(description='Removes raster files.')
parser.add_argument('folder', type=str, help='folder to clean up files')

args = parser.parse_args()
dir = args.folder


if __name__ == "__main__":

    max_files = [f for f in listdir(dir) if isfile(join(dir, f)) and "max" in f]
    print(max_files)
    min_files = [f for f in listdir(dir) if isfile(join(dir, f)) and "min" in f]
    print(min_files)
    idw_files = [f for f in listdir(dir) if isfile(join(dir, f)) and "idw" in f]
    print(idw_files)

    for file in max_files:
        file_name = file
        file_name = join(dir, file_name)
        os.remove(file_name)

    for file in min_files:
        file_name = file
        file_name = join(dir, file_name)
        os.remove(file_name)

    for file in idw_files:
            file_name = file
            file_name = join(dir, file_name)
            os.remove(file_name)

