import argparse
import os 
from os import listdir
from os.path import isfile, join
parser = argparse.ArgumentParser(description='Removes files with height calculations.')
parser.add_argument('folder', type=str, help='folder to clean up files')

args = parser.parse_args()
dir = args.folder


if __name__ == "__main__":

    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f)) and "_hag_delaunay" in f]
    print(onlyfiles)

    for file in onlyfiles:
        file_name = file
        file_name = join(dir, file_name)
        os.remove(file_name)
