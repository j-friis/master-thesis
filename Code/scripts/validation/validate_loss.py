import laspy
import argparse
from os import listdir
from os.path import isfile, join
import numpy as np
parser = argparse.ArgumentParser(description='Compare number of wire conductor points.')
parser.add_argument('nTfile', type=str, help='non transformed file')
parser.add_argument('tFile', type=str, help='transformed file')

args = parser.parse_args()
nTfile = args.nTfile
tFile = args.tFile


if __name__ == "__main__":
    non_transformed_data = laspy.read(nTfile, laz_backend=laspy.compression.LazBackend.LazrsParallel)
    transformed_data = laspy.read(tFile, laz_backend=laspy.compression.LazBackend.LazrsParallel)

    non_powerlines = laspy.create(point_format=non_transformed_data.header.point_format, file_version=non_transformed_data.header.version)
    non_powerlines.points = non_transformed_data.points[non_transformed_data.classification == 14]

    trans_powerlines = laspy.create(point_format=transformed_data.header.point_format, file_version=transformed_data.header.version)
    trans_powerlines.points = transformed_data.points[transformed_data.classification == 14]

    print(f"There are {np.sum(non_powerlines.classification == 14)} wire conductor points in the non transformed data")
    print(f"There are {np.sum(trans_powerlines.classification == 14)} wire conductor points in the transformed data")
