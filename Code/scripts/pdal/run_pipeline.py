import argparse
from filter_height import filter_height
from raster import rasterize
from cal_height import cal_height

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run entire pipeline.')
    parser.add_argument('folder', type=str, help='Folder with files to run pipeline on')

    args = parser.parse_args()
    dir = args.folder
    print("------------------- Calculating height -------------------")
    cal_height(dir)
    print("-------------------- Filtering height --------------------")
    filter_height(dir)
    print("-------------------- Rasterize files ---------------------")
    rasterize(dir)
