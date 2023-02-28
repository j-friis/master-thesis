import argparse
from filter_height import filter_height
from raster import rasterize
from cal_height import cal_height

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Laz to tif.')
    parser.add_argument('folder', type=str, help='folder to convert files')

    args = parser.parse_args()
    dir = args.folder
    cal_height(dir)
    filter_height(dir)
    rasterize(dir)
