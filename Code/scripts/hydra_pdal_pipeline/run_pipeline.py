import argparse
from filter_height import filter_height
from raster import rasterize
from cal_height import cal_height
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run entire pipeline.')
    parser.add_argument('folder', type=str, help='Folder with files to run pipeline on')

    args = parser.parse_args()
    dir = args.folder

    height_dir_name = "LazFilesWithHeightParam"
    height_removed_dir_name = "LazFilesWithHeightRemoved"
    raster_image_dir_name  = "ImagesGroundRemoved"

    height_dir = Path(f"{dir}/{height_dir_name}")
    height_removed_dir = Path(f"{dir}/{height_removed_dir_name}")
    raster_image_dir = Path(f"{dir}/{raster_image_dir_name}")
    dir = Path(f"{dir}")

    
    raster_image_dir.mkdir(exist_ok=True)
    height_dir.mkdir(exist_ok=True)
    height_removed_dir.mkdir(exist_ok=True)

    print("------------------- Calculating height -------------------")
    cal_height(dir, height_dir, 1)
    print("-------------------- Filtering height --------------------")
    filter_height(height_dir, height_removed_dir, 4)
    print("-------------------- Rasterize files ---------------------")
    rasterize(height_removed_dir, raster_image_dir, 2)
