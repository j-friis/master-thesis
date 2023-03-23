import argparse
from filter_height import filter_height
from raster import rasterize
from cal_height import cal_height
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run entire pipeline.')
    parser.add_argument('folder', type=str, help='Folder with files to run pipeline on')
    parser.add_argument('max_workers', type=int, help='The number of process used in Pool')
    parser.add_argument('height_filter', type=int, help='The amount of meters above ground we remove')

    args = parser.parse_args()
    dir = args.folder
    MAX_WORKERS = args.max_workers
    height_filter = args.height_filter

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
    cal_height(dir, height_dir, MAX_WORKERS)
    laz_with_height_dir = f"{dir}/LazFilesWithHeightParam"
    print("-------------------- Filtering height --------------------")
    filter_height(height_dir, height_removed_dir, MAX_WORKERS, height_filter)
    print("-------------------- Rasterize files ---------------------")
    laz_with_height_removed_dir = f"{dir}/LazFilesWithHeightRemoved"
    rasterize(height_removed_dir, raster_image_dir, MAX_WORKERS)
