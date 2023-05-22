import argparse
import pdal
#from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from functools import partial

def worker(output_dir: Path, file: str):
    input_file = file.name
    out_file = input_file.replace("_height_filtered",'')
    out_file = out_file.split(".")[0]

    json = """
    [
        "%s",   
        {
            "type":"writers.gdal",
            "filename":"%s/%s_max.tif",
            "output_type":"max",
            "gdaldriver":"GTiff",
            "resolution":1.0
        }
    ]
    """ % (str(file), str(output_dir), out_file)
    #print(json)
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    return


def rasterize(dir: Path, output_dir: Path, MAX_WORKERS: int):

    onlyfiles = [f for f in sorted(dir.glob('*.laz')) if f.is_file()]
    func = partial(worker, output_dir)

    with Pool(MAX_WORKERS) as p:
        p.map(func, onlyfiles)
    p.join()
    p.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rasterize the laz files.')
    parser.add_argument('folder', type=str, help='Folder with files to convert')

    args = parser.parse_args()
    dir = args.folder

    height_removed_dir_name = "LazFilesWithHeightRemoved"
    height_removed_dir = Path(f"{dir}/{height_removed_dir_name}")
    height_removed_dir.mkdir(exist_ok=True)

    raster_image_dir_name  = "ImagesGroundRemoved"
    raster_image_dir = Path(f"{dir}/{raster_image_dir_name}")
    raster_image_dir.mkdir(exist_ok=True)

    rasterize(height_removed_dir, raster_image_dir, 1)
