import argparse
import pdal
#from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from functools import partial

def worker(max_output_dir: Path, count_output_dir: Path, stdev_output_dir: Path, file: str):
    input_file = file.name
    out_file = input_file.replace("_height_filtered",'')
    out_file = out_file.split(".")[0]

    #input_file = file#"%s/%s_hag_nn.laz"  % (str(max_output_dir), out_file)
    # print("-------------------------------------------")
    # print(f"{file = }, {input_file = }, {max_output_dir = }, {out_file = }")

    # print("-------------------------------------------")
    # print("%s/%s_max.tif" % (str(max_output_dir), out_file))

    #print("-------------------------------------------")
    #print(f"{file = }, {str(max_output_dir) = }, {out_file = }")

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
    """ % (str(file), str(max_output_dir), out_file)
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()

    json = """
    [
        "%s",   
        {
            "type":"writers.gdal",
            "filename":"%s/%s_count.tif",
            "output_type":"count",
            "gdaldriver":"GTiff",
            "resolution":1.0
        }
    ]
    """ % (str(file), str(count_output_dir), out_file)
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()

    json = """
    [
        "%s",   
        {
            "type":"writers.gdal",
            "filename":"%s/%s_stdev.tif",
            "output_type":"stdev",
            "gdaldriver":"GTiff",
            "resolution":1.0
        }
    ]
    """ % (str(file), str(stdev_output_dir), out_file)
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    return

def rasterize(dir: Path, max_output_dir: Path, count_output_dir: Path, stdev_output_dir: Path, MAX_WORKERS: int):

    onlyfiles = [f for f in sorted(dir.glob('*.laz')) if f.is_file()]

    func = partial(worker, max_output_dir, count_output_dir, stdev_output_dir)

    with Pool(MAX_WORKERS) as p:
        p.map(func, onlyfiles)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rasterize the laz files.')
    parser.add_argument('folder', type=str, help='Folder with files to convert')

    args = parser.parse_args()
    dir = args.folder

    height_removed_dir_name = "LazFilesWithHeightRemoved"
    height_removed_dir = Path(f"{dir}/{height_removed_dir_name}")
    height_removed_dir.mkdir(exist_ok=True)

    max_image_dir_name  = "ImagesGroundRemoved"
    max_image_dir = Path(f"{dir}/{max_image_dir_name}")
    max_image_dir.mkdir(exist_ok=True)

    count_image_dir_name  = "ImagesGroundCountRemoved"
    count_image_dir = Path(f"{dir}/{count_image_dir_name}")
    count_image_dir.mkdir(exist_ok=True)

    stdev_image_dir_name  = "ImagesGroundStdevRemoved"
    stdev_image_dir = Path(f"{dir}/{stdev_image_dir_name}")
    stdev_image_dir.mkdir(exist_ok=True)

    rasterize(dir=height_removed_dir, max_output_dir=max_image_dir, count_output_dir=count_image_dir, stdev_output_dir=stdev_image_dir, MAX_WORKERS=1)
