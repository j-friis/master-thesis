import argparse
import pdal
#from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from functools import partial
from tqdm import tqdm

def worker(output_dir: Path, file: str):
    input_file = file.name
    out_file = input_file.replace("_height_filtered",'')
    out_file = out_file.split(".")[0]

    #input_file = file#"%s/%s_hag_nn.laz"  % (str(output_dir), out_file)
    # print("-------------------------------------------")
    # print(f"{file = }, {input_file = }, {output_dir = }, {out_file = }")

    # print("-------------------------------------------")
    # print("%s/%s_max.tif" % (str(output_dir), out_file))

    #print("-------------------------------------------")
    #print(f"{file = }, {str(output_dir) = }, {out_file = }")

    json = """
    [
        "%s",   
        {
            "type":"writers.gdal",
            "filename":"%s/%s_max.tif",
            "output_type":"max",
            "gdaldriver":"GTiff",
            "resolution":0.01
        }
    ]
    """ % (str(file), str(output_dir), out_file)
    #print(json)
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    return




def rasterize(dir: Path, output_dir: Path, MAX_WORKERS: int):

    onlyfiles = [f for f in sorted(dir.glob('*.laz')) if f.is_file()]

    #print(onlyfiles)

    func = partial(worker, output_dir)


    # with Pool(MAX_WORKERS) as p:
    #     # results = tqdm(
    #     #     p.imap_unordered(worker, onlyfiles),
    #     #     total=len(onlyfiles),
    #     # )  # 'total' is redundant here but can be useful
    #     # when the size of the iterable is unobvious
    #     p.map(func, onlyfiles)
    #     #p.map(worker, onlyfiles)
    #     # for result in results:
    #     #     print(result)


    for file in tqdm(onlyfiles):
    # for file in onlyfiles:
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
                "resolution":0.1
            }
        ]
        """ % (str(file), str(output_dir), out_file)
        pipeline = pdal.Pipeline(json)
        count = pipeline.execute()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rasterize the laz files.')
    parser.add_argument('folder', type=str, help='Folder with files to convert')

    args = parser.parse_args()
    dir = args.folder

    height_removed_dir_name = "LazFilesWithHeightRemoved"
    height_removed_dir = Path(f"{dir}/{height_removed_dir_name}")
    height_removed_dir.mkdir(exist_ok=True)

    raster_image_dir_name  = "ImagesGroundRemovedLarge"
    raster_image_dir = Path(f"{dir}/{raster_image_dir_name}")
    raster_image_dir.mkdir(exist_ok=True)

    rasterize(height_removed_dir, raster_image_dir, 1)
