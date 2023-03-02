import argparse
import pdal
from os import listdir
from os.path import isfile, join
#from tqdm import tqdm
from multiprocessing import Pool

from PDAL_CONSTANTS import MAX_WORKERS

def worker(in_file: str):
    out_file = in_file.split(".")[0]
    out_file = out_file.replace("_height_filtered",'')


    json = """
    [
        "%s",   
        {
            "type":"writers.gdal",
            "filename":"%s_max.tif",
            "output_type":"max",
            "gdaldriver":"GTiff",
            "resolution":0.08
        }
    ]
    """ % (in_file, out_file)
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()

    file_name = in_file.split('/')[-1]
    file_name = file_name.replace("_height_filtered",'')

    return f"Done with {file_name}"



def rasterize(dir: str):
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f)) and "_height_filtered" in f and "_max" not in f]
    print(onlyfiles)
    onlyfiles = [join(dir, f) for f in onlyfiles]


    with Pool(MAX_WORKERS) as p:
        # results = tqdm(
        #     p.imap_unordered(worker, onlyfiles),
        #     total=len(onlyfiles),
        # )  # 'total' is redundant here but can be useful
        # when the size of the iterable is unobvious
        p.map(worker, onlyfiles)
        # for result in results:
        #     print(result)


    # for file in tqdm(onlyfiles):
    #     file_name = file
    #     file_name = join(dir, file_name)
    #     out_file = file_name.split(".")[0]
    #     out_file = join(dir, out_file)
        
    #     # json = """
    #     # [
    #     #     "%s",
    #     #     {
    #     #         "type":"writers.gdal",
    #     #         "filename":"%s_min.tif",
    #     #         "output_type":"min",
    #     #         "gdaldriver":"GTiff",
    #     #         "resolution":0.08
    #     #     },
    #     #     {
    #     #         "type":"writers.gdal",
    #     #         "filename":"%s_max.tif",
    #     #         "output_type":"max",
    #     #         "gdaldriver":"GTiff",
    #     #         "resolution":0.08
    #     #     },
    #     #     {
    #     #         "type":"writers.gdal",
    #     #         "filename":"%s_idw.tif",
    #     #         "output_type":"idw",
    #     #         "gdaldriver":"GTiff",
    #     #         "resolution":0.08
    #     #     }
    #     # ]
    #     # """ % (file_name, out_file, out_file, out_file)

    #     json = """
    #     [
    #         "%s",
    #         {
    #             "type":"writers.gdal",
    #             "filename":"%s_max.tif",
    #             "output_type":"max",
    #             "gdaldriver":"GTiff",
    #             "resolution":0.08
    #         }
    #     ]
    #     """ % (file_name, out_file)
    #     pipeline = pdal.Pipeline(json)
    #     count = pipeline.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rasterize the laz files.')
    parser.add_argument('folder', type=str, help='Folder with files to convert')

    args = parser.parse_args()
    dir = args.folder

    rasterize(dir)
