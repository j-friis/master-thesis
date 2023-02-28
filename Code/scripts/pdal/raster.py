import argparse
import pdal
from os import listdir
from os.path import isfile, join
from tqdm import tqdm


def rasterize(dir: str):
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f)) and "_height_filtered" in f]
    print(onlyfiles)

    for file in tqdm(onlyfiles):
        file_name = file
        file_name = join(dir, file_name)
        out_file = file_name.split(".")[0]
        out_file = join(dir, out_file)
        
        # json = """
        # [
        #     "%s",
        #     {
        #         "type":"writers.gdal",
        #         "filename":"%s_min.tif",
        #         "output_type":"min",
        #         "gdaldriver":"GTiff",
        #         "resolution":0.08
        #     },
        #     {
        #         "type":"writers.gdal",
        #         "filename":"%s_max.tif",
        #         "output_type":"max",
        #         "gdaldriver":"GTiff",
        #         "resolution":0.08
        #     },
        #     {
        #         "type":"writers.gdal",
        #         "filename":"%s_idw.tif",
        #         "output_type":"idw",
        #         "gdaldriver":"GTiff",
        #         "resolution":0.08
        #     }
        # ]
        # """ % (file_name, out_file, out_file, out_file)

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
        """ % (file_name, out_file)
        pipeline = pdal.Pipeline(json)
        count = pipeline.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rasterize the laz files.')
    parser.add_argument('folder', type=str, help='Folder with files to convert')

    args = parser.parse_args()
    dir = args.folder

    rasterize(dir)
