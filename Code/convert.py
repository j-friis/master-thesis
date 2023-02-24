import argparse
import pdal
parser = argparse.ArgumentParser(description='Convert Laz to tif.')
parser.add_argument('file', type=str, help='file to convert')

args = parser.parse_args()
file_name = args.file
out_file = file_name.split(".")[0]

json = """
[
    "%s",
    {
        "type":"writers.gdal",
        "filename":"%s_min.tif",
        "output_type":"min",
        "gdaldriver":"GTiff",
        "resolution":0.08
    },
    {
        "type":"writers.gdal",
        "filename":"%s_max.tif",
        "output_type":"max",
        "gdaldriver":"GTiff",
        "resolution":0.08
    },
    {
        "type":"writers.gdal",
        "filename":"%s_idw.tif",
        "output_type":"idw",
        "gdaldriver":"GTiff",
        "resolution":0.08
    }
]
""" % (file_name, out_file, out_file, out_file)


pipeline = pdal.Pipeline(json)
count = pipeline.execute()
