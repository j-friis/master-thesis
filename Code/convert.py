import argparse
import pdal
parser = argparse.ArgumentParser(description='Convert Laz to tif.')
parser.add_argument('file', type=str, help='file to convert')

args = parser.parse_args()
file_name = args.file
out_file = file_name.split(".")[0] + ".tif"

json = """
[
    "%s",
    {
        "type":"writers.gdal",
        "filename":"min_%s",
        "output_type":"min",
        "gdaldriver":"GTiff",
        "resolution":10.0
    },
    {
        "type":"writers.gdal",
        "filename":"max_%s",
        "output_type":"max",
        "gdaldriver":"GTiff",
        "resolution":5.0
    },
    {
        "type":"writers.gdal",
        "filename":"idw_%s",
        "output_type":"idw",
        "gdaldriver":"GTiff",
        "resolution":3.0
    }
]
""" % (file_name, out_file, out_file, out_file)


pipeline = pdal.Pipeline(json)
count = pipeline.execute()
