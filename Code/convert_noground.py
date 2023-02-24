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
        "type":"filters.hag_delaunay"
    },
    {
        "type":"writers.las",
        "filename":"%s_hag_delaunay.las",
        "extra_dims":"HeightAboveGround=float32",
        "compression":"laszip"
    }
]
""" % (file_name, out_file)
pipeline = pdal.Pipeline(json)
count = pipeline.execute()

