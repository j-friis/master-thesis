import argparse
import pdal
from tqdm import tqdm
from multiprocessing import Pool

from pathlib import Path

def worker(in_file: str):

    out_file = in_file.split(".")[0]

    json = """
    [
        "%s",
        {
            "type":"filters.hag_nn"
        },
        {
            "type":"writers.las",
            "filename":"%s_hag_nn.laz",
            "extra_dims":"HeightAboveGround=float32",
            "compression":"laszip"
        }
    ]
    """ % (in_file, out_file)
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()

    file_name = in_file.split('/')[-1]
    print(f"Done with {file_name}")
    return f"Done with {file_name}"


def cal_height(dir: Path, output_dir: Path, MAX_WORKERS: int):

    onlyfiles = [f for f in sorted(dir.glob('*.laz')) if f.is_file()]

    with Pool(MAX_WORKERS) as p:
        p.map(worker, onlyfiles)
        
    p.close()
    p.join()

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate the height from the ground.')
    parser.add_argument('folder', type=str, help='Folder with files to calculation')

    args = parser.parse_args()
    dir = args.folder

    height_dir_name = "LazFilesWithHeightParam"

    height_dir = Path(f"{dir}/{height_dir_name}")
    height_dir.mkdir(exist_ok=True)

    dir = Path(f"{dir}")
    cal_height(dir, height_dir, 1)