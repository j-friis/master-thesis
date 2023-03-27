import argparse
from functools import partial
import pdal
from tqdm import tqdm
from multiprocessing import Pool

from pathlib import Path

def worker(dir: Path, output_dir: Path, file: str):
    file_name = file.name
    print(file_name)
    out_file = file_name.split(".")[0]
    input_file = dir.joinpath(file)#"%s/%s_hag_nn.laz"  % (str(output_dir), out_file)

    json = """
    [
        "%s",
        {
            "type":"filters.hag_nn"
        },
        {
            "type":"writers.las",
            "filename":"%s/%s_hag_nn.laz",
            "extra_dims":"HeightAboveGround=float32",
            "compression":"laszip"
        }
    ]
    """ % (input_file, str(output_dir), out_file)

    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    return 
    file_name = in_file.split('/')[-1]
    print(f"Done with {file_name}")
    return f"Done with {file_name}"


def cal_height(dir: Path, output_dir: Path, MAX_WORKERS: int):

    onlyfiles = [f for f in sorted(dir.glob('*.laz')) if f.is_file()]
    #print(onlyfiles)
    #onlyfiles = [join(dir, f) for f in onlyfiles]

    func = partial(worker, dir, output_dir)

    with Pool(MAX_WORKERS) as p:
        p.map(func, onlyfiles)

        # results = tqdm(
        #     p.imap_unordered(worker, onlyfiles),
        #     total=len(onlyfiles),
        # )  # 'total' is redundant here but can be useful
        # when the size of the iterable is unobvious
        #p.map(worker, onlyfiles)
        # for result in results:
        #     print(result)

    # for file in tqdm(onlyfiles):
    # #for file in onlyfiles:
    #     file_name = file.name
    #     out_file = file_name.split(".")[0]
    #     input_file = dir.joinpath(file)#"%s/%s_hag_nn.laz"  % (str(output_dir), out_file)
    #     json = """
    #     [
    #         "%s",
    #         {
    #             "type":"filters.hag_nn"
    #         },
    #         {
    #             "type":"writers.las",
    #             "filename":"%s/%s_hag_nn.laz",
    #             "extra_dims":"HeightAboveGround=float32",
    #             "compression":"laszip"
    #         }
    #     ]
    #     """ % (input_file, str(output_dir), out_file)
    #     pipeline = pdal.Pipeline(json)
    #     count = pipeline.execute()

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
    cal_height(dir, height_dir, 3)