import argparse
import laspy
from multiprocessing import Pool
from pathlib import Path
from functools import partial

def worker(output_dir: Path, file: str):
    file_name = file.name
    print(file_name)
    out_filename = file_name.split(".")[0]
    out_filename = out_filename.replace("_hag_delaunay",'')
    out_filename = out_filename.replace("_hag_nn",'')
    out_filename = f"{out_filename}_height_filtered.laz"
    out_file = output_dir.joinpath(out_filename)
    # print("-------------------------------------------")
    # print(f"{file = }, {file_name = }, {out_filename = }, {output_dir = }, {out_file = }")
    # print("-------------------------------------------")
    pdal_data = laspy.read(file, laz_backend=laspy.compression.LazBackend.LazrsParallel)
    pdal_data = pdal_data[pdal_data.HeightAboveGround>3 ]
    pdal_data.write(out_file, do_compress =True, laz_backend=laspy.compression.LazBackend.LazrsParallel)
    return
    file_name = out_filename.replace("_height_filtered","")

    return f"Done with {file_name}"



def filter_height(dir: Path, output_dir: Path, MAX_WORKERS: int):

    onlyfiles = [f for f in sorted(dir.glob('*.laz')) if f.is_file()]
    #onlyfiles = [f for f in sorted(dir.glob('*.laz')) if f.is_file()]
    #print(onlyfiles)
    #print(output_dir)

    func = partial(worker, output_dir)

    with Pool(MAX_WORKERS) as p:
        # results = tqdm(
        #     p.imap_unordered(worker, onlyfiles),
        #     total=len(onlyfiles),
        # )  # 'total' is redundant here but can be useful
        # when the size of the iterable is unobvious
        p.map(func, onlyfiles)
        # for result in results:
        #     print(result)

    # for file in tqdm(onlyfiles):
    #     file_name = file
    #     file_name = join(dir, file_name)

    #     out_filename = file_name.split(".")[0]
    #     out_filename = out_filename.replace("_hag_delaunay",'')
    #     out_filename = f"{out_filename}_height_filtered.laz"
    #     out_file = join(dir, out_filename)
        
    #     pdal_data = laspy.read(file_name, laz_backend=laspy.compression.LazBackend.LazrsParallel)
    #     pdal_data = pdal_data[pdal_data.HeightAboveGround>3 ]
    #     pdal_data.write(out_file, do_compress =True, laz_backend=laspy.compression.LazBackend.LazrsParallel)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter the height in laz files.')
    parser.add_argument('folder', type=str, help='Folder with files to filter')

    args = parser.parse_args()
    dir = args.folder

    height_dir_name = "LazFilesWithHeightParam"
    height_dir = Path(f"{dir}/{height_dir_name}")
    height_dir.mkdir(exist_ok=True)


    height_removed_dir_name = "LazFilesWithHeightRemoved"
    height_removed_dir = Path(f"{dir}/{height_removed_dir_name}")
    height_removed_dir.mkdir(exist_ok=True)

    dir = Path(f"{dir}")
    filter_height(height_dir, height_removed_dir, 1)