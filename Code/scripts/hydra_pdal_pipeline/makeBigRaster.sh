#!/bin/bash
#BATCH --job-name=RasterBig
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=64000M
#SBATCH --time=1-00:00:00

python raster.py ~/data/denmark/raw/train 4 3
python raster.py ~/data/denmark/raw/test 4 3
python raster.py ~/data/denmark/raw/val 4 3
