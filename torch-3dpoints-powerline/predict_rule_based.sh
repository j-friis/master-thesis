#!/bin/bash
echo "Runing pdal pipeline"
eval "$(conda shell.bash hook)"

path_to_test_data="$3/denmark/raw/test"
echo "Runing pdal pipeline"
conda activate pdal
python torch_points3d/core/data_transform/pdal/run_pipeline.py $path_to_test_data $1 $2 1.0

echo "Predecting with 3D CNN"
conda activate msc
python eval_to_las.py $3 $4 $5 ""

