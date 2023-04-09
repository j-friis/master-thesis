#!/bin/bash
n_cpu=$(nproc --all)
echo ${n_cpu}
n_cpu=$((n_cpu-1))

echo "Runing pdal pipeline"
eval "$(conda shell.bash hook)"
echo "$1"
echo "$2"
echo "$3"
eval pwd

conda activate pdal
python torch_points3d/core/data_transform/pdal/run_pipeline.py $1 $n_cpu 3
# conda activate powerlines
# python eval/torch_pipeline.py $2 $3