#!/bin/bash
# function parse_yaml {
#    local prefix=$2
#    local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
#    sed -ne "s|^\($s\):|\1|" \
#         -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
#         -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
#    awk -F$fs '{
#       indent = length($1)/2;
#       vname[indent] = $2;
#       for (i in vname) {if (i > indent) {delete vname[i]}}
#       if (length($3) > 0) {
#          vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
#          printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
#       }
#    }'
# }
#echo "$PWD"
#eval $(parse_yaml /home/jf/Documents/msc/torch-3dpoints-powerline/conf/data/segmentation/denmark_pl_hough_small.yaml)
#echo $pdal_workers
n_cpu=$(nproc --all)
echo "${n_cpu}"
echo Runing pdal pipeline
eval "$(conda shell.bash hook)"
#source ~/miniconda3/etc/profile.d/conda.sh
conda activate pdal
python torch_points3d/core/data_transform/pdal/run_pipeline.py /home/jf/msc_data/denmark/raw/train $n_cpu 3
python torch_points3d/core/data_transform/pdal/run_pipeline.py /home/jf/msc_data/denmark/raw/test $n_cpu 3
python torch_points3d/core/data_transform/pdal/run_pipeline.py /home/jf/msc_data/denmark/raw/val $n_cpu 3
conda activate msc
bash seg_sh/sb_train_denmank_pl_hough.sh
