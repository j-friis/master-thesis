
CUDA_VISIABLE_DEVICES=1 HYDRA_FULL_ERROR=1 python -u train.py task=segmentation models=segmentation/minkowski  \
model_name=SEUNet50 data=segmentation/den_pl_cnn_20cm_ones \
training.wandb.log=True training.wandb.project=NewDenmark-Seg-modified training.wandb.name=CNN_SEUNet50_20cm_ones_500 \
training=denmark/minkowski training.batch_size=32
