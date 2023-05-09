
CUDA_VISIABLE_DEVICES=1 HYDRA_FULL_ERROR=1 python -u train.py task=segmentation models=segmentation/minkowski  \
model_name=ResUNet50_ data=segmentation/den_pl_hough_small_20cm_ones training.wandb.log=True training.wandb.project=NewDenmark-Seg-modified \
training.wandb.name=minkowski_powerline_segmentation_ResUNet50_20cm_ones training.batch_size=32

