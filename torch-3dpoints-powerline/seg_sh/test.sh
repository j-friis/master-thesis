
CUDA_VISIABLE_DEVICES=1 HYDRA_FULL_ERROR=1 python -u train.py task=segmentation models=segmentation/minkowski  \
model_name=SEUNet18 data=segmentation/denmark_pl_sample training.wandb.log=True training.wandb.project=NewDenmark-Seg-modified \
training.wandb.name=minkowski_powerline_var_0.1_0.25_0.1_0522 training.batch_size=2