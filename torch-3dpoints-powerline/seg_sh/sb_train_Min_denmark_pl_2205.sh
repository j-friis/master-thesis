#CUDA_VISIABLE_DEVICES=1 HYDRA_FULL_ERROR=1 python -u train.py task=segmentation +models=segmentation/minkowski  \
#model_name=SEUNet18 data=segmentation/denmark_pl_2205 training.wandb.log=True training.wandb.project=NewDenmark-Seg-modified \
#training.wandb.name=minkowski_powerline_0.8_0.5_0.01_0522 training.batch_size=2

#CUDA_VISIABLE_DEVICES=1 HYDRA_FULL_ERROR=1 python -u train.py task=segmentation +models=segmentation/minkowski  \
#model_name=SEUNet18 data=segmentation/denmark_pl_2205_attention training.wandb.log=True training.wandb.project=NewDenmark-Seg-modified \
#training.wandb.name=minkowski_powerline_var_0.4_0.025_0.005_0522 training.batch_size=2

CUDA_VISIABLE_DEVICES=0 HYDRA_FULL_ERROR=1 python -u train.py task=segmentation +models=segmentation/minkowski  \
model_name=SEUNet18 data=segmentation/denmark_pl_2205 training.wandb.log=False training.wandb.project=NewDenmark-Seg-modified \
training.wandb.name=minkowski_powerline_var_0.4_0.025_0.05_0522 training.batch_size=2
