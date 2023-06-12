
CUDA_VISIABLE_DEVICES=1 HYDRA_FULL_ERROR=1 python -u train.py task=segmentation models=segmentation/minkowski  \
model_name=SEUNet18 data=segmentation/den_pl_cnn_10cm_ones \
training.wandb.log=True training.wandb.project=NewDenmark-Seg-modified training.wandb.name=Retrain_CNN_SEUNet18_10cm_ones \
training=denmark/minkowski training.batch_size=128 lr_scheduler=exponential lr_scheduler.params.gamma=0.998