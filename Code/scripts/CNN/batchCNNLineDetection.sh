#!/bin/bash
#SBATCH --job-name=nxw500cnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=1-00:00:00


echo $CUDA_VISIBLE_DEVICES

python -c "import torch; print('Cuda is available: ', torch.cuda.is_available())"
python -c "import torch; print('Torch version: ',torch.__version__)"
python -c "import torch; print('Torch version cuda: ', torch.version.cuda)"


echo "CNN Started"
echo "Training started by nxw500"
python CNN-LineDetection.py
