#!/bin/bash
#BATCH --job-name=minkowskiTrain
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=64000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=4-00:00:00

echo "Training Started"
module load cuda/11.3

eval "$(conda shell.bash hook)"
conda activate powerlines

python -c "import torch; print('Cuda is available: ', torch.cuda.is_available())"
python -c "import torch; print('Torch version: ',torch.__version__)"
python -c "import torch; print('Torch version cuda: ', torch.version.cuda)"
pip list


