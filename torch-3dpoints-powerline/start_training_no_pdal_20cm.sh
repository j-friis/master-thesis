#!/bin/bash
#BATCH --job-name=minkowskiTrain
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=64000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=2-00:00:00

echo "Training Started"
module load cuda/11.3

eval "$(conda shell.bash hook)"
conda activate powerlines
bash seg_sh/sb_train_denmank_pl_hough_20cm.sh
python -c "import MinkowskiEngine as ME; ME.print_diagnostics()"
python -c "import torch; print('Cuda is available: ', torch.cuda.is_available())"
python -c "import torch; print('Torch version: ',torch.__version__)"
python -c "import torch; print('Torch version cuda: ', torch.version.cuda)"
python -c "import torch; torch.cuda.device_count()"
python -c "import torch; torch.cuda.current_device()"
python -c "import torch; torch.cuda.device(0)"
python -c "import torch; torch.cuda.get_device_name(0)"
nvidia-smi -L
