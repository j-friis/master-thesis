#!/bin/bash
#BATCH --job-name=minkowskiTrain
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=64000M
#SBATCH -p gpu --gres=gpu:a40:1
#SBATCH --time=2-00:00:00

echo "Training Started"
module load cuda/11.3

eval "$(conda shell.bash hook)"
conda activate powerlines
bash seg_sh/sb_train_denmank_pl_hough_50cm.sh
python -c "import MinkowskiEngine as ME; ME.print_diagnostics()"



