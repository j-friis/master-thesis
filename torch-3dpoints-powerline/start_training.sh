#!/bin/bash
#BATCH --job-name=minkowskiTrain
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=64000M
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=4-00:00:00

echo "Training Started"
module load cuda/11.3

bash seg_sh/sb_train_denmank_pl_hough_2step.sh



