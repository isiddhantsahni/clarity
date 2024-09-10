#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-user=ssahni1@sheffield.ac.uk

module load Anaconda3/2019.07
module load CUDA/10.2.89-GCC-8.3.0
source activate clarityenv

python train_den.py
#python train_den_batch8.py
#python train_den_5.py
#python train_den_cec2.py
#python train_den_cec2_epoch200.py
