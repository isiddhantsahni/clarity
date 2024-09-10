#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --mail-user=ssahni1@sheffield.ac.uk

module load Anaconda3/2019.07
module load CUDA/10.2.89-GCC-8.3.0
source activate clarityenv

python train_v2.py

