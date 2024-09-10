#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --mail-user=ssahni1@sheffield.ac.uk

module load Anaconda3/2019.07
module load CUDA/10.2.89-GCC-8.3.0
source activate clarityenv

#python infer_den_batch8.py
#python infer_den_lr_5.py
#python infer_den_cec2.py
python infer_den_lr.py
