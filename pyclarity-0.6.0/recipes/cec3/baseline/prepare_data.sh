#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-user=ssahni1@sheffield.ac.uk

module load Anaconda3/2019.07
module load CUDA/10.2.89-GCC-8.3.0
source activate clarityenv

python prepare_data_cec2.py
