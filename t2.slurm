#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=0-10:00:00
#SBATCH --mem-per-cpu=16384
#SBATCH --job-name="Kaggle_t2"
#SBATCH --output outputs/Kaggle_t2_%A.out
#SBATCH --error outputs/Kaggle_t2_%A.err

module load python-gcc/2.7.10
#python get_all_features.py
python predict.py