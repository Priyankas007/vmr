#!/bin/bash

#BATCH --job-name=train_single_classifier         # Job name
#SBATCH --time=10:00:00
#SBATCH -p syyeung
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shrestp@stanford.edu # Email address for notifications
#SBATCH --output=logs/train_single_classifier.out
#SBATCH --error=logs/train_single_classifier.err

# Activate your environment
conda activate cs286
module load devel opencv/4.5.5
module load system libtiff/4.0.8
module load devel py-pytorch/2.0.0_py39
module load viz py-matplotlib/3.7.1_py39
module load devel py-torchvision/0.15.1_py39


python3 train.py --model_name="dinov2_vits14" --experiment_type="LoRa" --epochs=1 --batch_size=32 --lr=5e-5
