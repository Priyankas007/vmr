#!/bin/bash

#SBATCH -p syyeung
#SBATCH -c 4
#SBATCH -G 1
#BATCH --job-name=train_classifier         # Job name
#SBATCH --output=logs/%x_%j.out           # Standard output log
#SBATCH --error=logs/%x_%j.err            # Standard error log
#SBATCH --time=4:00:00                   # Time limit (HH:MM:SS)
#SBATCH --mem=16G                         # Memory (RAM)
#SBATCH --mail-type=END,FAIL              # Notifications (email on job completion/failure)
#SBATCH --mail-user=shrestp@stanford.edu # Email address for notifications

# Activate your environment
conda activate cs286
module load devel opencv/4.5.5
module load system libtiff/4.0.8
module load devel py-pytorch/2.0.0_py39
module load viz py-matplotlib/3.7.1_py39
module load devel py-torchvision/0.15.1_py39


python3 train_classifier.py --model_name="dinov2_vits14" --experiment_type="finetune" --epochs=1 --batch_size=256 --lr=5e-5
