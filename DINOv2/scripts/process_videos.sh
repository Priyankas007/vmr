#!/bin/bash

#BATCH --job-name=process_videos         # Job name
#SBATCH --output=logs/%x_%j.out           # Standard output log
#SBATCH --error=logs/%x_%j.err            # Standard error log
#SBATCH --time=04:00:00                   # Time limit (HH:MM:SS)
#SBATCH --cpus-per-task=4                 # Number of CPUs
#SBATCH --mem=16G                         # Memory (RAM)
#SBATCH --partition=normal              # Partition name
#SBATCH --mail-type=END,FAIL              # Notifications (email on job completion/failure)
#SBATCH --mail-user=shrestp@stanford.edu # Email address for notifications

# Activate your environment
#source activate vmr_environment
module load python/3.9
module load system ffmpeg
module load math opencv/4.10.0
module load devel py-pandas

# Define Python script to execute
echo "runing script"
python3 segment_videos.py

echo "done running script"
