#!/usr/bin/bash
#SBATCH --job-name=train_classifier
#SBATCH --time=10:00:00
#SBATCH -p syyeung
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=16GB
#SBATCH --array=1-8                 # Adjusted range to match total configurations
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shrestp@stanford.edu
#SBATCH --output=logs/train_classifier_%A_%a.out
#SBATCH --error=logs/train_classifier_%A_%a.err

# Load required modules and activate environment
# conda activate cs286
module load devel opencv/4.5.5
module load system libtiff/4.0.8
module load devel py-pytorch/2.0.0_py39
module load viz py-matplotlib/3.7.1_py39
module load devel py-torchvision/0.15.1_py39
module load math py-wandb/0.18.7_py312

export WANDB_API_KEY="eafabcd73d8954b61717e959bcc768d841ca2898"

# Configurations for the jobs
MODELS=("dinov2_vits14" "dinov2_vitb14")
EXPERIMENTS=("finetune" "scratch" "linearProbe" "LoRa")
EPOCHS=20
BATCH_SIZES=32
LEARNING_RATES=5e-5

# Calculate total configurations
TOTAL_CONFIGURATIONS=$((${#MODELS[@]} * ${#EXPERIMENTS[@]}))

# Ensure SLURM_ARRAY_TASK_ID is valid
if [ $SLURM_ARRAY_TASK_ID -gt $TOTAL_CONFIGURATIONS ]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total configurations ($TOTAL_CONFIGURATIONS)."
    exit 1
fi

# Determine configuration indices
MODEL_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) / ${#EXPERIMENTS[@]} ))
EXPERIMENT_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) % ${#EXPERIMENTS[@]} ))

# Retrieve corresponding values
MODEL=${MODELS[$MODEL_INDEX]}
EXPERIMENT=${EXPERIMENTS[$EXPERIMENT_INDEX]}
EPOCH=${EPOCHS[0]}               # Single value reused for all configurations
BATCH_SIZE=${BATCH_SIZES[0]}     # Single value reused for all configurations
LEARNING_RATE=${LEARNING_RATES[0]} # Single value reused for all configurations

# Log the chosen configuration
echo "Running task $SLURM_ARRAY_TASK_ID with configuration:"
echo "Model: $MODEL"
echo "Experiment: $EXPERIMENT"
echo "Epochs: $EPOCH"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"

# Run the training script with the selected configuration
python3 train.py --model_name="$MODEL" --experiment_type="$EXPERIMENT" --epochs="$EPOCH" --batch_size="$BATCH_SIZE" --lr="$LEARNING_RATE"

if [ $? -eq 0 ]; then
    echo "SUCCESS: Training completed for task $SLURM_ARRAY_TASK_ID."
else
    echo "ERROR: Training failed for task $SLURM_ARRAY_TASK_ID."
    exit 1
fi
