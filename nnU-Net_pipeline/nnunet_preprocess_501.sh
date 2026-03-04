#!/bin/bash
#
# SLURM Script for nnU-Net Planning and Preprocessing (OpenNeuro ds006248)
#
#SBATCH --job-name=nnunet_preprocess_501
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/home/openeuro_dataset/nnunet_preprocess_501_%j.out
#SBATCH --error=/home/openeuro_dataset/nnunet_preprocess_501_%j.err

echo "=== nnU-Net preprocessing job started ==="
echo "Hostname: $(hostname)"
echo "Start time: $(date)"

# --- ENVIRONMENT SETUP ---
module load cuda/12.2 2>/dev/null || true   # harmless if not needed/available
source ~/.bashrc
conda activate nnunet_new

# --- PROJECT ROOT (where you want nnU-Net folders to live) ---
PROJECT_ROOT="/home/fdv119/openeuro_dataset"

# --- NNUNET PATHS ---
export nnUNet_raw="$PROJECT_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_ROOT/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_ROOT/nnUNet_results"

echo "Project Root: $PROJECT_ROOT"
echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"

# --- DATASET SETTINGS ---
DATASET_ID=501
DATASET_FOLDER="Dataset501_Pituitary"

echo "Dataset ID: $DATASET_ID"
echo "Dataset folder: $nnUNet_raw/$DATASET_FOLDER"

# --- (Optional) Quick dataset sanity ---
echo "Training images count:"
ls -1 "$nnUNet_raw/$DATASET_FOLDER/imagesTr" | wc -l
echo "Training labels count:"
ls -1 "$nnUNet_raw/$DATASET_FOLDER/labelsTr" | wc -l

if [ -d "$nnUNet_raw/$DATASET_FOLDER/imagesTs" ]; then
  echo "Test images count:"
  ls -1 "$nnUNet_raw/$DATASET_FOLDER/imagesTs" | wc -l
fi

# --- EXECUTION ---
echo "Running: nnUNetv2_plan_and_preprocess -d $DATASET_ID -np $SLURM_CPUS_PER_T
ASK --verify_dataset_integrity"
nnUNetv2_plan_and_preprocess -d $DATASET_ID -np $SLURM_CPUS_PER_TASK --verify_da
taset_integrity

echo "End time: $(date)"
echo "=== Job finished ==="
