#!/bin/bash
#
# SLURM Script for nnU-Net Training (OpenNeuro ds006248) - 3D Full Resolution
#

# --- SLURM DIRECTIVES (GPU Job) ---
#SBATCH --job-name=nnunet_train_501_3dfullres
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/openeuro_dataset/nnunet_train_501_%j.out
#SBATCH --error=/home/openeuro_dataset/nnunet_train_501_%j.err

echo "=== nnU-Net TRAINING JOB STARTED ==="
echo "Hostname: $(hostname)"
echo "Start time: $(date)"

# --- ENVIRONMENT SETUP ---
module load cuda/12.2 2>/dev/null || true
source ~/.bashrc
conda activate nnunet_new

# --- PROJECT ROOT (OpenNeuro ds006248 project root) ---
PROJECT_ROOT="/home/fdv119/openeuro_dataset"

# --- NNUNET PATHS (lowercase used by nnU-Net v2) ---
export nnUNet_raw="$PROJECT_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_ROOT/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_ROOT/nnUNet_results"

# --- also set uppercase (harmless, future-proof) ---
export NNUNET_RAW_DATA="$nnUNet_raw"
export NNUNET_PREPROCESSED="$nnUNet_preprocessed"
export NNUNET_RESULTS="$nnUNet_results"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATASET_ID=501
DATASET_NAME="Dataset501_Pituitary"

echo "Project root: $PROJECT_ROOT"
echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"
echo "Dataset: $DATASET_ID ($DATASET_NAME)"

# --- QUICK SANITY CHECK ---
echo "Training images:"
ls -1 "$nnUNet_raw/$DATASET_NAME/imagesTr" | wc -l
echo "Training labels:"
ls -1 "$nnUNet_raw/$DATASET_NAME/labelsTr" | wc -l

echo "Preprocessed dataset contents:"
ls "$nnUNet_preprocessed/$DATASET_NAME" || { echo "❌ Preprocessing missing: $nnUN
et_preprocessed/$DATASET_NAME"; exit 1; }

# --- TRAINING ---
echo "Running: nnUNetv2_train $DATASET_ID 3d_fullres all --npz"
nnUNetv2_train $DATASET_ID 3d_fullres all --npz

echo "End time: $(date)"
echo "=== TRAINING JOB FINISHED ==="
