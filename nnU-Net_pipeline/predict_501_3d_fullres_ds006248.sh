#!/bin/bash
#
# SLURM Script for nnU-Net v2 Prediction (OpenNeuro ds006248, Dataset 501, 3d_fu
llres)
#
#SBATCH --job-name=nnunet_pred_501_3d
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/openeuro_dataset/nnunet_pred_501_3d_%j.out
#SBATCH --error=/home/openeuro_dataset/nnunet_pred_501_3d_%j.err

echo "=== nnU-Net prediction job started ==="
echo "Hostname: $(hostname)"
echo "Start time: $(date)"

module load cuda/12.2 2>/dev/null || true
source ~/.bashrc
conda activate nnunet_new

PROJECT_ROOT="/home/fdv119/openeuro_dataset"

export nnUNet_raw="$PROJECT_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_ROOT/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_ROOT/nnUNet_results"

# (Optional, harmless)
export NNUNET_RAW_DATA="$nnUNet_raw"
export NNUNET_PREPROCESSED="$nnUNet_preprocessed"
export NNUNET_RESULTS="$nnUNet_results"

DATASET_NAME="Dataset501_Pituitary"
INP="$nnUNet_raw/$DATASET_NAME/imagesTs"
OUT="$PROJECT_ROOT/predsTs_501_3d_fullres_ds006248"

# --- SAFETY CHECKS ---
echo "Input folder:  $INP"
if [ ! -d "$INP" ]; then
  echo "❌ Input folder does not exist: $INP"
  exit 1
fi

echo "Number of test images:"
ls -1 "$INP"/*.nii.gz 2>/dev/null | wc -l

# Clean output folder so you can tell if it worked
rm -rf "$OUT"
mkdir -p "$OUT"

echo "Output folder: $OUT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# More stable settings on clusters
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export nnUNet_n_proc_DA=4
export nnUNet_num_proc_DA=4

# --- RUN PREDICTION ---
# Note: -f all uses the 5 CV folds (ensemble). Good default.
echo "Running: nnUNetv2_predict -d 501 -c 3d_fullres -f all"
nnUNetv2_predict \
  -d 501 \
  -i "$INP" \
  -o "$OUT" \
  -c 3d_fullres \
  -f all

echo "=== Listing outputs ==="
ls -lh "$OUT" | head -n 50
echo "Total predicted files:"
ls -1 "$OUT"/*.nii.gz 2>/dev/null | wc -l

echo "End time: $(date)"
echo "=== Job finished ==="
