# 🧠 Pituitary Adenoma Segmentation in MRI using nnU-Net v2

Automated segmentation of pituitary adenomas from MRI scans using deep learning.
This project evaluates the **nnU-Net v2 framework** across heterogeneous datasets and investigates strategies to improve **cross-dataset generalization**.

---

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Framework](https://img.shields.io/badge/framework-nnU--Net%20v2-orange)
![DL](https://img.shields.io/badge/deep%20learning-PyTorch-red)
![Data](https://img.shields.io/badge/data-MRI%20NIfTI-green)
![Compute](https://img.shields.io/badge/compute-HPC%20SLURM-purple)
![Status](https://img.shields.io/badge/status-Completed-brightgreen)

---

## 📌 Project Overview

This project investigates **automated pituitary adenoma segmentation from MRI** using the **nnU-Net v2 deep learning framework** (Isensee et al., 2021). The study evaluates segmentation accuracy against **expert annotations** and explores strategies such as **data augmentation and dataset merging** to improve performance across heterogeneous clinical datasets.

> Developed as part of a **Master's Thesis in Bioinformatics** at the University of Copenhagen.

---

## 🧬 Clinical Motivation

| Challenge | Description |
|---|---|
| **Manual segmentation** | Time-consuming and labor-intensive |
| **Observer variability** | High variability between clinicians |
| **Anatomical complexity** | Small gland size and unclear tumor boundaries |
| **Clinical importance** | Accurate segmentation needed for surgical planning and monitoring |

Deep learning offers the potential for **reproducible, scalable, and faster** segmentation workflows.

---

## 🎯 Research Questions

| ID | Research Question |
|---|---|
| **RQ1 — Accuracy** | How accurately can deep learning segment pituitary adenomas compared to expert annotations? |
| **RQ2 — Heterogeneity** | How can segmentation models better handle differences in scanners, protocols, and image quality? |

<p align="center">
  <img src="assets/workflow.png" alt="Project Workflow" width="700">
</p>

---

## 📊 Datasets

| Dataset | MRI Sequences | Patients | Dimensionality | Resolution | Scanner | Format |
|---|---|---|---|---|---|---|
| **Cheng et al. (2017)** | T1 CE | 233 | 2D | 512×512 | NR | Image slices |
| **Pandit et al. (2025)** | T1 CE, T2 | 128 | 3D | 512×512×114 | Siemens 3T | NIfTI |
| **Černý et al. (2025)** | T1 CE, T2, Diffusion | 50 | 3D | 512×512×336 | GE 3T | NIfTI |

These datasets differ in **scanner hardware, acquisition protocols, and annotation strategies**, enabling evaluation of model robustness under **domain shift**.

---

## 🔄 Project Pipeline

```
Dataset Preparation → nnU-Net Preprocessing → Model Training → Inference → Evaluation
```

---

## ⚙️ nnU-Net Workflow

<p align="center">
  <img src="assets/workflow_1.png" alt="nnU-Net Workflow" width="700">
</p>

| Stage | Purpose | Command |
|---|---|---|
| **Preprocessing** | Prepare and normalize MRI data | `nnUNetv2_plan_and_preprocess` |
| **Training** | Train segmentation model (5-fold CV) | `nnUNetv2_train DATASET_ID 3d_fullres` |
| **Inference** | Generate predictions | `nnUNetv2_predict` |
| **Evaluation** | Compare predictions vs. ground truth | Custom evaluation scripts |

### Training Configuration

| Parameter | Value |
|---|---|
| Architecture | 3D full resolution nnU-Net |
| Optimizer | Adam |
| Loss function | Dice + Cross-Entropy |
| Cross-validation | 5 folds |

---

## 📈 Evaluation Metrics

| Metric | Formula | Measures | Interpretation |
|---|---|---|---|
| **Dice (DSC)** | 2TP / (2TP + FP + FN) | Overlap between prediction and ground truth | Higher = better |
| **IoU** | TP / (TP + FP + FN) | Intersection over union | Higher = better |
| **Precision** | TP / (TP + FP) | Correctness of tumor predictions | High → few false positives |
| **Recall** | TP / (TP + FN) | Tumor detection ability | High → few missed tumors |
| **Specificity** | TN / (TN + FP) | Background classification | High → few background errors |
| **RVD** | (V_pred − V_gt) / V_gt | Tumor volume error | 0 = perfect |
| **HD95** | 95th percentile Hausdorff distance | Boundary mismatch | Lower = better |
| **ASD** | Mean surface distance | Average contour error | Lower = better |

---

## 🗂 Dataset Structure

The project follows the **nnU-Net dataset format**:

```
nnUNet_raw/
└── DatasetXXX_Pituitary/
    ├── imagesTr/          # Training MRI volumes
    ├── labelsTr/          # Ground truth segmentation masks
    ├── imagesTs/          # Test MRI images
    ├── labelsTs/          # Reference annotations
    └── dataset.json       # Dataset metadata
```

All images are stored in **NIfTI format** (`.nii.gz`).

---

## ⚙️ Technical Stack

| Category | Tools |
|---|---|
| **Programming** | Python, Bash |
| **Deep Learning** | PyTorch, nnU-Net v2 |
| **Medical Imaging** | NiBabel |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Computing** | HPC GPU cluster (SLURM) |

---

## 💻 Computing Environment

| Resource | Configuration |
|---|---|
| **GPU** | NVIDIA GPU |
| **CUDA** | 12.x |
| **CPU** | 8 threads |
| **Memory** | 20–24 GB |
| **Runtime** | 6–24 hours per training |

<details>
<summary>Example SLURM job configuration</summary>

```bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
```

</details>

---

## 📦 Installation

### Requirements

- Python ≥ 3.10
- PyTorch (with CUDA support)
- nnU-Net v2

### Setup

```bash
conda create -n nnunet python=3.10
conda activate nnunet

pip install nnunetv2
pip install nibabel numpy pandas matplotlib seaborn
```

---

## ▶️ Running the Pipeline

```bash
# 1. Preprocess data
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

# 2. Train models (5-fold cross-validation)
nnUNetv2_train DATASET_ID 3d_fullres FOLD

# 3. Generate predictions
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres

# 4. Evaluate results
python evaluate.py --pred OUTPUT_FOLDER --gt LABELS_FOLDER
```

---

## 👥 Authors & Affiliation

| Role | Name | Affiliation |
|---|---|---|
| 🎓 Master's Student | **Manuela Del Castillo** | MSc Bioinformatics, University of Copenhagen |
| 🧑‍🏫 Supervisor | **Mostafa Mehdipour Ghazi** | Assistant Professor, Department of Computer Science, University of Copenhagen |

---

## 📄 License

*To be added.*

---

## 🙏 Acknowledgements

*To be added.*

---

> ✨ Developed as a **Master's Thesis in Bioinformatics** at the **University of Copenhagen**
