# 🔀 Data Augmentation Strategies

This folder contains custom nnU-Net trainer variants used to evaluate the impact of **data augmentation intensity** on pituitary adenoma segmentation performance.

---

## 📌 Overview

To improve model generalization across variations in pituitary MRI data, two custom augmentation strategies were implemented by extending the default nnU-Net training pipeline. Rather than introducing new augmentation types, the **strength and frequency** of the existing nnU-Net augmentations were modified.

All strategies were evaluated on the **Pandit et al. (2025)** dataset.

---

## 🧪 Augmentation Strategies

Three augmentation configurations were compared:

| Augmentation Variant | Rotation Range Scaling | Noise / Blur / Gamma Probability | Bias Field Probability |
|---|---|---|---|
| **Default** (nnU-Net baseline) | ×1.0 | +0.00 | +0.00 |
| **Mild** (`nnUNetTrainer_MildAug`) | ×1.15 | +0.10 | +0.05 |
| **Strong** (`nnUNetTrainer_StrongAug`) | ×1.5 | +0.20 | +0.20 |

> All values are **relative to the default nnU-Net augmentation**. Rotation scaling is a multiplicative factor applied to the default rotation range. Probability values are additive increases to the probability of applying the corresponding intensity transformation.

---

## 🧠 Rationale

| Strategy | Goal |
|---|---|
| **Default** | Baseline — nnU-Net's auto-configured augmentation based on data characteristics |
| **Mild** | Conservative increase — safer for small structures like pituitary tumors, where aggressive augmentation could distort relevant anatomy |
| **Strong** | More aggressive — aims to expose the model to greater variability in intensity and geometry, potentially improving robustness under domain shift |

---

## 📂 File Structure

```
nnunetv2/training/nnUNetTrainer/
├── nnUNetTrainer.py                 # Default nnU-Net trainer (baseline)
├── nnUNetTrainer_MildAug.py         # Mild augmentation strategy
├── nnUNetTrainer_StrongAug.py       # Strong augmentation strategy
└── ...
```

---

## ⚙️ Implementation Details

Both custom trainers inherit from `nnUNetTrainer` and override two methods:

### `configure_rotation_dummyDA_mirroring_and_inital_patch_size()`

Widens the default rotation ranges by a multiplicative factor:

- **Mild**: ×1.15
- **Strong**: ×1.5

### `get_training_transforms()`

Increases `p_per_sample` for existing intensity transforms in the default pipeline:

| Transform | Mild (+) | Strong (+) |
|---|---|---|
| Gamma | +0.10 | +0.20 |
| Gaussian Noise | +0.10 | +0.20 |
| Gaussian Blur | +0.10 | +0.20 |
| Bias Field | +0.05 | +0.20 |

All probabilities are capped at 1.0.

---

## ▶️ Usage

To train with a custom augmentation strategy, specify the trainer class when calling `nnUNetv2_train`:

```bash
# Default augmentation
nnUNetv2_train DATASET_ID 3d_fullres FOLD

# Mild augmentation
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainer_MildAug

# Strong augmentation
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainer_StrongAug
```

---

## 📚 Reference

- Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203–211.
