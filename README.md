# Lightweight ConvNeXt-ViT for Explainable Breast Cancer Histopathology Classification

PyTorch implementation of a lightweight ConvNeXt-ViT hybrid model for explainable breast cancer histopathology image classification.

## Overview

This repository contains the training, evaluation, reporting, and explainability code used for a ConvNeXt-ViT hybrid classifier on breast cancer histopathology images. The model is designed to combine the global context modeling of Vision Transformers with the local texture sensitivity of ConvNeXt-style depth-wise convolution.

The codebase supports:

- Fold-based BreakHis experiments for binary and multiclass classification.
- ConvNeXt-ViT, ViT-S, and ConvNeXt-tiny model configurations.
- Training and evaluation with YAML configuration files.
- Confusion matrices, ROC curves, per-class metrics, and fold summaries.
- Grad-CAM, Score-CAM, Eigen-CAM, attention maps, and agreement-map style visual analysis.
- LC25000 fold/config generation and runner utilities.

The manuscript associated with this repository evaluates the proposed model under a magnification-independent BreakHis protocol using 5-fold cross-validation.

## Proposed Model

The proposed ConvNeXt-ViT model replaces the feed-forward MLP stage inside a ViT-style transformer block with a ConvNeXt unit. Token embeddings are reshaped from a 1D sequence into a 2D representation before the ConvNeXt block, allowing depth-wise convolution to model local histopathology texture while multi-head self-attention preserves global tissue context.

The hybrid model uses four sequential ConvNeXt-ViT blocks. In the manuscript, the main comparators are ViT-S, which uses twelve transformer blocks, and ConvNeXt-tiny, a convolutional baseline. The hybrid design keeps a late convolutional feature map available, which enables class-discriminative CAM methods in addition to attention-map analysis.

![Proposed ConvNeXt-ViT model architecture](figures/ConvNeXt-ViT.png)

Figure: Architecture of the proposed ConvNeXt-ViT model.

The original TIFF version of this figure is kept at `figures/ConvNeXt-ViT.tif`. A PNG copy is included for reliable GitHub Markdown preview.

## Dataset and Experimental Protocol

The manuscript reports experiments on the BreakHis breast cancer histopathology dataset. BreakHis contains 7,909 RGB histopathology images from 82 patients, with images stored at 700 x 460 pixels across four magnification levels: 40x, 100x, 200x, and 400x.

The multiclass task uses eight tumor subclasses:

- Benign: adenosis, fibroadenoma, phyllodes tumor, tubular adenoma.
- Malignant: ductal carcinoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma.

The binary task uses benign versus malignant labels. The experiments use all magnification levels together in a magnification-independent setting.

The manuscript describes the following data protocol:

- BreakHis predefined 5-fold groups are used.
- The original test portion is split into validation and test subsets, giving an approximate 65:15:20 train/validation/test split.
- Training augmentation includes rotations at 0, 90, 180, and 270 degrees, mirroring on the x and y axes, and three overlapped square segmentations.
- Testing uses only the center segment with no additional augmentation.
- Images are resized to 224 x 224.

Manuscript-reported training settings:

| Parameter | Value |
| --- | --- |
| Max epochs | 50 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | Cosine |
| Minimum learning rate | 1e-6 |

Note: batch size may vary across repository configurations and model runners. Check the selected YAML file before running an experiment.

## Repository Structure

```text
.
|-- configs/                 # YAML experiment configurations
|   |-- convnext-vit/        # BreakHis ConvNeXt-ViT fold configs
|   |-- convnext/            # BreakHis ConvNeXt baseline fold configs
|   |-- vit-s/               # BreakHis ViT-S baseline fold configs
|   |-- binary/              # Binary-classification configs
|   `-- LC25000/             # LC25000 ConvNeXt-ViT, ConvNeXt, and ViT-S configs
|-- data/                    # Fold CSV manifests and data placeholders
|   |-- folds/               # BreakHis fold CSVs and class map
|   `-- LC25000/             # LC25000 fold CSVs and class map
|-- figures/                 # Project figures, including the proposed architecture
|-- notebooks/               # Notebooks, if used for analysis or experiments
|-- outputs/                 # Generated checkpoints, logs, predictions, and figures
|-- scripts/                 # PowerShell runners and post-processing scripts
|-- src/                     # Core package code
|   |-- datasets/            # Dataset, transform, loader, and fold-creation utilities
|   |-- models/              # ConvNeXt, ViT, and ConvNeXt-ViT model definitions
|   |-- train/               # Training loop, metrics, losses, checkpoints, and history
|   |-- utils/               # Config, device, seed, I/O, logger, and smoke helpers
|   |-- train_cli.py         # Training command-line entry point
|   |-- evaluate_cli.py      # Evaluation command-line entry point
|   `-- explainability*.py   # Explanation and panel-generation utilities
|-- requirements.txt         # Python dependencies
|-- test_env.py              # Environment validation helper
`-- README.md
```

## Installation

Create and activate a Python virtual environment from the repository root, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The checked-in dependency file includes CUDA-enabled PyTorch package versions. Adjust the PyTorch installation command if your local CUDA, CPU-only, or platform requirements differ.

You can run the environment helper after installation:

```powershell
python test_env.py
```

## Usage

Run commands from the repository root with the virtual environment activated.

### Train One Fold

```powershell
python -m src.train_cli --config configs/convnext-vit/breakhis_fold1.yaml --device cuda --seed 42 --smoke-train
```

### Evaluate One Fold

```powershell
python -m src.evaluate_cli --config configs/convnext-vit/breakhis_fold1.yaml --device cuda --seed 42 --checkpoint outputs/checkpoints/convnext-vit/breakhis_fold1_last.pt
```

If a test split is not available, the evaluation entry point falls back to the validation split.

### Resume Training

```powershell
python -m src.train_cli --config configs/convnext-vit/breakhis_fold1.yaml --device cuda --seed 42 --smoke-train --resume outputs/checkpoints/convnext-vit/breakhis_fold1_last.pt
```

### Run BreakHis Folds

Use the unified BreakHis runner:

```powershell
scripts\run_all_folds.ps1 -Device cuda -Model convnext-vit -Seed 42 -StartFold 1 -EndFold 5
```

Supported `-Model` values are:

- `convnext-vit`
- `vit-s`
- `convnext`

Useful flags:

- `-Binary` uses configurations under `configs/binary/`.
- `-SkipTrain` runs evaluation only.
- `-SkipEval` runs training only.
- `-Resume` resumes from the latest fold checkpoint when present.
- `-Deterministic` enables deterministic mode.
- `-ContinueOnError` continues after a failed fold.

### Run LC25000 Folds

Use the LC25000 runner:

```powershell
scripts\run_all_lc25000_folds.ps1 -Device cuda -Models convnext-vit,vit-s,convnext -Seed 42 -StartFold 1 -EndFold 5
```

Useful flags include `-SkipTrain`, `-SkipEval`, `-Deterministic`, `-KeepEpochCheckpoints`, and `-ContinueOnError`.

### Generate LC25000 Fold Files

The repository includes a helper script for creating LC25000 CSV folds and model YAML files:

```powershell
python scripts\create_lc25000_data.py --source-root E:\lung_colon_image_set --out-dir data\LC25000 --seed 42
```

Adjust `--source-root` to match your local dataset location.

## Results / Outputs

Training and evaluation artifacts are written under `outputs/` according to each YAML configuration.

Typical training outputs include:

- Checkpoints in `outputs/checkpoints/`
- Training histories in `outputs/logs/`

Typical evaluation outputs for experiment `<exp>` and split `<split>` include:

- `outputs/predictions/<exp>_<split>_metrics.json`
- `outputs/predictions/<exp>_<split>_metrics.csv`
- `outputs/predictions/<exp>_<split>_per_class_metrics.json`
- `outputs/predictions/<exp>_<split>_per_class_metrics.csv`
- `outputs/predictions/<exp>_<split>_roc_curves.json`
- `outputs/predictions/<exp>_<split>_confusion_matrix.csv`
- `outputs/predictions/<exp>_<split>_label_names.json`
- `outputs/predictions/<exp>_<split>_label_names.csv`
- `outputs/figures/<exp>_<split>_confusion_matrix.png`
- `outputs/figures/<exp>_<split>_roc.png`
- `outputs/figures/<exp>_training_validation_curves.png`, when a history CSV exists

### Manuscript-Reported BreakHis Results

The following values are reported in the manuscript for 5-fold cross-validation on BreakHis.

Binary classification:

| Metric | ViT-S | ConvNeXt-tiny | ConvNeXt-ViT |
| --- | --- | --- | --- |
| Accuracy | 98.9475 +/- 0.6504% | 99.9611 +/- 0.0870% | 99.5489 +/- 0.4831% |
| Precision | 98.5461 +/- 0.9669% | 99.9442 +/- 0.1247% | 99.3915 +/- 0.6627% |
| Recall | 99.0944 +/- 0.4953% | 99.9702 +/- 0.0666% | 99.6116 +/- 0.3775% |
| F1 Score | 98.8066 +/- 0.7352% | 99.9571 +/- 0.0959% | 99.4977 +/- 0.5249% |
| J | 98.1889 +/- 0.9905% | 99.9404 +/- 0.1332% | 99.2231 +/- 0.7551% |
| MCC | 97.6377 +/- 1.4411% | 99.9144 +/- 0.1913% | 99.0024 +/- 1.0357% |
| Kappa | 97.6136 +/- 1.4699% | 99.9142 +/- 0.1918% | 98.9954 +/- 1.0496% |
| TNR | 99.0944 +/- 0.4953% | 99.9702 +/- 0.0666% | 99.6116 +/- 0.3775% |
| AUC | 99.9691 +/- 0.0253% | 100.0000 +/- 0.0000% | 99.9927 +/- 0.0133% |

Multiclass classification:

| Metric | ViT-S | ConvNeXt | ConvNeXt-ViT |
| --- | --- | --- | --- |
| Accuracy | 94.5951 +/- 3.1257% | 98.3519 +/- 1.7459% | 97.1518 +/- 2.0973% |
| Precision | 93.5222 +/- 3.0498% | 98.1260 +/- 2.2967% | 96.6195 +/- 2.4764% |
| Recall | 96.7607 +/- 2.0567% | 99.2961 +/- 0.5466% | 98.9103 +/- 0.5307% |
| F1 Score | 94.7272 +/- 2.8336% | 98.5139 +/- 1.6200% | 97.4660 +/- 1.8855% |
| JI | 95.9861 +/- 2.4824% | 99.0620 +/- 0.7513% | 98.5149 +/- 0.7954% |
| MCC | 93.3462 +/- 3.7875% | 97.9626 +/- 2.1459% | 96.5144 +/- 2.5032% |
| Kappa | 93.1788 +/- 3.9674% | 97.8976 +/- 2.2294% | 96.3922 +/- 2.6595% |
| TNR | 99.2254 +/- 0.4447% | 99.7659 +/- 0.2287% | 99.6045 +/- 0.2723% |
| AUC | 99.8641 +/- 0.0841% | 100.0000 +/- 0.0000% | 99.9883 +/- 0.0049% |

Computational comparison reported in the manuscript:

| Parameter | ViT-S | ConvNeXt-tiny | ConvNeXt-ViT |
| --- | --- | --- | --- |
| Input size | 224 x 224 | 224 x 224 | 224 x 224 |
| Total parameters | 21.67 M | 27.83 M | 1.04 M |
| Trainable parameters | 21.67 M | 27.83 M | 1.04 M |
| Forward compute, batch = 1 | 8.48 GFLOPs | 8.91 GFLOPs | 0.36 GFLOPs |
| Forward compute, batch = 1 | 4.24 GMACs | 4.45 GMACs | 0.18 GMACs |
| Training compute per epoch | 1.95 PFLOPs | 2.05 PFLOPs | 0.082 PFLOPs |
| Total training compute | 97.7 PFLOPs | 103 PFLOPs | 4.12 PFLOPs |
| Checkpoint size | 248.17 MB | 318.68 MB | 12.05 MB |

## Citation

Please cite the manuscript associated with this repository:

```bibtex
@misc{hussein_convnext_vit_histopathology,
  title  = {Lightweight ConvNeXt-ViT for Explainable Breast Cancer Histopathology Classification},
  author = {Hussein, Mustafa Adil and Huddin, Aqilah Baseri and Hashim, Fazida Hanim and Alani, Ahmed Sameer},
  note   = {Manuscript in preparation}
}
```


## Funding

This work was supported by the Ministry of Higher Education, Malaysia, and The National University of Malaysia through the Fundamental Research Grant Scheme (FRGS), Grant No. `FRGS/1/2023/TK07/UKM/02/6`.

## License

This repository is provided as open access.
