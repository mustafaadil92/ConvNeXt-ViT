# ConvNeXt-ViT PyTorch

PyTorch training and evaluation pipeline for a ConvNeXt-ViT hybrid classifier on BreaKHis fold CSVs.

## Project Layout
- `src/` core code (`train_cli.py`, `evaluate_cli.py`, datasets, model, engine)
- `configs/` fold configs (`breakhis_fold1.yaml` ... `breakhis_fold5.yaml`)
- `scripts/` PowerShell runners
- `data/folds/` fold CSV manifests
- `outputs/` generated checkpoints, logs, predictions, and figures

## Fold Configs
All fold configs are aligned to the same settings as fold 1, with fold-specific CSV paths:
- `device: cuda`
- `data.pin_memory: false`
- `train.batch_size: 64`
- `train.epochs: 100`
- `model.num_layers : 4`

## Run Single Fold
From project root with virtual environment activated:

```powershell
# train (creates checkpoint + history csv)
python -m src.train_cli --config configs/breakhis_fold1.yaml --device cuda --seed 42 --smoke-train
```

```powershell
# evaluate on test split (falls back to val if test missing)
python -m src.evaluate_cli --config configs/breakhis_fold1.yaml --device cuda --seed 42 --checkpoint outputs/checkpoints/breakhis_fold1_last.pt

# resume training from last checkpoint
python -m src.train_cli --config configs/breakhis_fold1.yaml --device cuda --seed 42 --smoke-train --resume outputs/checkpoints/breakhis_fold1_last.pt
```

## Run All Folds
Use the unified runner:

```powershell
scripts\run_all_folds.ps1 -Device cuda -Seed 42 -StartFold 1 -EndFold 5
```

Optional flags:
- `-SkipTrain` evaluate only
- `-SkipEval` train only
- `-Deterministic` deterministic run mode
- `-ContinueOnError` continue if a fold fails

## Evaluation Metrics
Saved summary metrics include:
- `Accuracy`
- `Precision`
- `Recall`
- `F1 Score`
- `J`
- `MCC`
- `κ` (also `kappa`)
- `TNR`
- `AUC`
- plus `eval_loss`, `eval_acc`, `num_samples`

## Evaluation Outputs
For `<exp>` and split `<split>` (`test` by default):

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
- `outputs/figures/<exp>_training_validation_curves.png` (if history CSV exists)
