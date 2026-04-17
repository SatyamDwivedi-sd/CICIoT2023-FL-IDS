# Pipeline Output Map

Canonical mapping of each notebook to its outputs and authoritative storage location.

**Mental model:**

| Directory | Role |
|-----------|------|
| `extracted/` | Official data artifacts — splits, scalers, encoders |
| `results/binary/` | Official binary pipeline results |
| `models/binary/` | Official binary pipeline models |
| `eda_outputs/` | Notebooks + historical/Colab notebook outputs (not authoritative) |
| `results/cat8/` | Intended multiclass results — **empty until cluster jobs complete** |
| `models/cat8/` | Intended multiclass models — **empty until cluster jobs complete** |

---

## nb01 — 01_EDA.ipynb
**Run environment:** Local

**Outputs → `eda_outputs/`**
- `category_distribution.png`
- `feature_statistics.csv`

**Outputs → `extracted/`**
- `ciciot_clean.csv` — deduplicated cleaned dataset (21 M rows, 4.7 GB)

---

## nb02 — 02_Preprocessing.ipynb
**Run environment:** Local

**Outputs → `extracted/`**

*Feature splits (binary task):*
- `X_train_binary.csv`, `X_val_binary.csv`, `X_test_binary.csv`

*Feature splits (8-category task):*
- `X_train_cat8.csv`, `X_val_cat8.csv`, `X_test_cat8.csv`

*Label splits:*
- `y_train_binary.csv`, `y_val_binary.csv`, `y_test_binary.csv`
- `y_train_cat8.csv`, `y_val_cat8.csv`, `y_test_cat8.csv`
- `y_train_fine34.csv`, `y_val_fine34.csv`, `y_test_fine34.csv`
  *(fine34 is labels-only — no separate X splits; fine34 shares X_*_cat8 feature files)*

*Preprocessing artifacts:*
- `scaler_binary.pkl`, `scaler_cat8.pkl`
- `label_encoder_fine34.pkl`
- `cat8_mapping.json`, `fine34_mapping.json`, `feature_columns.json`
- `split_summary.json`

---

## nb03 — 03_Baseline_ML.ipynb
**Run environment:** Local

**Outputs → `results/binary/`** *(authoritative)*
- `rf_confusion_matrix.png`, `xgb_confusion_matrix.png`
- `ml_baseline_results_binary.csv`
- `ml_baseline_comparison_binary.png`

**Outputs → `models/binary/`** *(authoritative)*
- `rf_model.pkl`, `xgb_model.pkl`

---

## nb04 — 04_Baseline_DL.ipynb
**Run environment:** Local

**Outputs → `results/binary/`** *(authoritative)*
- `dl_confusion_matrices_binary.png`
- `dl_baseline_results_binary.csv`
- `dl_baseline_comparison_binary.png`

**Outputs → `models/binary/`** *(authoritative)*
- `mlp_model.keras`, `cnn_model.keras`
  *(also: `mlp_best.keras`, `cnn_best.keras` — checkpoint files written by scripts/05_train_binary_dl.py, not nb04)*

---

## nb05 — 05_Class_Balancing.ipynb
**Run environment:** Local

**Outputs → `eda_outputs/`**
- `category_balancing_strategy.png`
- `category_balancing_updated.png`

---

## nb06 — 06_Balancing_Multiclass.ipynb
**Run environment:** Google Colab

**Outputs → Google Drive** (`MyDrive/CICIoT2023_Research/`)
- `X_train_balanced_8cat_smoteenn.csv` ← **Drive only; not downloaded locally**
- `y_train_balanced_8cat_smoteenn.csv` ← **Drive only; not downloaded locally**
- `category_encoding_8cat.json`
- `balanced_class_counts_8cat_smoteenn.json`
- `balancing_summary_8cat_smoteenn.json`

> **Filename mismatch note:** Colab saves as `X_train_balanced_8cat_smoteenn.csv`.
> The local pipeline (`scripts/06`, `07`) and cluster SLURM jobs expect `X_train_balanced_cat8.csv`.
> `slurm/run_03_balance.sh` (scripts/03_balance.py) must run on the cluster first —
> it reads the unbalanced cat8 splits and produces the correctly-named file.

---

## nb07 — 07_Baseline_Multiclass_ML.ipynb
**Run environment:** Google Colab

**Outputs → Google Drive** (`MyDrive/CICIoT2023_Research/results_cat8_ml/`)
- `cat8_ml_results.csv`
- `cat8_ml_confusion_matrices.png`

**Outputs → Google Drive** (`MyDrive/CICIoT2023_Research/models_cat8_ml/`)
- `rf_cat8.pkl`, `xgb_cat8.pkl` ← **Drive only; not in local models/**

**Manually downloaded → `eda_outputs/`** *(historical, not authoritative)*
- `results_ml_baseline_8cat_imbalanced_test.csv`
- `rf_confusion_matrix_test.png`, `xgb_confusion_matrix_test.png`
- `rf_feature_importance.png`, `xgb_feature_importance.png`
- `ml_baseline_f1_comparison.png`
- `experiment_log_07_ml_baseline.json`

**Authoritative destination (cluster):** `results/cat8/` and `models/cat8/`
produced by `slurm/run_06_multiclass_ml.sh` → `scripts/06_train_multiclass_ml.py`

---

## nb08 — 08_Baseline_Multiclass_DL.ipynb
**Run environment:** Google Colab

**Outputs → Google Drive** (`MyDrive/CICIoT2023_Research/results_cat8_dl/`)
- `cat8_dl_results.csv`
- `cat8_dl_confusion_matrices.png`
- `cat8_dl_training_history.json`

**Outputs → Google Drive** (`MyDrive/CICIoT2023_Research/models_cat8_dl/`)
- `mlp_cat8_best.keras`, `mlp_cat8_final.keras` ← **Drive only; not in local models/**
- `cnn_cat8_best.keras`, `cnn_cat8_final.keras` ← **Drive only; not in local models/**

**Manually downloaded → `eda_outputs/`** *(historical, not authoritative)*
- `results_dl_baseline_8cat_imbalanced_test.csv`
- `mlp_confusion_matrix_test.png`, `cnn_confusion_matrix_test.png`, `dl_confusion_matrices.png`
- `dl_training_curves.png`, `dl_baseline_f1_comparison.png`
- `full_baseline_comparison_v2.png`, `full_baseline_results_v2.csv`
- `experiment_log_08_dl_baseline.json`

**Authoritative destination (cluster):** `results/cat8/` and `models/cat8/`
produced by `slurm/run_07_multiclass_dl.sh` → `scripts/07_train_multiclass_dl.py`

---

## Pending cluster jobs

The following must run on the Cyber Innovation Cluster to complete Stage 1:

```
# Step 1 — balance training data (creates X_train_balanced_cat8.csv)
BALANCE_JOB=$(sbatch --parsable slurm/run_03_balance.sh)

# Step 2 — multiclass baselines (depend on step 1)
sbatch --dependency=afterok:$BALANCE_JOB slurm/run_06_multiclass_ml.sh
sbatch --dependency=afterok:$BALANCE_JOB slurm/run_07_multiclass_dl.sh
```

After these complete, `results/cat8/` and `models/cat8/` will contain the
authoritative multiclass results, superseding the Colab/Drive outputs.
