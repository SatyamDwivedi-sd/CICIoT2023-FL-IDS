# CICIoT2023 Baseline Results Interpretation

---

## 1. Binary ML Baseline (RF + XGBoost)

Results from `results/binary/ml_baseline_results_binary.csv`. Trained on imbalanced binary split; evaluated on imbalanced test set.

| Model         | Accuracy | F1 Macro | F1 Weighted | ROC-AUC | PR-AUC  | MCC    | FPR    | FNR    | Train (s) |
|---------------|----------|----------|-------------|---------|---------|--------|--------|--------|-----------|
| Random Forest | 0.9855   | 0.9238   | 0.9855      | 0.9960  | 0.9998  | 0.8475 | 0.1440 | 0.0077 | 153.9     |
| XGBoost       | 0.9720   | 0.8828   | 0.9749      | 0.9961  | 0.9998  | 0.7880 | 0.0018 | 0.0293 | 46.2      |

**Interpretation:**
- Both models achieve near-perfect ROC-AUC (0.9960–0.9961) and PR-AUC (0.9998), confirming the features are highly discriminative for binary detection.
- RF has higher F1 Macro (0.9238 vs 0.8828) and lower FNR (0.0077 vs 0.0293), meaning fewer missed attacks.
- XGBoost has dramatically lower FPR (0.0018 vs 0.1440), meaning far fewer false alarms.
- Neither model is strictly better: the choice depends on whether false alarms (FPR) or missed attacks (FNR) are more costly for deployment.

---

## 2. Binary DL Baseline (MLP + 1D CNN)

Results from `results/binary/dl_baseline_results_binary.csv`. PyTorch, 10 epochs, batch 4096, lr 1e-4.

| Model  | Accuracy | F1 Macro | F1 Weighted | ROC-AUC | PR-AUC  | MCC    | FPR    | FNR    | Train (s) |
|--------|----------|----------|-------------|---------|---------|--------|--------|--------|-----------|
| MLP    | 0.9520   | 0.8244   | 0.9591      | 0.9883  | 0.9994  | 0.6953 | 0.0003 | 0.0505 | 636.9     |
| 1D CNN | 0.9529   | 0.8270   | 0.9599      | 0.9897  | 0.9995  | 0.6992 | 0.0002 | 0.0495 | 707.6     |

**Interpretation:**
- MLP and 1D CNN perform almost identically; the slight CNN edge (F1 Macro 0.8270 vs 0.8244, MCC 0.6992 vs 0.6953) does not justify the extra 70s training time in isolation.
- DL models achieve extremely low FPR (0.0003 and 0.0002) — the best of all four binary models — at the cost of higher FNR (~0.05).
- ROC-AUC and PR-AUC are slightly lower than ML models (0.988–0.990 vs 0.996), likely because DL models were trained for only 10 epochs with no class weighting.

---

## 3. Operating Point Trade-off (FPR vs FNR)

The four binary models occupy distinct operating points on the FPR/FNR surface:

| Model         | FPR    | FNR    | Profile                        |
|---------------|--------|--------|--------------------------------|
| Random Forest | 0.1440 | 0.0077 | Low miss rate, high alarm rate |
| XGBoost       | 0.0018 | 0.0293 | Balanced                       |
| MLP           | 0.0003 | 0.0505 | Very low alarms, misses ~5%    |
| 1D CNN        | 0.0002 | 0.0495 | Very low alarms, misses ~5%    |

**For FL-IDS Stage 2 and 3:**
- RF's FPR of 0.1440 is problematic for federated clients — high false-alarm rates inflate update volumes and can mask poisoned gradients in the NIMA attack scenario.
- XGBoost is the recommended binary baseline to carry into Stage 2: balanced FPR/FNR, fastest training (46s), and best generalization evidence (ROC-AUC 0.9961).
- DL models are useful for CA-FedDef confidence threshold calibration due to their probability outputs.

---

## 4. Multiclass Baseline Status (8-Category)

Multiclass result CSVs have not yet been generated — `scripts/06` and `scripts/07` must be run after the fixes below. Interim figures are from notebook outputs.

**Notebook-reported results (pre-fix, for reference only):**

| Model    | Accuracy | F1 Macro | F1 Weighted | MCC   |
|----------|----------|----------|-------------|-------|
| MLP      | 0.7863   | 0.6274   | 0.8001      | —     |
| 1D CNN   | —        | ~0.627   | —           | —     |

**Root cause of weak F1 Macro (~0.63):** Training data was SMOTE-balanced (Web→100K, BruteForce→50K) but test Web count is only 4,695 and BruteForce 2,524. Models over-predict these minority classes → precision collapses to ~0.05–0.15 for Web/BruteForce. Fixes applied (see Section 5) address this at the modelling level.

**Fixes applied before rerunning:**
1. RF `class_weight` removed (`None`) — data is already SMOTE-balanced, double-weighting hurts precision.
2. XGBoost `objective` changed from `multi:softmax` to `multi:softprob` — required for Stage 2 confidence-based FL thresholds.
3. DL script `--use-class-weights` flag removed — matches notebook behaviour exactly.

---

## 5. Key Weaknesses and Next Steps

| Weakness | Affected Models | Recommended Fix |
|---|---|---|
| RF FPR = 0.144 on binary | RF | Use XGBoost or DL for deployment baseline |
| F1 Macro ~0.63 on cat8 | All multiclass | Rerun scripts/06 + 07 after fixes; consider focal loss or label-smoothing |
| Web precision ~0.05–0.10 | MLP, CNN, RF | Adjust SMOTE targets; evaluate test-time threshold tuning |
| No fine-34 baseline yet | — | Run scripts/08 (fine34) once cat8 baseline is stable |
| DL binary: 10 epochs only | MLP, CNN | Extend to 20 epochs with ReduceLROnPlateau for fairer comparison |

---

## 6. Summary Table — All Baselines

| Task    | Model         | Accuracy | F1 Macro | MCC    | FPR    | FNR    |
|---------|---------------|----------|----------|--------|--------|--------|
| Binary  | Random Forest | 0.9855   | 0.9238   | 0.8475 | 0.1440 | 0.0077 |
| Binary  | XGBoost       | 0.9720   | 0.8828   | 0.7880 | 0.0018 | 0.0293 |
| Binary  | MLP           | 0.9520   | 0.8244   | 0.6953 | 0.0003 | 0.0505 |
| Binary  | 1D CNN        | 0.9529   | 0.8270   | 0.6992 | 0.0002 | 0.0495 |
| Cat8    | RF            | —        | —        | —      | —      | —      |
| Cat8    | XGBoost       | —        | —        | —      | —      | —      |
| Cat8    | MLP (Keras)   | 0.7863*  | 0.6274*  | —      | —      | —      |
| Cat8    | 1D CNN (Keras)| —        | ~0.627*  | —      | —      | —      |

*Notebook output before script fixes — rerun `scripts/06_train_multiclass_ml.py` and `scripts/07_train_multiclass_dl.py` to populate missing rows.
