# Stage 1 — Multiclass Baseline Results (8-Category)

**Dataset:** CICIoT2023  
**Task:** 8-class network intrusion detection  
**Training set:** SMOTE-ENN balanced (~1.6M rows after balancing)  
**Test set:** Original imbalanced distribution (4,201,031 samples)  
**Run date:** 2026-04-17 (Cyber Innovation Cluster, node c1)

---

## Per-Class Classification Report

### Precision

| Class | Support | Random Forest | XGBoost | MLP | 1D CNN |
|-------|--------:|:-------------:|:-------:|:---:|:------:|
| DDoS | 2,458,413 | 0.929 | 0.935 | 0.928 | 0.924 |
| DoS | 835,769 | 0.530 | 0.532 | 0.473 | 0.458 |
| Mirai | 471,837 | 0.991 | 0.993 | 0.982 | 0.986 |
| Benign | 209,462 | 0.789 | 0.792 | 0.736 | 0.697 |
| Recon | 131,093 | 0.854 | 0.860 | 0.820 | 0.807 |
| Spoofing | 87,212 | 0.950 | 0.953 | 0.729 | 0.734 |
| Web | 4,741 | 0.098 | 0.107 | 0.038 | 0.038 |
| BruteForce | 2,504 | 0.231 | 0.229 | 0.218 | 0.219 |

### Recall

| Class | Support | Random Forest | XGBoost | MLP | 1D CNN |
|-------|--------:|:-------------:|:-------:|:---:|:------:|
| DDoS | 2,458,413 | 0.748 | 0.746 | 0.678 | 0.661 |
| DoS | 835,769 | 0.832 | 0.846 | 0.843 | 0.837 |
| Mirai | 471,837 | 0.999 | 0.999 | 0.997 | 0.994 |
| Benign | 209,462 | 0.908 | 0.911 | 0.836 | 0.853 |
| Recon | 131,093 | 0.580 | 0.593 | 0.465 | 0.473 |
| Spoofing | 87,212 | 0.811 | 0.831 | 0.516 | 0.411 |
| Web | 4,741 | 0.531 | 0.513 | 0.482 | 0.417 |
| BruteForce | 2,504 | 0.441 | 0.452 | 0.320 | 0.288 |

### F1-Score

| Class | Support | Random Forest | XGBoost | MLP | 1D CNN |
|-------|--------:|:-------------:|:-------:|:---:|:------:|
| DDoS | 2,458,413 | 0.829 | 0.830 | 0.783 | 0.771 |
| DoS | 835,769 | 0.648 | 0.653 | 0.606 | 0.592 |
| Mirai | 471,837 | **0.995** | **0.996** | **0.989** | **0.990** |
| Benign | 209,462 | 0.844 | 0.847 | 0.783 | 0.768 |
| Recon | 131,093 | 0.691 | 0.702 | 0.593 | 0.596 |
| Spoofing | 87,212 | 0.875 | 0.888 | 0.604 | 0.527 |
| Web | 4,741 | 0.165 | 0.177 | 0.071 | 0.070 |
| BruteForce | 2,504 | 0.303 | 0.304 | 0.260 | 0.248 |

---

## Overall Summary

| Metric | Random Forest | XGBoost | MLP | 1D CNN |
|--------|:-------------:|:-------:|:---:|:------:|
| Accuracy | 0.7966 | 0.7989 | 0.7439 | 0.7316 |
| F1 Macro | 0.6687 | 0.6746 | 0.5861 | 0.5701 |
| F1 Weighted | 0.8079 | 0.8101 | 0.7604 | 0.7480 |
| MCC | — | — | 0.6385 | 0.6239 |
| Train Time | 66 s | 12 s | 57 s | 924 s |

---

## Train vs Test Metrics

Values from experiment logs (ground truth files). Gap = train_f1_macro − test_f1_macro.

| Model | Train Acc | Train F1-M | Train F1-W | Test Acc | Test F1-M | Test F1-W | Gap (F1-M) |
|-------|:---------:|:----------:|:----------:|:--------:|:---------:|:---------:|:----------:|
| Random Forest | 1.000 | 1.000 | 1.000 | 0.7966 | 0.6687 | 0.8079 | **0.331** |
| XGBoost | 0.9825 | 0.9701 | 0.9824 | 0.7989 | 0.6746 | 0.8101 | **0.296** |
| MLP | 0.9160 | 0.8266 | 0.9138 | 0.7439 | 0.5861 | 0.7604 | **0.241** |
| 1D CNN | 0.8971 | 0.7940 | 0.8930 | 0.7316 | 0.5701 | 0.7480 | **0.224** |

- **RF is fully overfit on training data** (train F1-macro = 1.000) — decision trees memorise the balanced training set exactly. The 0.331 gap is the largest of all models and confirms RF's high variance on this task.
- **XGBoost generalises better than RF** (gap = 0.296) due to gradient boosting's regularisation (shrinkage, subsampling), but still shows substantial gap — the balanced→imbalanced distribution shift is the dominant cause, not classical overfitting.
- **DL models show the healthiest gaps** (MLP 0.241, CNN 0.224) — Dropout and BatchNorm act as strong regularisers, preventing memorisation of the balanced training distribution. The smaller gap suggests DL learns more transferable representations.
- **Train/test distribution shift is the primary driver of all gaps**: models are trained on SMOTE-ENN balanced data (~200K/class) but tested on the original imbalanced distribution (DDoS = 58.5%, Web = 0.11%). Even a perfectly calibrated model would show a gap purely from this shift.

---

## Notes

- **Training:** SMOTE-ENN balanced splits; `class_weight=None` (no double-weighting).  
  **Testing:** Original imbalanced distribution — metrics reflect real-world deployment conditions.
- **ML > DL** on this tabular dataset — consistent with published literature on network intrusion detection.
- **XGBoost** is the best single model: highest F1-macro (0.675), fastest ML training (12 s).
- **Web** and **BruteForce** are persistently weak across all models — extreme minority classes  
  (Web = 0.11%, BruteForce = 0.06% of test set) even after SMOTE-ENN oversampling.
- **DoS** shows high recall but low precision — models over-predict this class,  
  likely due to feature overlap with DDoS traffic.
- **1D CNN** train time (924 s) reflects 15 epochs × CPU-only TensorFlow (no GPU for Keras in this run), plus `ValF1Callback` running a full validation prediction pass after every epoch.
- These results serve as the **centralized baseline** for Stage 2 (Federated Learning simulation).
