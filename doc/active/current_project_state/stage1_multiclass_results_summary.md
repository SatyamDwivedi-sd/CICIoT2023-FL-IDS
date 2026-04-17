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
| DDoS | 2,458,413 | 0.929 | 0.935 | 0.931 | 0.928 |
| DoS | 835,769 | 0.530 | 0.532 | 0.466 | 0.465 |
| Mirai | 471,837 | 0.991 | 0.993 | 0.984 | 0.981 |
| Benign | 209,462 | 0.789 | 0.792 | 0.744 | 0.721 |
| Recon | 131,093 | 0.854 | 0.860 | 0.804 | 0.816 |
| Spoofing | 87,212 | 0.950 | 0.953 | 0.729 | 0.762 |
| Web | 4,741 | 0.098 | 0.107 | 0.038 | 0.041 |
| BruteForce | 2,504 | 0.231 | 0.229 | 0.201 | 0.173 |

### Recall

| Class | Support | Random Forest | XGBoost | MLP | 1D CNN |
|-------|--------:|:-------------:|:-------:|:---:|:------:|
| DDoS | 2,458,413 | 0.748 | 0.746 | 0.665 | 0.668 |
| DoS | 835,769 | 0.832 | 0.846 | 0.852 | 0.844 |
| Mirai | 471,837 | 0.999 | 0.999 | 0.996 | 0.997 |
| Benign | 209,462 | 0.908 | 0.911 | 0.827 | 0.850 |
| Recon | 131,093 | 0.580 | 0.593 | 0.476 | 0.482 |
| Spoofing | 87,212 | 0.811 | 0.831 | 0.538 | 0.487 |
| Web | 4,741 | 0.531 | 0.513 | 0.474 | 0.440 |
| BruteForce | 2,504 | 0.441 | 0.452 | 0.338 | 0.344 |

### F1-Score

| Class | Support | Random Forest | XGBoost | MLP | 1D CNN |
|-------|--------:|:-------------:|:-------:|:---:|:------:|
| DDoS | 2,458,413 | 0.829 | 0.830 | 0.776 | 0.777 |
| DoS | 835,769 | 0.648 | 0.653 | 0.602 | 0.600 |
| Mirai | 471,837 | **0.995** | **0.996** | **0.990** | **0.989** |
| Benign | 209,462 | 0.844 | 0.847 | 0.784 | 0.780 |
| Recon | 131,093 | 0.691 | 0.702 | 0.598 | 0.606 |
| Spoofing | 87,212 | 0.875 | 0.888 | 0.619 | 0.594 |
| Web | 4,741 | 0.165 | 0.177 | 0.071 | 0.074 |
| BruteForce | 2,504 | 0.303 | 0.304 | 0.252 | 0.230 |

---

## Overall Summary

| Metric | Random Forest | XGBoost | MLP | 1D CNN |
|--------|:-------------:|:-------:|:---:|:------:|
| Accuracy | 0.7966 | 0.7989 | 0.7387 | 0.7388 |
| F1 Macro | 0.6687 | 0.6746 | 0.5864 | 0.5812 |
| F1 Weighted | 0.8079 | 0.8101 | 0.7559 | 0.7552 |
| MCC | — | — | 0.6353 | 0.6339 |
| Train Time | 66 s | 12 s | 57 s | 924 s |

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
